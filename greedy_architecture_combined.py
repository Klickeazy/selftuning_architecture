import numpy as np
import networkx as netx
import random
import time
import scipy
from copy import deepcopy as dc

import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
# from matplotlib.ticker import MaxNLocator
import matplotlib.animation

matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['legend.title_fontsize'] = 10
matplotlib.rcParams['legend.framealpha'] = 0.5
matplotlib.rcParams['lines.markersize'] = 5
matplotlib.rcParams['image.cmap'] = 'Blues'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'


class System:
    def __init__(self, graph_model=None, architecture=None, additive=None, initial_conditions=None):
        self.dynamics = {}
        if graph_model is None:
            graph_model = {'number_of_nodes': 10, 'type': 'rand', 'self_loop': True, 'rho': 1}
        self.graph_initialize(graph_model)
        self.architecture = {'B': {}, 'C': {}}
        self.metric_model = {'type1': 'x', 'type2': 'scalar', 'type3': 'scalar'}
        self.trajectory = {'X0': np.identity(self.dynamics['number_of_nodes']),
                           'x': [], 'x_estimate': [], 'y': [], 'u': [], 'enhanced': [],
                           'cost': {'running': 0, 'switching': 0, 'true_stage': 0,
                                    'estimate_stage': 0, 'estimate_total': [], 'true_total': []},
                           'P': [], 'E': []}
        self.additive = {'W': np.identity(self.dynamics['number_of_nodes']), 'V': np.identity(self.dynamics['number_of_nodes'])}

        self.initialize_initial_conditions(initial_conditions)
        self.noise = {}
        self.initialize_architecture(architecture)
        if additive is not None:
            self.initialize_additive_noise(additive)
        self.architecture['C']['cost']['R1'] = self.architecture['C']['cost']['R1'] @ self.additive['V']
        self.architecture['C']['cost']['Q'] = self.architecture['C']['cost']['Q'] @ self.additive['W']
        self.enhanced_system_matrix()

    def graph_initialize(self, graph_model):
        connected_network_check = False
        self.dynamics['number_of_nodes'] = graph_model['number_of_nodes']
        G = netx.Graph()
        while not connected_network_check:
            if graph_model['type'] == 'ER':
                if 'p' not in graph_model:
                    graph_model['p'] = 0.4
                G = netx.generators.random_graphs.erdos_renyi_graph(self.dynamics['number_of_nodes'], graph_model['p'])
            elif graph_model['type'] == 'BA':
                if 'p' not in graph_model:
                    graph_model['p'] = self.dynamics['number_of_nodes'] // 2
                G = netx.generators.random_graphs.barabasi_albert_graph(self.dynamics['number_of_nodes'], graph_model['p'])
            elif graph_model['type'] == 'rand':
                A = np.random.rand(self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes'])
                G = netx.from_numpy_matrix(A)
            elif graph_model['type'] == 'path':
                G = netx.generators.classic.path_graph(self.dynamics['number_of_nodes'])
            else:
                raise Exception('Check graph model')
            connected_network_check = netx.algorithms.components.is_connected(G)

        self.dynamics['Adj'] = netx.to_numpy_array(G)
        if 'self_loop' not in graph_model or graph_model['self_loop']:
            self.dynamics['Adj'] += np.identity(self.dynamics['number_of_nodes'])
        self.dynamics['A'] = self.dynamics['Adj'] * graph_model['rho'] / np.max(np.abs(np.linalg.eigvals(self.dynamics['Adj'])))

    def initialize_architecture(self, architecture):
        architecture_model = {'min': 1, 'max': self.dynamics['number_of_nodes'],
                              'cost': {'Q': np.identity(self.dynamics['number_of_nodes']),
                                       'R1': np.identity(self.dynamics['number_of_nodes']),
                                       'R2': 1,
                                       'R3': 1},
                              'active': [],
                              'matrix': np.zeros((self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes'])),
                              'available': range(0, self.dynamics['number_of_nodes']),
                              'set': [],
                              'history': [],
                              'gain': np.identity(self.dynamics['number_of_nodes'])}
        for architecture_type in self.architecture:
            if architecture is not None:
                if architecture_type in architecture:
                    for i in architecture[architecture_type]:
                        self.architecture[architecture_type][i] = architecture[architecture_type][i]
            for k in architecture_model:
                if k not in self.architecture[architecture_type]:
                    self.architecture[architecture_type][k] = dc(architecture_model[k])
        self.initialize_architecture_set_as_basis_vectors()
        if architecture['rand']:
            self.random_architecture()

    def initialize_architecture_set_as_basis_vectors(self):
        basis = []
        for i in range(0, self.dynamics['number_of_nodes']):
            basis.append(np.zeros(self.dynamics['number_of_nodes']))
            basis[-1][i] = 1
        if len(self.architecture['B']['set']) == 0:
            self.architecture['B']['set'] = basis
        if len(self.architecture['C']['set']) == 0:
            self.architecture['C']['set'] = [b.T for b in basis]

    def active_architecture_update(self, parameters):
        if parameters['architecture_type'] is not None:
            self.architecture[parameters['architecture_type']]['history'].append(self.architecture[parameters['architecture_type']]['active'])
            if parameters['algorithm'] == "select":
                self.architecture[parameters['architecture_type']]['active'] = self.architecture[parameters['architecture_type']]['active'] + [parameters['id']]
            elif parameters['algorithm'] == "reject":
                self.architecture[parameters['architecture_type']]['active'] = [i for i in self.architecture[parameters['architecture_type']]['active'] if i != parameters['id']]
            else:
                raise Exception('Check algorithm')
            self.architecture_active_to_matrix(parameters['architecture_type'])
            # self.architecture_costs(parameters['architecture_type'])

    def architecture_active_to_matrix(self, architecture_type=None):
        if architecture_type is None:
            architecture_type = ['B', 'C']
        for i in architecture_type:
            self.architecture[i]['matrix'] = np.zeros((self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes']))
            for k in range(0, len(self.architecture[i]['active'])):
                if i == 'B':
                    self.architecture[i]['matrix'][:, k] = self.architecture[i]['set'][self.architecture[i]['active'][k]]
                elif i == 'C':
                    self.architecture[i]['matrix'][k, :] = self.architecture[i]['set'][self.architecture[i]['active'][k]]

    def random_architecture(self, architecture_type=None):
        if architecture_type is None:
            architecture_type = ['B', 'C']
        for i in architecture_type:
            n_rand_set = max((self.architecture[i]['min'] + self.architecture[i]['max']) // 2, self.architecture[i]['min'])
            sample_ids = random.sample(self.architecture[i]['available'], k=n_rand_set)
            for j in sample_ids:
                self.active_architecture_update({'id': j, 'architecture_type': i, 'algorithm': 'select'})

    def initialize_additive_noise(self, additive):
        if additive == "normal":
            self.additive = {'W': np.identity(self.dynamics['number_of_nodes']),
                             'V': np.identity(self.dynamics['number_of_nodes'])}
        else:
            if additive is not None:
                for i in additive:
                    self.additive[i] = additive[i]

    def initialize_initial_conditions(self, initial_conditions=None):
        if initial_conditions is not None and 'X0' in initial_conditions:
            self.trajectory['X0'] = initial_conditions['X0']
        for key in ['x', 'x_estimate', 'y']:
            if initial_conditions is not None and key in initial_conditions:
                self.trajectory[key].append(initial_conditions[key])
            else:
                self.trajectory[key].append(np.random.default_rng().multivariate_normal(np.zeros(self.dynamics['number_of_nodes']), self.trajectory['X0']))

    def noise_gen(self):
        self.noise['w'] = np.random.default_rng().multivariate_normal(np.zeros(self.dynamics['number_of_nodes']), self.additive['W'])
        self.noise['v'] = np.random.default_rng().multivariate_normal(np.zeros(self.dynamics['number_of_nodes']), self.additive['V'])
        self.noise['enhanced_vector'] = np.block([self.noise['w'], self.noise['v']])

    def available_choices(self, algorithm, fixed_architecture=None):
        choices = []
        for a in ['B', 'C']:
            if algorithm in ['select', 'reject']:
                if algorithm == 'select' and max_limit(self.architecture[a], algorithm):
                    choice_a = compare_lists(self.architecture[a]['available'], self.architecture[a]['active'])['a1only']
                elif algorithm == 'reject' and min_limit(self.architecture[a], algorithm):
                    choice_a = compare_lists(self.architecture[a]['available'], self.architecture[a]['active'])['common']
                else:
                    return choices
            else:
                raise Exception('Check algorithm')
            if fixed_architecture is not None and a in fixed_architecture:
                choice_a = compare_lists(choice_a, fixed_architecture[a])['a1only']
            for i in choice_a:
                choices.append({'architecture_type': a, 'id': i})
        return choices

    def display_system(self):
        sys_plot = self.display_graph_gen()
        fig = plt.figure()
        gs = GridSpec(1, 1, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        netx.draw_networkx(sys_plot['G'], ax=ax1, pos=sys_plot['pos'], node_color=sys_plot['node_color'])
        ax1.set_aspect('equal')
        plt.show()

    def network_matrix(self):
        A_mat = self.dynamics['A']
        B_mat = self.architecture['B']['matrix']
        C_mat = self.architecture['C']['matrix']
        net_matrix = np.block([[A_mat, B_mat, C_mat.T],
                               [B_mat.T, np.zeros((self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes'])), np.zeros((self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes']))],
                               [C_mat, np.zeros((self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes'])), np.zeros((self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes']))]])
        return net_matrix

    def display_graph_gen(self, node_pos=None):
        net_matrix = self.network_matrix()
        G = netx.from_numpy_matrix(net_matrix)
        node_labels = {}
        for i in range(0, self.dynamics['number_of_nodes']):
            node_labels[i] = str(i + 1)
            node_labels[i + self.dynamics['number_of_nodes']] = "B" + str(i + 1)
            node_labels[i + (2 * self.dynamics['number_of_nodes'])] = "C" + str(i + 1)
        netx.relabel_nodes(G, node_labels, copy=False)
        G_degree = dc(G.degree())
        for n, d in G_degree:
            if d == 0:
                G.remove_node(n)
        node_color = {'B': 'C3', 'C': 'C1'}
        nc = []
        for i in G.nodes():
            if i[0] in node_color:
                nc.append(node_color[i[0]])
            else:
                nc.append('C2')
        if node_pos is None:
            G_base = netx.from_numpy_matrix(self.dynamics['A'])
            netx.relabel_nodes(G_base, node_labels, copy=False)
            node_pos = netx.circular_layout(G_base)
        node_pos = netx.spring_layout(G, pos=node_pos, fixed=[str(i + 1) for i in range(0, self.dynamics['number_of_nodes'])])
        return {'G': G, 'pos': node_pos, 'node_color': nc}

    def enhanced_system_matrix(self):
        BK = self.architecture['B']['matrix']@self.architecture['B']['gain']
        LC = self.architecture['C']['gain'] @ self.architecture['C']['matrix']
        self.dynamics['enhanced'] = np.block([
            [self.dynamics['A'], -self.architecture['B']['matrix'] @ self.architecture['B']['gain']],
            [self.architecture['C']['gain'] @ self.architecture['C']['matrix'] @ self.dynamics['A'], self.dynamics['A'] - BK - (LC@self.dynamics['A'])]])
        self.noise['enhanced_matrix'] = np.block([
            [np.identity(self.dynamics['number_of_nodes']), np.zeros((self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes']))],
            [LC, self.architecture['C']['gain']]])

    def optimal_control_feedback_wrapper(self, T_steps=None):
        # print('Control Feedback')
        self.trajectory['P'] = [self.architecture['B']['cost']['Q']]
        loop_check = False
        t = 0
        while not loop_check:
            self.optimal_feedback_control_cost_matrix()
            if len(self.trajectory['P']) > 1:
                if matrix_convergence_check(self.trajectory['P'][-1], self.trajectory['P'][-2]):
                    loop_check = True
            t += 1
            if T_steps is not None and (T_steps - t) <= 0:
                break
            if np.min(np.linalg.eigvals(self.trajectory['P'][-1])) <= 0:
                raise Exception('Non-positive cost matrix')

    def optimal_feedback_control_cost_matrix(self):
        self.optimal_feedback_control_cost_gain()
        self.trajectory['P'].append((self.dynamics['A'].T @ self.trajectory['P'][-1] @ self.dynamics['A']) - (self.dynamics['A'].T @ self.trajectory['P'][-1] @ self.architecture['B']['matrix'] @ self.architecture['B']['gain']) + self.architecture['B']['cost']['Q'])

    def optimal_feedback_control_cost_gain(self):
        self.architecture['B']['gain'] = np.linalg.inv(self.architecture['B']['matrix'].T @ self.trajectory['P'][-1] @ self.architecture['B']['matrix'] + self.architecture['B']['cost']['R1']) @ self.architecture['B']['matrix'].T @ self.trajectory['P'][-1] @ self.dynamics['A']

    def optimal_estimation_feedback_wrapper(self, T_steps=None):
        # print('Estimation Feedback')
        self.trajectory['E'] = [self.additive['V']]
        loop_check = False
        t = 0
        while not loop_check:
            self.optimal_feedback_estimation_covariance_matrix()
            if len(self.trajectory['E']) > 1:
                if matrix_convergence_check(self.trajectory['E'][-1], self.trajectory['E'][-2]):
                    loop_check = True
            t += 1
            if T_steps is not None and t >= T_steps:
                break
            if np.min(np.linalg.eigvals(self.trajectory['E'][-1])) <= 0:
                raise Exception('Non-positive error covariance matrix')

    def optimal_feedback_estimation_covariance_matrix(self):
        self.optimal_feedback_estimation_gain()
        self.trajectory['E'].append(self.dynamics['A'] @ (self.trajectory['E'][-1] - self.architecture['C']['gain'] @ self.architecture['C']['matrix'] @ self.trajectory['E'][-1]) @ self.dynamics['A'].T + self.architecture['C']['cost']['Q'])

    def optimal_feedback_estimation_gain(self):
        self.architecture['C']['gain'] = self.trajectory['E'][-1] @ self.architecture['C']['matrix'].T @ np.linalg.inv(self.architecture['C']['matrix'] @ self.trajectory['E'][-1] @ self.architecture['C']['matrix'].T + self.architecture['C']['cost']['R1'])

    def system_one_step_update_enhanced(self):
        self.enhanced_system_matrix()
        self.noise_gen()
        self.trajectory['u'].append(-self.architecture['B']['gain']@self.trajectory['x_estimate'][-1])
        self.trajectory['y'].append(self.architecture['C']['matrix']@self.trajectory['x'] + self.noise['v'])
        vector = self.dynamics['enhanced']@np.block([self.trajectory['x'][-1], self.trajectory['x_estimate'][-1]]) + self.noise['enhanced_matrix']@self.noise['enhanced_vector']
        self.trajectory['x'].append(vector[0:self.dynamics['number_of_nodes']])
        self.trajectory['x_estimate'].append(vector[self.dynamics['number_of_nodes']:])

    def architecture_costs(self, architecture_type=None):
        if architecture_type is None:
            architecture_type = ['B', 'C']
        for a in architecture_type:
            active_vector = np.zeros(self.dynamics['number_of_nodes'])
            history_vector = np.zeros(self.dynamics['number_of_nodes'])
            for i in self.architecture[a]['active']:
                active_vector[i] = 1
            for i in self.architecture[a]['history'][-1]:
                history_vector[i] = 1
            if self.metric_model['type2'] == 'matrix':
                self.trajectory['cost']['running'] = active_vector.T @ self.architecture[a]['cost']['R2'] @ active_vector
            elif self.metric_model['type2'] == 'scalar':
                self.trajectory['cost']['running'] = self.architecture[a]['cost']['R2'] * (active_vector.T @ active_vector)
            else:
                raise Exception('Check Metric for Type 2 - Running Costs')
            if self.metric_model['type3'] == 'matrix':
                self.trajectory['cost']['switching'] = (active_vector - history_vector).T @ self.architecture[a]['cost']['R3'] @ (active_vector - history_vector)
            elif self.metric_model['type3'] == 'scalar':
                self.trajectory['cost']['switching'] = self.architecture[a]['cost']['R3']*((active_vector - history_vector).T @ (active_vector - history_vector))
            else:
                raise Exception('Check Metric for Type 2 - Running Costs')

    def true_stage_costs(self):
        self.trajectory['cost']['true_stage'] = 0
        self.trajectory['cost']['true_stage'] += self.trajectory['x'][-1].T @ self.architecture['B']['cost']['Q'] @ self.trajectory['x'][-1]
        self.trajectory['cost']['true_stage'] += (-self.architecture['B']['gain']@self.trajectory['x_estimate'][-1]).T @ self.architecture['B']['cost']['R1'] @ (-self.architecture['B']['gain']@self.trajectory['x_estimate'][-1])
        self.trajectory['cost']['true_stage'] += (self.trajectory['x'][-1] - self.trajectory['x_estimate'][-1]).T @ self.architecture['C']['cost']['Q'] @ (self.trajectory['x'][-1] - self.trajectory['x_estimate'][-1])

    def estimate_stage_costs(self):
        self.trajectory['cost']['estimate_stage'] = 0
        self.trajectory['cost']['estimate_stage'] += self.trajectory['x_estimate'][-1].T @ self.trajectory['P'][-1] @ self.trajectory['x_estimate'][-1]
        self.trajectory['cost']['estimate_stage'] += (-self.architecture['B']['gain']@self.trajectory['x_estimate'][-1]).T @ self.architecture['B']['cost']['R1'] @ (-self.architecture['B']['gain']@self.trajectory['x_estimate'][-1])
        self.trajectory['cost']['estimate_stage'] += np.trace(self.trajectory['E'][-1] @ self.architecture['C']['cost']['Q'])

    def cost_wrapper(self, cost_type):
        self.optimal_control_feedback_wrapper()
        self.optimal_estimation_feedback_wrapper()
        self.architecture_costs()
        # cost = self.trajectory['cost']['running'] + self.trajectory['cost']['switching']
        if cost_type == 'estimate':
            self.estimate_stage_costs()
            # cost += self.trajectory['cost']['estimate_stage']
            self.trajectory['cost']['estimate_total'].append(self.trajectory['cost']['running'] + self.trajectory['cost']['switching'] + self.trajectory['cost']['estimate_stage'])
        elif cost_type == 'true':
            self.true_stage_costs()
            self.trajectory['cost']['true_total'].append(self.trajectory['cost']['running'] + self.trajectory['cost']['switching'] + self.trajectory['cost']['true_stage'])
        else:
            raise Exception('Check cost type')

    def architecture_update_check(self):
        check = False
        for i in ['B', 'C']:
            architecture_compare = compare_lists(self.architecture[i]['active'], self.architecture[i]['history'][-1])
            if len(architecture_compare['a1only']) != 0 or len(architecture_compare['a2only']) != 0:
                check = True
                break
        return check
    
    def architecture_limit_modifier(self, min_mod=None, max_mod=None):
        for architecture_type in ['B', 'C']:
            if min_mod is not None:
                self.architecture[architecture_type]['min'] += min_mod
            if max_mod is not None:
                self.architecture[architecture_type]['max'] += max_mod


def matrix_convergence_check(A, B, accuracy=10**(-3), check_type=None):
    np_norm_methods = ['inf', 'fro', 2, None]
    if check_type is None:
        return np.allclose(A, B, atol=accuracy, rtol=accuracy)
    elif check_type in np_norm_methods:
        return np.norm(A-B, ord=check_type) < accuracy
    else:
        raise Exception('Check Matrix Convergence')


def max_limit(architecture, algorithm):
    correction = {'select': 0, 'reject': 1}
    return len(architecture['active']) < (architecture['max'] + correction[algorithm])


def min_limit(architecture, algorithm):
    correction = {'select': 0, 'reject': 1}
    return len(architecture['active']) >= (architecture['min'] + correction[algorithm])


def compare_lists(array1, array2):
    return {'a1only': [i for i in array1 if i not in array2], 'a2only': [i for i in array2 if i not in array1], 'common': [i for i in array1 if i in array2]}


def item_index_from_policy(values, policy):
    if policy == "max":
        return values.index(max(values))
    elif policy == "min":
        return values.index(min(values))
    else:
        raise Exception('Check policy')


def greedy_architecture_selection(sys, number_of_changes=None, policy="min", no_select=False, status_check=False, t_start=time.time()):
    if not isinstance(sys, System):
        raise Exception('Check data type')

    work_iteration = dc(sys)
    work_history, choice_history, value_history = [], [], []
    selection_check = max_limit(work_iteration.architecture['B'], 'select') or max_limit(work_iteration.architecture['C'], 'select')
    count_of_changes = 0
    while selection_check:
        # print('greedy select:', count_of_changes)
        # print(work_iteration)
        work_history.append(work_iteration)
        choices = work_iteration.available_choices('select')
        if len(choices) == 0:
            if status_check:
                print('No selections available')
            selection_check = False
            break
        if no_select and min_limit(work_iteration.architecture['B'], 'select') and min_limit(work_iteration.architecture['C'], 'select'):
            choices.append({'architecture_type': None})
        choice_history.append(choices)
        for i in choices:
            i['algorithm'] = 'select'
            test_sys = dc(work_iteration)
            test_sys.active_architecture_update(i)
            test_sys.cost_wrapper('estimate')
            # print(test_sys.trajectory['cost']['estimate_total'][-1])
            i['value'] = test_sys.trajectory['cost']['estimate_total'][-1]
        target_idx = item_index_from_policy([i['value'] for i in choices], policy)
        value_history.append([i['value'] for i in choices])
        work_iteration.active_architecture_update(choices[target_idx])
        if not work_iteration.architecture_update_check():
            print('No valuable architecture updates')
            break
        count_of_changes += 1
        if number_of_changes is not None and count_of_changes == number_of_changes:
            if status_check:
                print('Maximum number of changes done')
            break
        selection_check = max_limit(work_iteration.architecture['B'], 'select') or max_limit(work_iteration.architecture['C'], 'select')
    work_history.append(work_iteration)
    return {'work_set': work_iteration, 'work_history': work_history, 'choice_history': choice_history, 'value_history': value_history, 'time': time.time() - t_start}


def greedy_architecture_rejection(sys, number_of_changes=None, policy="min", no_reject=False, status_check=False, t_start=time.time()):
    if not isinstance(sys, System):
        raise Exception('Check data type')

    work_iteration = dc(sys)
    work_history, choice_history, value_history = [], [], []
    rejection_check = min_limit(work_iteration.architecture['B'], 'reject') or max_limit(work_iteration.architecture['C'], 'reject')
    count_of_changes = 0
    while rejection_check:
        # print('greedy select:', count_of_changes)
        # print(work_iteration)
        work_history.append(work_iteration)
        choices = work_iteration.available_choices('reject')
        if len(choices) == 0:
            if status_check:
                print('No selections available')
            rejection_check = False
            break
        if no_reject and min_limit(work_iteration.architecture['B'], 'select') and min_limit(work_iteration.architecture['C'], 'reject'):
            choices.append({'architecture_type': None})
        choice_history.append(choices)
        for i in choices:
            i['algorithm'] = 'reject'
            test_sys = dc(work_iteration)
            test_sys.active_architecture_update(i)
            test_sys.cost_wrapper('estimate')
            # print(test_sys.trajectory['cost']['estimate_total'][-1])
            i['value'] = test_sys.trajectory['cost']['estimate_total'][-1]
        target_idx = item_index_from_policy([i['value'] for i in choices], policy)
        value_history.append([i['value'] for i in choices])
        work_iteration.active_architecture_update(choices[target_idx])
        if not work_iteration.architecture_update_check():
            print('No valuable architecture updates')
            break
        count_of_changes += 1
        if number_of_changes is not None and count_of_changes == number_of_changes:
            if status_check:
                print('Maximum number of changes done')
            break
        rejection_check = min_limit(work_iteration.architecture['B'], 'reject') or max_limit(work_iteration.architecture['C'], 'reject')
    work_history.append(work_iteration)
    return {'work_set': work_iteration, 'work_history': work_history, 'choice_history': choice_history, 'value_history': value_history, 'time': time.time() - t_start}
    

def greedy_simultaneous(sys, iterations=1, changes_per_iteration=1, fixed_set=None, failure_set=None, policy="min", t_start=time.time(), status_check=False):
    if not isinstance(sys, System):
        raise Exception('Incorrect data type')
    work_iteration = dc(sys)
    work_history = [work_iteration]
    value_history = []
    for _ in range(0, iterations):
        # Keep same
        values = []
        iteration_cases = []

        iteration_cases.append(work_iteration)
        iteration_cases[-1].cost_wrapper('estimate')
        # print(iteration_cases[-1].trajectory['cost']['estimate_total'][-1])
        values.append(iteration_cases[-1].trajectory['cost']['estimate_total'][-1])
        all_values = [values[-1]]

        # Select one
        select = greedy_architecture_selection(sys, number_of_changes=changes_per_iteration, policy=policy, t_start=t_start, no_select=True, status_check=status_check)
        iteration_cases.append(select['work_set'])
        iteration_cases[-1].cost_wrapper('estimate')
        values.append(iteration_cases[-1].trajectory['cost']['estimate_total'][-1])
        all_values += select['value_history']

        # Reject one
        reject = greedy_architecture_rejection(sys, number_of_changes=changes_per_iteration, policy=policy, t_start=t_start, no_reject=True, status_check=status_check)
        iteration_cases.append(reject['work_set'])
        iteration_cases[-1].cost_wrapper('estimate')
        values.append(iteration_cases[-1].trajectory['cost']['estimate_total'][-1])
        all_values += reject['value_history']

        # Swap: add then drop
        sys_select = dc(work_iteration)
        sys_select.architecture_limit_modifier(min_mod=1, max_mod=1)
        swap_select1 = greedy_architecture_selection(sys_select, number_of_changes=changes_per_iteration, policy=policy, t_start=t_start, no_select=True, status_check=False)
        sys_select = dc(swap_select1['work_set'])
        sys_select.architecture_limit_modifier(min_mod=-1, max_mod=-1)
        swap_reject1 = greedy_architecture_rejection(sys_select, number_of_changes=changes_per_iteration, policy=policy, t_start=t_start, status_check=status_check)
        iteration_cases.append(swap_reject1['work_set'])
        iteration_cases[-1].cost_wrapper('estimate')
        values.append(iteration_cases[-1].trajectory['cost']['estimate_total'][-1])
        all_values += swap_reject1['value_history']

        target_idx = item_index_from_policy(values, policy)
        work_iteration = iteration_cases[target_idx]
        work_history.append(work_iteration)
        value_history.append(all_values)
        if len(compare_lists(work_history[-1].architecture['B']['active'], work_history[-2].architecture['B']['active'])['a1only']) == 0 and len(compare_lists(work_history[-1].architecture['C']['active'], work_history[-2].architecture['C']['active'])['a1only']) == 0:
            print('No changes to work_set')
            break

    return {'work_set': work_iteration, 'work_history': work_history, 'value_history': value_history, 'time': time.time()-t_start}
