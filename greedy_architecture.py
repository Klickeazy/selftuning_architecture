import numpy as np
import networkx as netx
import random
import time
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
    def __init__(self, number_of_nodes=None, graph_model=None, rho=1, architecture=None, additive=None, mpl=None, initial_conditions=None):
        if number_of_nodes is None:
            number_of_nodes = 10
        self.dynamics = {'number': number_of_nodes}
        self.dynamics['A'] = np.zeros((self.dynamics['number'], self.dynamics['number']))
        if graph_model is None:
            graph_model = {'type': 'rand', 'self_loop': True}
        self.graph_initialize(graph_model)
        self.dynamics['A'] *= rho

        self.architecture = {'B': {}, 'C': {}}
        self.initialize_architecture(architecture)

        keys = ['dynamics', 'B', 'C']
        if mpl is not None:
            if 'dynamics' in mpl:
                self.dynamics['covariances'] = mpl['dynamics']['covariances']
                self.dynamics['matrices'] = mpl['dynamics']['matrices']
                keys = [i for i in keys if i != 'dynamics']
            if 'B' in mpl:
                self.architecture['B']['covariances'] = mpl['B']['covariances']
                self.architecture['B']['matrices'] = mpl['B']['matrices']
                keys = [i for i in keys if i != 'B']
            if 'C' in mpl:
                self.architecture['C']['covariances'] = mpl['C']['covariances']
                self.architecture['C']['matrices'] = mpl['C']['matrices']
                keys = [i for i in keys if i != 'C']
        self.initialize_no_mpl(keys)

        self.additive = {'W': None, 'V': None}
        if additive is not None:
            for i in additive:
                self.additive[i] = additive[i]

        self.vector = {'x0': None, 'X0': np.identity(self.dynamics['number']), 'x': None, 'x_estimate': None, 'y': None, 'x_history': [], 'x_estimate_history': [], 'y_history': [], 'u_history': []}
        self.metric_model = {'type1': 'x', 'type2': 'scalar', 'type3': 'scalar'}
        self.initialize_initial_conditions(initial_conditions)
        
        self.noise = {}

    def graph_initialize(self, graph_model):
        connected_network_check = False
        G = netx.Graph()
        while not connected_network_check:
            if graph_model['type'] == 'ER':
                if 'p' not in graph_model:
                    graph_model['p'] = 0.4
                G = netx.generators.random_graphs.erdos_renyi_graph(self.dynamics['number'], graph_model['p'])
            elif graph_model['type'] == 'BA':
                if 'p' not in graph_model:
                    graph_model['p'] = self.dynamics['number'] // 2
                G = netx.generators.random_graphs.barabasi_albert_graph(self.dynamics['number'], graph_model['p'])
            elif graph_model['type'] == 'rand':
                A = np.random.rand(self.dynamics['number'], self.dynamics['number'])
                G = netx.from_numpy_matrix(A)
            elif graph_model['type'] == 'path':
                G = netx.generators.classic.path_graph(self.dynamics['number'])
            else:
                raise Exception('Check graph model')
            connected_network_check = netx.algorithms.components.is_connected(G)

        self.dynamics['Adj'] = netx.to_numpy_array(G)
        if 'self_loop' not in graph_model or graph_model['self_loop']:
            self.dynamics['Adj'] += np.identity(self.dynamics['number'])
        e = np.max(np.abs(np.linalg.eigvals(self.dynamics['Adj'])))
        self.dynamics['A'] = self.dynamics['Adj'] / e

    def random_architecture(self, architecture_type=None):
        if architecture_type is None:
            architecture_type = ['B', 'C']
        for i in architecture_type:
            self.active_architecture_update(random.sample(self.architecture[i]['available'], k=(self.architecture[i]['min'] + self.architecture[i]['max']) // 2), i)

    def architecture_active_to_matrix(self, architecture_type=None):
        if architecture_type is None:
            architecture_type = ['B', 'C']
        for i in architecture_type:
            self.architecture[i]['matrix'] = np.zeros((self.dynamics['number'], self.dynamics['number']))
            for k in range(0, len(self.architecture[i]['active'])):
                if i == 'B':
                    self.architecture[i]['matrix'][:, k] = self.architecture[i]['set'][self.architecture[i]['active'][k]]
                elif i == 'C':
                    self.architecture[i]['matrix'][k, :] = self.architecture[i]['set'][self.architecture[i]['active'][k]]

    def initialize_architecture(self, architecture):
        architecture_model = {'min': 0, 'max': self.dynamics['number'],
                              'cost': {'Q': np.identity(self.dynamics['number']),
                                       'R1': np.identity(self.dynamics['number']),
                                       'R2': 1,
                                       'R3': 1},
                              'active': [],
                              'matrix': np.zeros((self.dynamics['number'], self.dynamics['number'])),
                              'available': range(0, self.dynamics['number']),
                              'set': [],
                              'history': [],
                              'gain': np.identity(self.dynamics['number'])}
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

    def active_architecture_update(self, new_architecture, architecture_type):
        self.architecture[architecture_type]['history'].append(self.architecture[architecture_type]['active'])
        self.architecture[architecture_type]['active'] = dc(new_architecture)
        self.architecture_active_to_matrix([architecture_type])

    def initialize_architecture_set_as_basis_vectors(self):
        basis = []
        for i in range(0, self.dynamics['number']):
            basis.append(np.zeros(self.dynamics['number']))
            basis[-1][i] = 1
        if len(self.architecture['B']['set']) == 0:
            self.architecture['B']['set'] = basis
        if len(self.architecture['C']['set']) == 0:
            self.architecture['C']['set'] = [b.T for b in basis]

    def initialize_no_mpl(self, keys):
        mpl_none = {'covariances': None, 'matrices': None}
        if 'dynamics' in keys:
            for i in mpl_none:
                self.dynamics[i] = mpl_none[i]
        if 'B' in keys:
            for i in mpl_none:
                self.architecture['B'][i] = mpl_none[i]
        if 'C' in keys:
            for i in mpl_none:
                self.architecture['C'][i] = mpl_none[i]

    def initialize_initial_conditions(self, initial_conditions=None):
        if initial_conditions is not None and 'X0' in initial_conditions:
            self.vector['X0'] = initial_conditions['X0']
        for key in ['x', 'x0', 'x_estimate', 'y_estimate']:
            if initial_conditions is not None and key in initial_conditions:
                self.vector[key] = dc(initial_conditions[key])
            else:
                self.vector[key] = np.random.default_rng().multivariate_normal(np.zeros(self.dynamics['number']), self.vector['X0'])

    def dynamics_update(self):
        self.vector['x_history'].append(self.vector['x'])
        self.vector['y_history'].append(self.vector['y'])
        Ai_mat = np.zeros((self.dynamics['number'], self.dynamics['number']))
        if self.dynamics['covariances'] is not None:
            for i in range(0, len(self.dynamics['covariances'])):
                Ai_mat += self.noise['alpha_i'][i, 0]*self.dynamics['matrix'][i]
            self.noise['alpha_i'] = self.noise['alpha_i'][:, 1:]
        Bj_mat = np.zeros((self.dynamics['number'], self.dynamics['number']))
        if self.architecture['B']['covariances'] is not None:
            for j in range(0, len(self.architecture['B']['covariances'])):
                Bj_mat += self.noise['beta_j'][j, 0]*self.architecture['B']['matrix'][j]
            self.noise['beta_j'] = self.noise['beta_j'][:, 1:]
        self.vector['x'] = (self.dynamics['A'] + Ai_mat - (self.architecture['B']['matrix'] + Bj_mat) @ self.architecture['B']['gain']) @ self.vector['x']
        if 'w' in self.noise:
            self.vector['x'] += self.noise['w'][:, 0]
            self.noise['w'] = self.noise['w'][:, 1:]

        Ck_mat = np.zeros((self.dynamics['number'], self.dynamics['number']))
        if self.architecture['C']['covariances'] is not None:
            for k in range(0, len(self.architecture['C']['covariances'])):
                Ck_mat += self.noise['gamma_k'][k, 0] * self.architecture['C']['matrix'][k]
            self.noise['gamma_k'] = self.noise['gamma_k'][:, 1:]
        self.vector['y'] = (self.architecture['C']['matrix'] + Ck_mat)@self.vector['x']
        if 'v' in self.noise:
            self.vector['y'] += self.noise['v'][:, 0]
            self.noise['v'] = self.noise['v'][:, 1:]

    def noise_gen(self, T_sim=1):
        if self.additive['W'] is not None:
            self.noise['w'] = np.random.default_rng().multivariate_normal(np.zeros(self.dynamics['number']), self.additive['W'], T_sim)
        if self.additive['V'] is not None:
            self.noise['v'] = np.random.default_rng().multivariate_normal(np.zeros(self.dynamics['number']), self.additive['V'], T_sim)
        if self.dynamics['covariances'] is not None:
            self.noise['alpha_i'] = np.random.default_rng().multivariate_normal(np.zeros(len(self.dynamics['covariances'])), np.diag(self.dynamics['covariances']), T_sim)
        if self.architecture['B']['covariances'] is not None:
            self.noise['beta_j'] = np.random.default_rng().multivariate_normal(np.zeros(len(self.architecture['B']['covariances'])), np.diag(self.architecture['B']['covariances']), T_sim)
        if self.architecture['C']['covariances'] is not None:
            self.noise['gamma_k'] = np.random.default_rng().multivariate_normal(np.zeros(len(self.architecture['C']['covariances'])), np.diag(self.architecture['C']['covariances']), T_sim)

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
        if self.dynamics['covariances'] is not None:
            for i in range(0, len(self.dynamics['covariances'])):
                A_mat += (self.dynamics['covariances'][i] * self.dynamics['matrices'][i])
        B_mat = self.architecture['B']['matrix']
        if self.architecture['B']['covariances'] is not None:
            for i in range(0, len(self.architecture['B']['covariances'])):
                B_mat += (self.architecture['B']['covariances'][i] * self.architecture['B']['matrices'][i])
        C_mat = self.architecture['C']['matrix']
        if self.architecture['C']['covariances'] is not None:
            for i in range(0, len(self.architecture['C']['covariances'])):
                C_mat += (self.architecture['C']['covariances'][i] * self.architecture['C']['matrices'][i])
        net_matrix = np.block([[A_mat, B_mat, C_mat.T],
                               [B_mat.T, np.zeros((self.dynamics['number'], self.dynamics['number'])), np.zeros((self.dynamics['number'], self.dynamics['number']))],
                               [C_mat, np.zeros((self.dynamics['number'], self.dynamics['number'])), np.zeros((self.dynamics['number'], self.dynamics['number']))]])
        return net_matrix

    def display_graph_gen(self, node_pos=None):
        net_matrix = self.network_matrix()
        G = netx.from_numpy_matrix(net_matrix)
        node_labels = {}
        for i in range(0, self.dynamics['number']):
            node_labels[i] = str(i + 1)
            node_labels[i + self.dynamics['number']] = "B" + str(i + 1)
            node_labels[i + (2 * self.dynamics['number'])] = "C" + str(i + 1)
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
        node_pos = netx.spring_layout(G, pos=node_pos, fixed=[str(i + 1) for i in range(0, self.dynamics['number'])])
        return {'G': G, 'pos': node_pos, 'node_color': nc}

    def evaluate_control_cost_matrix(self, P):
        if self.metric_model['type1'] == 'eigP':
            return np.max(np.linalg.eigvals(P))
        elif self.metric_model['type1'] == 'x':
            return self.vector['x'].T @ P @ self.vector['x']
        elif self.metric_model['type1'] == 'X0':
            return np.matrix.trace(P @ self.vector['X0'])
        else:
            raise Exception('Check Metric for Type 1')

    def evaluate_estimation_cost_matrix(self, P):
        if self.metric_model['type1'] == 'eigP':
            return np.max(np.linalg.eigvals(P))
        elif self.metric_model['type1'] == 'x':
            return (self.vector['x'] - self.vector['x_estimate']).T @ P @ (self.vector['x'] - self.vector['x_estimate'])
        elif self.metric_model['type1'] == 'X0':
            return np.matrix.trace(P @ self.vector['X0'])
        else:
            raise Exception('Check Metric for Type 1')

    def architecture_running_cost(self, architecture_type):
        active_vector = np.zeros(self.dynamics['number'])
        for i in self.architecture[architecture_type]['active']:
            active_vector[i] = 1
        if self.metric_model['type2'] == 'matrix':
            return active_vector.T @ self.architecture[architecture_type]['cost']['R2'] @ active_vector
        elif self.metric_model['type2'] == 'scalar':
            return self.architecture[architecture_type]['cost']['R2'] * (active_vector.T @ active_vector)
        else:
            raise Exception('Check Metric for Type 2 - Running Costs')

    def architecture_switching_cost(self, architecture_type):
        active_vector = np.zeros(self.dynamics['number'])
        history_vector = np.zeros(self.dynamics['number'])
        for i in self.architecture[architecture_type]['active']:
            active_vector[i] = 1
        for i in self.architecture[architecture_type]['history'][-1]:
            history_vector[i] = 1

        if self.metric_model['type3'] == 'matrix':
            return (active_vector - history_vector).T @ self.architecture[architecture_type]['cost']['R3'] @ (active_vector - history_vector)
        elif self.metric_model['type3'] == 'scalar':
            return self.architecture[architecture_type]['cost']['R3']*((active_vector - history_vector).T @ (active_vector - history_vector))
        else:
            raise Exception('Check Metric for Type 2 - Running Costs')

    def architecture_cost_calculator(self, architecture_type, active_architecture=None):
        self_model = dc(self)
        if active_architecture is not None:
            self_model.active_architecture_update(active_architecture, architecture_type)
        if architecture_type == 'B':
            return self_model.total_control_cost()
        elif architecture_type == 'C':
            return self_model.total_estimation_cost()
        else:
            raise Exception('Check architecture type')

    def total_control_cost(self, T_horizon=30):
        feedback_costs = self.control_feedback(T_horizon=T_horizon)
        cost = 0
        cost += self.evaluate_control_cost_matrix(feedback_costs['P'][-1])
        cost += self.architecture_running_cost('B')
        cost += self.architecture_switching_cost('B')
        return {'cost': cost, 'gain': feedback_costs['K'][-1]}

    def control_feedback(self, T_horizon=30, converge_accuracy=10 ** (-3)):
        P = dc(self.architecture['B']['cost']['Q'])
        P_history = [P]
        K_history = []
        convergence_check = False
        T_run = 0
        for t in range(0, T_horizon):
            P, K = self.iteration_control_cost(P)
            P_history.append(P)
            K_history.append(K)
            convergence_check = matrix_convergence_check(P_history[-1], P_history[-2], accuracy=converge_accuracy)
            if convergence_check:
                T_run = t
                break
        return {'P': P_history, 'K': K_history, 't_run': T_run, 'convergence_check': convergence_check}

    def iteration_control_cost(self, P):
        K_mat = self.iteration_control_feedback(P)
        P_mat = np.zeros_like(self.dynamics['A'])
        P_mat += self.architecture['B']['cost']['Q'] + self.dynamics['A'].T @ P @ self.dynamics['A']
        if self.dynamics['covariances'] is not None:
            P_mat += self.dynamics_mpl(P, 'B')
        P_mat -= self.dynamics['A'].T @ P @ self.architecture['B']['matrix'] @ K_mat
        return P_mat, K_mat

    def iteration_control_feedback(self, P):
        K_mat = self.architecture['B']['cost']['R1'] + self.architecture['B']['matrix'].T @ P @ self.architecture['B']['matrix']
        if self.architecture['B']['covariances'] is not None:
            for j in range(0, self.architecture['B']['covariances']):
                K_mat += self.architecture['B']['covariances'][j] * (
                            self.architecture['B']['matrices'][j].T @ P @ self.architecture['B']['matrices'][j])
        K_mat = np.linalg.inv(K_mat) @ self.architecture['B']['matrix'].T @ P @ self.dynamics['A']
        return K_mat

    def total_estimation_cost(self, T_horizon=30):
        feedback_costs = self.estimation_feedback(T_horizon=T_horizon)
        cost = 0
        cost += self.evaluate_estimation_cost_matrix(feedback_costs['P'][-1])
        cost += self.architecture_running_cost('B')
        cost += self.architecture_switching_cost('B')
        return {'cost': cost, 'gain': feedback_costs['L'][-1]}

    def estimation_feedback(self, T_horizon=30, converge_accuracy=10 ** (-3)):
        P = dc(self.architecture['B']['cost']['Q'])
        P_history = [P]
        L_history = []
        convergence_check = False
        T_run = 0
        for t in range(0, T_horizon):
            P, L = self.iteration_estimation_cost(P)
            P_history.append(P)
            L_history.append(L)
            convergence_check = matrix_convergence_check(P_history[-1], P_history[-2], accuracy=converge_accuracy)
            if convergence_check:
                T_run = t
                break
        return {'P': P_history, 'L': L_history, 't_run': T_run, 'convergence_check': convergence_check}

    def iteration_estimation_cost(self, P):
        L_mat = self.iteration_estimation_feedback(P)
        P_mat = np.zeros_like(self.dynamics['A'])
        P_mat += self.architecture['C']['cost']['Q'] + self.dynamics['A'] @ P @ self.dynamics['A'].T
        if self.dynamics['covariances'] is not None:
            P_mat += self.dynamics_mpl(P, 'C')
        P_mat -= self.dynamics['A'] @ L_mat @ self.architecture['C']['matrix'] @ P @ self.dynamics['A'].T
        return P_mat, L_mat

    def iteration_estimation_feedback(self, P):
        L_mat = self.architecture['C']['matrix'] @ P @ self.architecture['C']['matrix'].T + self.architecture['C']['cost']['R1']
        if self.architecture['C']['covariances'] is not None:
            for j in range(0, len(self.architecture['C']['covariances'])):
                L_mat += self.architecture['C']['covariances'][j] * (self.architecture['C']['matrices'][j] @ P @ self.architecture['C']['matrices'][j].T)
        L_mat = P @ self.architecture['C']['matrix'].T @ np.linalg.inv(L_mat)
        return L_mat

    def dynamics_mpl(self, P, architecture_type):
        Mat = np.zeros((self.dynamics['number'], self.dynamics['number']))
        for i in range(0, len(self.dynamics['covariances'])):
            if architecture_type == 'B':
                Mat += self.dynamics['covariances'][i] * (self.dynamics['matrices'][i].T @ P @ self.dynamics['matrices'][i])
            elif architecture_type == 'C':
                Mat += self.dynamics['covariances'][i] * (self.dynamics['matrices'][i] @ P @ self.dynamics['matrices'][i].T)
        return Mat


def animate_architecture(S, architecture_history):
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    ax1.axis('off')
    min_lim = -1.7
    max_lim = -1.7
    ax1.set_xlim(min_lim, max_lim)
    ax1.set_ylim(min_lim, max_lim)
    Sys_dummy = dc(S)
    Sys_dummy.architecture['B']['active'] = Sys_dummy.architecture['B']['available']
    Sys_dummy.architecture['C']['active'] = Sys_dummy.architecture['C']['available']
    Sys_dummy.architecture_active_to_matrix()
    pos_gen = Sys_dummy.display_graph_gen()['pos']
    node_pos = {i: pos_gen[i] for i in pos_gen if i.isnumeric()}

    def update(t):
        ax1.clear()
        sys_t = dc(S)
        for i in architecture_history[t]:
            sys_t.architecture[i]['active'] = dc(architecture_history[t][i]['active'])
        sys_t.architecture_active_to_matrix()
        sys_t_plot = sys_t.display_graph_gen(node_pos)
        netx.draw_networkx(sys_t_plot['G'], ax=ax1, pos=sys_t_plot['pos'], node_color=sys_t_plot['node_color'])
        ax1.set_title('t=' + str(t))
        ax1.set_xlim(min_lim, max_lim)
        ax1.set_ylim(min_lim, max_lim)

    ani = matplotlib.animation.FuncAnimation(fig, update, frames=np.arange(0, len(architecture_history), 1), interval=1000, repeat=False)
    ani.save("Test.mp4")
    plt.show()


def max_limit(work_set, architecture, algorithm):
    correction = {'selection': 0, 'rejection': 1}
    return len(work_set) < (architecture['max'] + correction[algorithm])


def min_limit(work_set, architecture, algorithm):
    correction = {'selection': 0, 'rejection': 1}
    return len(work_set) >= (architecture['min'] + correction[algorithm])


def compare_lists(array1, array2):
    return {'a1only': [i for i in array1 if i not in array2], 'a2only': [i for i in array2 if i not in array1], 'common': [i for i in array1 if i in array2]}


def item_index_from_policy(values, policy):
    if policy == "max":
        return values.index(max(values))
    elif policy == "min":
        return values.index(min(values))
    else:
        raise Exception('Check policy')


def matrix_convergence_check(A, B, accuracy=10**(-3), check_type=None):
    np_norm_methods = ['inf', 'fro', 2, None]
    if check_type is None:
        return np.allclose(A, B, atol=accuracy, rtol=accuracy)
    elif check_type in np_norm_methods:
        return np.norm(A-B, ord=check_type) < accuracy
    else:
        raise Exception('Check Matrix Convergence')


def greedy_architecture_selection(sys_model, architecture_type, number_of_changes=None, fixed_set=None, failure_set=None, policy="min", t_start=time.time(), no_select=False, status_check=False):
    work_sys, available_set, work_iteration = initialize_greedy(sys_model, architecture_type, failure_set)
    choice_history, work_history, value_history = [], [], []
    count_of_changes = 0

    while max_limit(work_iteration, work_sys.architecture[architecture_type], 'selection'):
        work_history.append(dc(work_iteration))
        choice_iteration = compare_lists(available_set, work_iteration)['a1only']
        if fixed_set is not None:
            choice_iteration = compare_lists(choice_iteration, fixed_set)['a1only']
        choice_history.append(choice_iteration)
        if len(choice_iteration) == 0:
            if status_check:
                print('No selections possible')
            break
        iteration_cases = []
        values = []
        if no_select and min_limit(work_iteration, work_sys.architecture[architecture_type], 'selection'):
            iteration_cases.append(dc(work_iteration))
            values.append(work_sys.architecture_cost_calculator(architecture_type, iteration_cases[-1])['cost'])
        for i in range(0, len(choice_iteration)):
            iteration_cases.append(dc(work_iteration))
            iteration_cases[-1].append(choice_iteration[i])
            values.append(work_sys.architecture_cost_calculator(architecture_type, iteration_cases[-1])['cost'])
        value_history.append(values)
        target_idx = item_index_from_policy(values, policy)
        work_iteration = dc(iteration_cases[target_idx])
        if len(compare_lists(work_iteration, work_history[-1])['a1only']) == 0:
            if status_check:
                print('No valuable selections')
            break
        count_of_changes += 1
        if number_of_changes is not None and count_of_changes == number_of_changes:
            if status_check:
                print('Maximum number of changes done')
            break
    work_history.append(work_iteration)
    return {'work_set': work_iteration, 'work_history': work_history, 'choice_history': choice_history, 'value_history': value_history, 'time': time.time() - t_start}


def greedy_architecture_rejection(sys_model, architecture_type, number_of_changes=None, fixed_set=None, failure_set=None, policy="min", t_start=time.time(), no_reject=False, status_check=False):
    work_sys, available_set, work_iteration = initialize_greedy(sys_model, architecture_type, failure_set)
    choice_history, work_history, value_history = [], [], []
    count_of_changes = 0

    while min_limit(work_iteration, work_sys.architecture[architecture_type], 'rejection'):
        work_history.append(dc(work_iteration))
        choice_iteration = dc(work_iteration)
        if fixed_set is not None:
            choice_iteration = compare_lists(choice_iteration, fixed_set)['a1only']
        choice_history.append(choice_iteration)
        if len(choice_iteration) == 0:
            if status_check:
                print('No rejections possible')
            break
        iteration_cases = []
        values = []
        if no_reject and max_limit(work_iteration, work_sys.architecture[architecture_type], 'rejection'):
            iteration_cases.append(dc(work_iteration))
            values.append(work_sys.architecture_cost_calculator(architecture_type, iteration_cases[-1])['cost'])
        for i in range(0, len(choice_iteration)):
            iteration_cases.append([k for k in work_iteration if k != choice_iteration[i]])
            values.append(work_sys.architecture_cost_calculator(architecture_type, iteration_cases[-1])['cost'])
        value_history.append(values)
        target_idx = item_index_from_policy(values, policy)
        work_iteration = dc(iteration_cases[target_idx])
        if len(compare_lists(work_iteration, work_history[-1])['a2only']) == 0:
            if status_check:
                print('No valuable rejections')
            break
        count_of_changes += 1
        if number_of_changes is not None and count_of_changes == number_of_changes:
            if status_check:
                print('Maximum number of changes done')
            break
    work_history.append(work_iteration)
    return {'work_set': work_iteration, 'work_history': work_history, 'choice_history': choice_history, 'value_history': value_history, 'time': time.time() - t_start}


def initialize_greedy(sys_model, architecture_type, failure_set):
    if not isinstance(sys_model, System):
        raise Exception('Incorrect data type')
    work_sys = dc(sys_model)
    available_set = dc(work_sys.architecture[architecture_type]['available'])
    work_iteration = dc(work_sys.architecture[architecture_type]['active'])
    if failure_set is not None:
        work_iteration = compare_lists(work_iteration, failure_set)['a1only']
        available_set = compare_lists(available_set, failure_set)['a1only']
    return work_sys, available_set, work_iteration


def greedy_simultaneous(sys_model, architecture_type, iterations=1, changes_per_iteration=1, fixed_set=None, failure_set=None, policy="min", t_start=time.time(), status_check=False):
    if not isinstance(sys_model, System):
        raise Exception('Incorrect data type')
    work_sys = dc(sys_model)
    work_iteration = work_sys.architecture[architecture_type]['active']
    work_history = [work_iteration]
    value_history = []
    for _ in range(0, iterations):
        # Keep same
        iteration_cases = [work_iteration]
        values = [work_sys.architecture_cost_calculator(architecture_type, iteration_cases[-1])['cost']]
        all_values = [values[-1]]

        # Select one
        select = greedy_architecture_selection(sys_model, architecture_type, number_of_changes=changes_per_iteration, fixed_set=fixed_set, failure_set=failure_set, policy=policy, t_start=t_start, no_select=True, status_check=status_check)
        iteration_cases.append(select['work_set'])
        values.append(work_sys.architecture_cost_calculator(architecture_type, iteration_cases[-1])['cost'])
        all_values += select['value_history']

        # Reject one
        reject = greedy_architecture_rejection(sys_model, architecture_type, number_of_changes=changes_per_iteration, fixed_set=fixed_set, failure_set=failure_set, policy=policy, t_start=t_start, no_reject=True, status_check=status_check)
        iteration_cases.append(reject['work_set'])
        values.append(work_sys.architecture_cost_calculator(architecture_type, iteration_cases[-1])['cost'])
        all_values += reject['value_history']

        # Swap: add then drop
        sys_select = dc(work_sys)
        sys_select.architecture[architecture_type]['min'] += 1
        sys_select.architecture[architecture_type]['max'] += 1
        swap_select1 = greedy_architecture_selection(sys_select, architecture_type, number_of_changes=changes_per_iteration, fixed_set=fixed_set, failure_set=failure_set, policy=policy, t_start=t_start, no_select=True, status_check=False)
        work_sys.active_architecture_update(swap_select1['work_set'], architecture_type)
        swap_reject1 = greedy_architecture_rejection(work_sys, architecture_type, number_of_changes=changes_per_iteration, fixed_set=fixed_set, failure_set=failure_set, policy=policy, t_start=t_start, status_check=status_check)
        iteration_cases.append(swap_reject1['work_set'])
        values.append(work_sys.architecture_cost_calculator(architecture_type, iteration_cases[-1])['cost'])
        all_values += swap_reject1['value_history']

        target_idx = item_index_from_policy(values, policy)
        work_iteration = iteration_cases[target_idx]
        work_history.append(work_iteration)
        value_history.append(all_values)
        if len(compare_lists(work_history[-1], work_history[-2])['a1only']) == 0:
            print('No changes to work_set')
            break

    return {'work_set': work_iteration, 'work_history': work_history, 'value_history': value_history, 'time': time.time()-t_start}


# def simulate_system_fixed_architecture(S, T_sim=100):
#     if not isinstance(S, System):
#         raise Exception('Incorrect data type')
#
#     actuator_update = greedy_simultaneous(S, 'B', status_check=True)
#     S.active_architecture_update(actuator_update['work_'], 'B')
