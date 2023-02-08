import numpy as np
import networkx as netx
import random
import time
import scipy as scp
from copy import deepcopy as dc

import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg
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
        default_graph_model = {'number_of_nodes': 10, 'type': 'rand', 'self_loop': True, 'rho': 1}
        if graph_model is None:
            graph_model = {}
        for key in default_graph_model:
            if key not in graph_model:
                graph_model[key] = default_graph_model[key]
        self.graph_initialize(graph_model)
        self.architecture = {'B': {}, 'C': {}}
        self.metric_model = {'type1': 'x', 'type2': 'scalar', 'type3': 'scalar'}
        self.trajectory = {'X0': np.identity(self.dynamics['number_of_nodes']),
                           'x': [], 'x_estimate': [], 'enhanced': [], 'u': [],
                           'cost': {'running': 0, 'switching': 0, 'control': 0, 'stage': 0, 'predicted': [], 'true': []},
                           'P': [], 'E': [], 'P_enhanced': []}
        self.additive = {'W': 0.5*np.identity(self.dynamics['number_of_nodes']),
                         'V': 0.5*np.identity(self.dynamics['number_of_nodes'])}
        self.initialize_initial_conditions(initial_conditions)
        self.noise = {}
        self.initialize_architecture(architecture)
        if additive is not None:
            self.initialize_additive_noise(additive)
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
                G = netx.from_numpy_array(A)
            elif graph_model['type'] == 'path':
                G = netx.generators.classic.path_graph(self.dynamics['number_of_nodes'])
            else:
                raise Exception('Check graph model')
            connected_network_check = netx.algorithms.components.is_connected(G)

        self.dynamics['Adj'] = netx.to_numpy_array(G)
        if 'self_loop' not in graph_model or graph_model['self_loop']:
            self.dynamics['Adj'] += np.identity(self.dynamics['number_of_nodes'])
        self.dynamics['A'] = self.dynamics['Adj'] * graph_model['rho'] / np.max(np.abs(np.linalg.eigvals(self.dynamics['Adj'])))
        self.dynamics['rho'] = graph_model['rho']

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
            self.random_architecture(n_select=architecture['rand'])

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
            if parameters['algorithm'] == "select":
                self.architecture[parameters['architecture_type']]['active'] = self.architecture[parameters['architecture_type']]['active'] + [parameters['id']]
            elif parameters['algorithm'] == "reject":
                self.architecture[parameters['architecture_type']]['active'] = [i for i in self.architecture[parameters['architecture_type']]['active'] if i != parameters['id']]
            else:
                raise Exception('Check algorithm')
        for a in ['B', 'C']:
            self.architecture[a]['history'].append(dc(self.architecture[a]['active']))
        self.architecture_active_to_matrix(parameters['architecture_type'])
        # self.architecture_costs()

    def active_architecture_duplicate(self, S):
        for a in ['B', 'C']:
            self.architecture[a]['active'] = S.architecture[a]['active']
        for a in ['B', 'C']:
            self.architecture[a]['history'].append(dc(self.architecture[a]['active']))
        self.architecture_active_to_matrix()

    def architecture_active_to_matrix(self, architecture_type=None):
        if architecture_type is None:
            architecture_type = ['B', 'C']
        for i in architecture_type:
            self.architecture[i]['matrix'] = np.zeros(
                (self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes']))
            for k in range(0, len(self.architecture[i]['active'])):
                if i == 'B':
                    self.architecture[i]['matrix'][:, k] = self.architecture[i]['set'][self.architecture[i]['active'][k]]
                elif i == 'C':
                    self.architecture[i]['matrix'][k, :] = self.architecture[i]['set'][self.architecture[i]['active'][k]]

    def random_architecture(self, architecture_type=None, n_select=None):
        if architecture_type is None:
            architecture_type = ['B', 'C']
        for i in architecture_type:
            if n_select is not None:
                n_arch = n_select
            else:
                n_arch = max((self.architecture[i]['min'] + self.architecture[i]['max']) // 2, self.architecture[i]['min'])
            sample_ids = random.sample(self.architecture[i]['available'], k=n_arch)
            for j in sample_ids:
                self.active_architecture_update({'id': j, 'architecture_type': i, 'algorithm': 'select'})
        for i in architecture_type:
            self.architecture[i]['history'] = [self.architecture[i]['active']]

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
        for key in ['x', 'x_estimate']:
            if initial_conditions is not None and key in initial_conditions:
                self.trajectory[key].append(initial_conditions[key])
            else:
                self.trajectory[key].append(np.squeeze(np.random.default_rng().multivariate_normal(np.zeros(self.dynamics['number_of_nodes']), self.trajectory['X0'], 1).T))
        self.trajectory['enhanced'].append(np.squeeze(np.concatenate((self.trajectory['x'][-1], self.trajectory['x_estimate'][-1]))))

    def noise_gen(self):
        self.noise['w'] = np.random.default_rng().multivariate_normal(np.zeros(self.dynamics['number_of_nodes']), self.additive['W'])
        self.noise['v'] = np.random.default_rng().multivariate_normal(np.zeros(self.dynamics['number_of_nodes']), self.additive['V'])
        self.noise['enhanced_vector'] = np.concatenate((self.noise['w'], self.noise['v']))

    def available_choices(self, algorithm, fixed_architecture=None):
        choices = []
        for a in ['B', 'C']:
            if algorithm in ['select', 'reject']:
                if algorithm == 'select' and max_limit(self.architecture[a], algorithm):
                    choice_a = compare_lists(self.architecture[a]['available'], self.architecture[a]['active'])['a1only']
                elif algorithm == 'reject' and min_limit(self.architecture[a], algorithm):
                    choice_a = compare_lists(self.architecture[a]['available'], self.architecture[a]['active'])['common']
                else:
                    continue
                if fixed_architecture is not None and a in fixed_architecture:
                    choice_a = compare_lists(choice_a, fixed_architecture[a])['a1only']
                for i in choice_a:
                    choices.append({'architecture_type': a, 'id': i})
            else:
                raise Exception('Check algorithm')
        return choices

    def enhanced_system_matrix(self):
        BK = self.architecture['B']['matrix'] @ self.architecture['B']['gain']
        LC = self.architecture['C']['gain'] @ self.architecture['C']['matrix']
        self.dynamics['enhanced'] = np.block([
            [self.dynamics['A'], -BK],
            [LC @ self.dynamics['A'], self.dynamics['A'] - BK - (LC @ self.dynamics['A'])]])
        self.noise['enhanced_noise_matrix'] = np.block([
            [np.identity(self.dynamics['number_of_nodes']), np.zeros((self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes']))],
            [LC, LC]])
        self.noise['enhanced_noise_expectation'] = self.noise['enhanced_noise_matrix'].T @ np.block([
            [self.additive['W'], np.zeros((self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes']))],
            [np.zeros((self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes'])), self.additive['V']]
            ]) @ self.noise['enhanced_noise_matrix']
        self.dynamics['enhanced_stage_cost'] = np.block([
            [self.architecture['B']['cost']['Q'], np.zeros((self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes']))],
            [np.zeros((self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes'])), self.architecture['B']['gain'].T @ self.architecture['B']['cost']['R1'] @ self.architecture['B']['gain']]])
        self.dynamics['enhanced_terminal_cost'] = np.block([
            [self.architecture['B']['cost']['Q'], np.zeros((self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes']))],
            [np.zeros((self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes'])), np.zeros((self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes']))]])

    def optimal_feedback_control_cost_gain(self):
        self.architecture['B']['gain'] = np.linalg.inv(self.architecture['B']['matrix'].T @ self.trajectory['P'][-1] @ self.architecture['B']['matrix'] + self.architecture['B']['cost']['R1']) @ self.architecture['B']['matrix'].T @ self.trajectory['P'][-1] @ self.dynamics['A']

    def optimal_feedback_estimation_gain(self):
        self.architecture['C']['gain'] = self.trajectory['E'][-1] @ self.architecture['C']['matrix'].T @ np.linalg.inv(self.architecture['C']['matrix'] @ self.trajectory['E'][-1] @ self.architecture['C']['matrix'].T + self.architecture['C']['cost']['R1'])

    def optimal_control_feedback_wrapper(self):
        self.trajectory['P'].append(scp.linalg.solve_discrete_are(self.dynamics['A'], self.architecture['B']['matrix'], self.architecture['B']['cost']['Q'], self.architecture['B']['cost']['R1']))
        self.optimal_feedback_control_cost_gain()

    def optimal_estimation_feedback_wrapper(self):
        self.trajectory['E'].append(scp.linalg.solve_discrete_are(self.dynamics['A'].T, self.architecture['C']['matrix'].T, self.additive['W'], self.additive['V']))
        self.optimal_feedback_estimation_gain()

    def enhanced_lyapunov_control_cost(self, T_sim=30):
        self.trajectory['cost']['control'] = 0
        self.trajectory['P_enhanced'] = [dc(self.dynamics['enhanced_terminal_cost'])]
        for t in range(0, T_sim):
            self.trajectory['cost']['control'] += np.trace(self.trajectory['P_enhanced'][-1] @ self.noise['enhanced_noise_expectation'])
            self.trajectory['P_enhanced'].append(self.dynamics['enhanced'].T @ self.trajectory['P_enhanced'][-1] @ self.dynamics['enhanced'] + self.dynamics['enhanced_stage_cost'])
        self.trajectory['cost']['control'] += np.trace(self.trajectory['P_enhanced'][-1] @ self.noise['enhanced_noise_expectation'])
        self.trajectory['cost']['control'] += np.squeeze(self.trajectory['enhanced'][-1].T @ self.trajectory['P_enhanced'][-1] @ self.trajectory['enhanced'][-1])

    def enhanced_stage_control_cost(self):
        self.trajectory['cost']['stage'] = np.squeeze(self.trajectory['enhanced'][-1].T @ self.dynamics['enhanced_stage_cost'] @ self.trajectory['enhanced'][-1])

    def system_one_step_update_enhanced(self):
        self.noise_gen()
        self.trajectory['enhanced'].append((self.dynamics['enhanced'] @ self.trajectory['enhanced'][-1]) + (self.noise['enhanced_noise_matrix'] @ self.noise['enhanced_vector']))
        self.trajectory['x'].append(self.trajectory['enhanced'][-1][0:self.dynamics['number_of_nodes']])
        self.trajectory['x_estimate'].append(self.trajectory['enhanced'][-1][self.dynamics['number_of_nodes']:])

    def architecture_costs(self):
        self.trajectory['cost']['running'] = 0
        self.trajectory['cost']['switching'] = 0
        architecture_type = ['B', 'C']
        for a in architecture_type:
            active_vector = np.zeros(self.dynamics['number_of_nodes'])
            history_vector = np.zeros(self.dynamics['number_of_nodes'])
            for i in self.architecture[a]['active']:
                active_vector[i] = 1
            if len(self.architecture[a]['history']) > 0:
                for i in self.architecture[a]['history'][-1]:
                    history_vector[i] = 1
            if self.metric_model['type2'] == 'matrix':
                self.trajectory['cost']['running'] += active_vector.T @ self.architecture[a]['cost']['R2'] @ active_vector
            elif self.metric_model['type2'] == 'scalar':
                self.trajectory['cost']['running'] += self.architecture[a]['cost']['R2'] * np.inner(active_vector, active_vector)
            else:
                raise Exception('Check Metric for Type 2 - Running Costs')
            if self.metric_model['type3'] == 'matrix':
                self.trajectory['cost']['switching'] += (active_vector - history_vector).T @ self.architecture[a]['cost']['R3'] @ (active_vector - history_vector)
            elif self.metric_model['type3'] == 'scalar':
                self.trajectory['cost']['switching'] += self.architecture[a]['cost']['R3'] * np.inner(active_vector - history_vector, active_vector - history_vector)
            else:
                raise Exception('Check Metric for Type 2 - Running Costs')

    def cost_wrapper_enhanced_prediction(self):
        self.optimal_estimation_feedback_wrapper()
        self.optimal_control_feedback_wrapper()
        self.architecture_costs()
        self.enhanced_system_matrix()
        self.enhanced_lyapunov_control_cost()
        self.trajectory['cost']['predicted'].append(0)
        for i in ['running', 'switching', 'control']:
            if self.trajectory['cost'][i] < 0:
                raise Exception('Negative cost: ', i)
            self.trajectory['cost']['predicted'][-1] += self.trajectory['cost'][i]

    def cost_wrapper_enhanced_true(self):
        self.optimal_estimation_feedback_wrapper()
        self.optimal_control_feedback_wrapper()
        self.architecture_costs()
        self.enhanced_system_matrix()
        self.enhanced_stage_control_cost()
        self.trajectory['cost']['true'].append(0)
        for i in ['running', 'switching', 'stage']:
            if self.trajectory['cost'][i] < 0:
                raise Exception('Negative cost: ', i)
            self.trajectory['cost']['true'][-1] += self.trajectory['cost'][i]

    def architecture_update_check(self):
        check = False
        for i in ['B', 'C']:
            architecture_compare = compare_lists(self.architecture[i]['history'][-2], self.architecture[i]['history'][-1])
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

    def display_active_architecture(self):
        for a in ['B', 'C']:
            print(a, ':', self.architecture[a]['active'])

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
                               [B_mat.T, np.zeros((self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes'])),
                                np.zeros((self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes']))],
                               [C_mat, np.zeros((self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes'])),
                                np.zeros((self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes']))]])
        return net_matrix

    def display_graph_gen(self, node_pos=None):
        net_matrix = self.network_matrix()
        G = netx.from_numpy_array(net_matrix)
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
            G_base = netx.from_numpy_array(self.dynamics['A'])
            netx.relabel_nodes(G_base, node_labels, copy=False)
            node_pos = netx.circular_layout(G_base)
        node_pos = netx.spring_layout(G, pos=node_pos, fixed=[str(i + 1) for i in range(0, self.dynamics['number_of_nodes'])])
        return {'G': G, 'pos': node_pos, 'node_color': nc}

    # def animate_architecture(self):
    #     fig = plt.figure(figsize=(6, 4))
    #     # fig = plt.subplots(2, 3, figsize=(6, 4))
    #     grid = fig.add_gridspec(2, 3)
    #     ax_network = fig.add_subplot(grid[0:2, 0:2])
    #     ax_trajectory = fig.add_subplot(grid[0, 2])
    #     ax_cost = fig.add_subplot(grid[1, 2], sharex=ax_trajectory)
    #
    #     ax_network.axis('off')
    #     min_lim = -1.7
    #     max_lim = 1.7
    #     ax_network.set_xlim(min_lim, max_lim)
    #     ax_network.set_ylim(min_lim, max_lim)
    #     Sys_dummy = dc(self)
    #     Sys_dummy.architecture['B']['active'] = Sys_dummy.architecture['B']['available']
    #     Sys_dummy.architecture['C']['active'] = Sys_dummy.architecture['C']['available']
    #     Sys_dummy.architecture_active_to_matrix()
    #     pos_gen = Sys_dummy.display_graph_gen()['pos']
    #     node_pos = {i: pos_gen[i] for i in pos_gen if i.isnumeric()}
    #     t_range = np.arange(0, len(self.architecture['B']['history']), 1)
    #
    #     for i in range(0, self.dynamics['number_of_nodes']):
    #         ax_trajectory.plot(range(0, len(self.trajectory['x'])), [x_t[i] for x_t in self.trajectory['x']])
    #         ax_trajectory.plot(range(0, len(self.trajectory['x_estimate'])), [x_t[i] for x_t in self.trajectory['x_estimate']])
    #     ax_trajectory.set_ylabel('x-trajectory')
    #
    #     ax_cost.plot(range(0, len(self.trajectory['cost']['estimate_total'])), self.trajectory['cost']['estimate_total'])
    #     ax_cost.plot(range(0, len(self.trajectory['cost']['true_total'])), self.trajectory['cost']['true_total'])
    #     ax_cost.set_xlabel('time')
    #     ax_cost.set_ylabel('cost')
    #
    #     def update(t):
    #         ax_network.clear()
    #         sys_t = dc(self)
    #         for a in ['B', 'C']:
    #             sys_t.architecture[a]['active'] = self.architecture[a]['history'][t]
    #         sys_t.architecture_active_to_matrix()
    #         sys_t_plot = sys_t.display_graph_gen(node_pos)
    #         netx.draw_networkx(sys_t_plot['G'], ax=ax_network, pos=sys_t_plot['pos'], node_color=sys_t_plot['node_color'])
    #         ax_network.set_title('t=' + str(t))
    #         ax_network.set_xlim(min_lim, max_lim)
    #         ax_network.set_ylim(min_lim, max_lim)
    #
    #     ani = matplotlib.animation.FuncAnimation(fig, update, frames=t_range, interval=1000, repeat=False)
    #     ani.save("Test.mp4")
    #     plt.show()

    def cost_plot(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(6, 4))
            grid = fig.add_gridspec(2, 1)
            ax_trajectory = fig.add_subplot(grid[0, 0])
            ax_cost = fig.add_subplot(grid[1, 0], sharex=ax_trajectory)
        else:
            ax_trajectory = ax[0]
            ax_cost = ax[1]

        for i in range(0, self.dynamics['number_of_nodes']):
            ax_trajectory.plot(range(0, len(self.trajectory['x'])), [x_t[i] for x_t in self.trajectory['x']])
            ax_trajectory.plot(range(0, len(self.trajectory['x_estimate'])), [x_t[i] for x_t in self.trajectory['x_estimate']])
        ax_trajectory.set_ylabel('x-trajectory')

        ax_cost.plot(range(0, len(self.trajectory['cost']['estimate_total'])), self.trajectory['cost']['estimate_total'])
        ax_cost.set_xlabel('time')
        ax_cost.set_ylabel('cost')
        plt.show()

    def cost_plot_history(self):
        # for i in range(0, self.dynamics['number_of_nodes']):
        #     print([x_t[i] for x_t in self.trajectory['x']])
        cumulative_cost = [self.trajectory['cost']['true'][0]]
        for i in range(1, len(self.trajectory['cost']['true'])):
            cumulative_cost.append(self.trajectory['cost']['true'][i]+cumulative_cost[-1])
        fig = plt.figure(figsize=(6, 4))
        grid = fig.add_gridspec(2, 1)
        ax_trajectory = fig.add_subplot(grid[0, 0])
        ax_cost = fig.add_subplot(grid[1, 0], sharex=ax_trajectory)
        for i in range(0, self.dynamics['number_of_nodes']):
            ax_trajectory.plot(range(0, len(self.trajectory['x'])), [x_t[i] for x_t in self.trajectory['x']])
            # ax_trajectory.plot(range(0, len(self.trajectory['x_estimate'])), [x_t[i] for x_t in self.trajectory['x_estimate']])
        ax_trajectory.set_ylabel('x-trajectory')
        ax_cost.plot(range(0, len(cumulative_cost)), cumulative_cost)
        ax_cost.set_xlabel('time')
        ax_cost.set_ylabel('cost')
        ax_cost.set_yscale('log')
        plt.show()

    def architecture_history_plot(self, f_name=None):
        fig = plt.figure(figsize=(6, 4))
        grid = fig.add_gridspec(2, 2)
        ax_B = fig.add_subplot(grid[0, 0])
        ax_C = fig.add_subplot(grid[1, 0], sharex=ax_B)
        ax_Bhist = fig.add_subplot(grid[0, 1], sharey=ax_B)
        ax_Chist = fig.add_subplot(grid[1, 1], sharex=ax_Bhist, sharey=ax_C)
        B_hist = np.zeros(self.dynamics['number_of_nodes'])
        C_hist = np.zeros(self.dynamics['number_of_nodes'])
        B_list = np.zeros((self.dynamics['number_of_nodes'], len(self.architecture['B']['history'])))
        C_list = np.zeros((self.dynamics['number_of_nodes'], len(self.architecture['C']['history'])))
        for t in range(0, len(self.architecture['B']['history'])):
            for i in self.architecture['B']['history'][t]:
                B_list[i, t] = 1
                B_hist[i] += 1
            for i in self.architecture['C']['history'][t]:
                C_list[i, t] = 1
                C_hist[i] += 1
        ax_B.imshow(B_list)
        ax_C.imshow(C_list)
        ax_Bhist.hist(B_hist, bins=self.dynamics['number_of_nodes'], orientation="horizontal")
        ax_Chist.hist(C_hist, bins=self.dynamics['number_of_nodes'], orientation="horizontal")
        ax_B.set_ylabel('Sensor ID/Node')
        ax_C.set_ylabel('Sensor ID/Node')
        ax_C.set_xlabel('Time')
        ax_B.set_title('B')
        ax_C.set_title('C')
        ax_Chist.set_xlabel('Number of uses')
        if f_name is None:
            f_name = "images/architecture_history_rho" + str(self.dynamics['rho']) + ".png"
        plt.savefig(f_name)
        plt.show()


def matrix_convergence_check(A, B, accuracy=10 ** (-3), check_type=None):
    np_norm_methods = ['inf', 'fro', 2, None]
    if check_type is None:
        return np.allclose(A, B, atol=accuracy, rtol=accuracy)
    elif check_type in np_norm_methods:
        return np.norm(A - B, ord=check_type) < accuracy
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


def print_choices(choices):
    for c in choices:
        print(c)


def greedy_architecture_selection(sys, number_of_changes=None, policy="min", no_select=False, status_check=False, t_start=time.time()):
    if not isinstance(sys, System):
        raise Exception('Check data type')
    if status_check:
        print('\nSelection\n')
    work_iteration = dc(sys)
    # work_iteration.display_active_architecture()
    work_history, choice_history, value_history = [], [], []
    selection_check = (max_limit(work_iteration.architecture['B'], 'select') or max_limit(work_iteration.architecture['C'], 'select'))
    count_of_changes = 0
    while selection_check:
        if status_check:
            print('\n Change:', count_of_changes)
            work_iteration.display_active_architecture()
        work_history.append(work_iteration)
        choices = work_iteration.available_choices('select')
        if len(choices) == 0:
            if status_check:
                print('No selections available')
            selection_check = False
            break
        if no_select and (min_limit(work_iteration.architecture['B'], 'select') or min_limit(work_iteration.architecture['C'], 'select')):
            choices.append({'architecture_type': None})
        choice_history.append(choices)
        for i in choices:
            i['algorithm'] = 'select'
            test_sys = dc(work_iteration)
            test_sys.active_architecture_update(i)
            test_sys.cost_wrapper_enhanced_prediction()
            i['value'] = test_sys.trajectory['cost']['predicted'][-1]
        if status_check:
            print_choices(choices)
        value_history.append([i['value'] for i in choices])
        target_idx = item_index_from_policy(value_history[-1], policy)
        work_iteration.active_architecture_update(choices[target_idx])
        work_iteration.cost_wrapper_enhanced_prediction()
        if not work_iteration.architecture_update_check():
            if status_check:
                print('No valuable architecture updates')
            break
        count_of_changes += 1
        if number_of_changes is not None and count_of_changes == number_of_changes:
            if status_check:
                print('Maximum number of changes done')
            break
        selection_check = (max_limit(work_iteration.architecture['B'], 'select') or max_limit(work_iteration.architecture['C'], 'select'))
        if status_check:
            print('Selection check: B: ', max_limit(work_iteration.architecture['B'], 'select'), ' |C: ', max_limit(work_iteration.architecture['C'], 'select'))
    return_set = dc(sys)
    return_set.active_architecture_duplicate(work_iteration)
    return_set.cost_wrapper_enhanced_prediction()
    if status_check:
        return_set.display_active_architecture()
    work_history.append(return_set)
    return {'work_set': return_set, 'work_history': work_history, 'choice_history': choice_history, 'value_history': value_history, 'time': time.time() - t_start, 'steps': count_of_changes}


def greedy_architecture_rejection(sys, number_of_changes=None, policy="min", no_reject=False, status_check=False, t_start=time.time()):
    if not isinstance(sys, System):
        raise Exception('Check data type')
    if status_check:
        print('\nRejection\n')
    work_iteration = dc(sys)
    work_history, choice_history, value_history = [], [], []
    rejection_check = (min_limit(work_iteration.architecture['B'], 'reject') or min_limit(work_iteration.architecture['C'], 'reject'))
    count_of_changes = 0
    while rejection_check:
        if status_check:
            print('Change:', count_of_changes)
            work_iteration.display_active_architecture()
        work_history.append(work_iteration)
        choices = work_iteration.available_choices('reject')
        if len(choices) == 0:
            if status_check:
                print('No selections available')
            rejection_check = False
            break
        if no_reject and (max_limit(work_iteration.architecture['B'], 'select') or max_limit(work_iteration.architecture['C'], 'reject')):
            choices.append({'architecture_type': None})
        choice_history.append(choices)
        for i in choices:
            i['algorithm'] = 'reject'
            test_sys = dc(work_iteration)
            test_sys.active_architecture_update(i)
            test_sys.cost_wrapper_enhanced_prediction()
            i['value'] = test_sys.trajectory['cost']['predicted'][-1]
        if status_check:
            print_choices(choices)
        target_idx = item_index_from_policy([i['value'] for i in choices], policy)
        value_history.append([i['value'] for i in choices])
        work_iteration.active_architecture_update(choices[target_idx])
        work_iteration.cost_wrapper_enhanced_prediction()
        if not work_iteration.architecture_update_check():
            if status_check:
                print('No valuable architecture updates')
            break
        count_of_changes += 1
        if number_of_changes is not None and count_of_changes == number_of_changes:
            if status_check:
                print('Maximum number of changes done')
            break
        rejection_check = min_limit(work_iteration.architecture['B'], 'reject') or min_limit(work_iteration.architecture['C'], 'reject')
        if status_check:
            print('Selection check: B: ', min_limit(work_iteration.architecture['B'], 'select'), ' |C: ', min_limit(work_iteration.architecture['C'], 'select'))
    return_set = dc(sys)
    return_set.active_architecture_duplicate(work_iteration)
    return_set.cost_wrapper_enhanced_prediction()
    if status_check:
        return_set.display_active_architecture()
    work_history.append(return_set)
    return {'work_set': return_set, 'work_history': work_history, 'choice_history': choice_history, 'value_history': value_history, 'time': time.time() - t_start, 'steps': count_of_changes}


def greedy_simultaneous(sys, iterations=None, changes_per_iteration=1, fixed_set=None, failure_set=None, policy="min", t_start=time.time(), status_check=False):
    if not isinstance(sys, System):
        raise Exception('Incorrect data type')
    if status_check:
        print('\nSimultaneous\n')
    work_iteration = dc(sys)
    # work_iteration.cost_wrapper_enhanced_prediction()
    # print(type(work_iteration.trajectory['cost']['predicted'][-1]))
    work_history = [work_iteration]
    value_history = []
    iteration_stop = False
    iteration_count = 0
    while not iteration_stop:
        # print('\n')
        # print('Iteration:', iteration_count)
        # work_iteration.display_active_architecture()
        # print('\n')
        if status_check:
            print('Iteration:', iteration_count)
            work_iteration.display_active_architecture()
        # Keep same
        values = []
        iteration_cases = []

        # iteration_cases.append(dc(work_iteration))
        # iteration_cases[-1].active_architecture_update({'architecture_type': None})
        # iteration_cases[-1].cost_wrapper_enhanced_prediction()
        # values.append(iteration_cases[-1].trajectory['cost']['predicted'][-1])
        # if status_check:
        #     print('Maintain Value: ', values[-1])
        #     iteration_cases[-1].display_active_architecture()

        # Select one
        select = greedy_architecture_selection(work_iteration, number_of_changes=changes_per_iteration, policy=policy, t_start=t_start, no_select=True, status_check=status_check)
        iteration_cases.append(select['work_set'])
        values.append(iteration_cases[-1].trajectory['cost']['predicted'][-1])
        if status_check:
            print('Add Value: ', values[-1])
            iteration_cases[-1].display_active_architecture()

        # Reject one
        reject = greedy_architecture_rejection(work_iteration, number_of_changes=changes_per_iteration, policy=policy, t_start=t_start, no_reject=True, status_check=status_check)
        iteration_cases.append(reject['work_set'])
        values.append(iteration_cases[-1].trajectory['cost']['predicted'][-1])
        if status_check:
            print('Subtract Value: ', values[-1])
            iteration_cases[-1].display_active_architecture()

        # Swap: add then drop
        if status_check:
            print('\nSwap\n')
        sys_swap = dc(work_iteration)
        sys_swap.architecture_limit_modifier(min_mod=1, max_mod=1)
        sys_swap = greedy_architecture_selection(sys_swap, number_of_changes=changes_per_iteration, policy=policy, t_start=t_start, no_select=False, status_check=status_check)['work_set']
        sys_swap.architecture_limit_modifier(min_mod=-1, max_mod=-1)
        sys_swap = greedy_architecture_rejection(sys_swap, number_of_changes=changes_per_iteration, policy=policy, t_start=t_start, status_check=status_check)['work_set']
        sys_swap.trajectory['cost']['predicted'] = sys_swap.trajectory['cost']['predicted'][:-2] + [sys_swap.trajectory['cost']['predicted'][-1]]
        iteration_cases.append(sys_swap)
        values.append(iteration_cases[-1].trajectory['cost']['predicted'][-1])
        if status_check:
            print('Swap Value: ', values[-1])
            iteration_cases[-1].display_active_architecture()

        target_idx = item_index_from_policy(values, policy)
        work_iteration = dc(iteration_cases[target_idx])
        work_history.append(work_iteration)
        value_history.append(values)
        if status_check:
            print('Iteration:', iteration_count)
            print(values)
            work_iteration.display_active_architecture()
        if not work_iteration.architecture_update_check():
            if status_check:
                print('No changes to work_set')
            break
        iteration_count += 1
        if iterations is not None and iteration_count >= iterations:
            if status_check:
                print('Max iterations of simultaneous greedy')
            iteration_stop = True
    return_set = dc(sys)
    return_set.active_architecture_duplicate(work_iteration)
    return_set.cost_wrapper_enhanced_prediction()
    if status_check:
        return_set.display_active_architecture()
    work_history.append(return_set)
    return {'work_set': return_set, 'work_history': work_history, 'value_history': value_history, 'time': time.time() - t_start, 'steps': iteration_count}


def simultaneous_cost_plot(value_history):
    for i in value_history:
        print(i)
    fig = plt.figure(figsize=(6, 4))
    grid = fig.add_gridspec(1, 1)
    ax_cost = fig.add_subplot(grid[0, 0])
    for i in range(0, len(value_history[0])):
        ax_cost.scatter(range(0, len(value_history)), [v[i] for v in value_history])
    ax_cost.legend(['select', 'reject', 'swap'])
    ax_cost.set_yscale('log')
    plt.show()


def cost_plots(cost, f_name=None):
    fig = plt.figure(figsize=(6, 4), )
    grid = fig.add_gridspec(1, 1)
    ax_cost = fig.add_subplot(grid[0, 0])
    for i in cost:
        cumulative_cost = [i[0]]
        for t in range(1, len(i)):
            cumulative_cost.append(cumulative_cost[-1] + i[t])
        ax_cost.plot(range(0, len(cumulative_cost)), cumulative_cost)
    ax_cost.set_xlabel('time')
    ax_cost.set_ylabel('cost')
    ax_cost.set_yscale('log')
    ax_cost.legend(['fixed', 'optimal'])
    ax_cost.set_title('Architecture comparison')
    if f_name is None:
        f_name = "images/cost_trajectory.png"
    else:
        f_name = "images/cost_trajectory"+f_name+".png"
    plt.savefig(f_name)
    plt.show()


if __name__ == "__main__":
    print('Check function file')