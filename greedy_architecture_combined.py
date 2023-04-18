import numpy as np
import networkx as netx
import random
import time
import scipy as scp
from copy import deepcopy as dc
import shelve
import os
import socket
from tqdm import tqdm
from shutil import rmtree
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button, TextBox
import matplotlib.patches as patches
# from matplotlib.ticker import MaxNLocator
import matplotlib.animation
import multiprocessing

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

if socket.gethostname() == 'melap257805':
    datadump_folder_path = 'C:/Users/kxg161630/Box/KarthikGanapathy_Research/SpeedyGreedyAlgorithm/DataDump/'
else:
    datadump_folder_path = 'D:/Box/KarthikGanapathy_Research/SpeedyGreedyAlgorithm/DataDump/'

image_save_folder_path = 'Images/'


class System:
    def __init__(self, graph_model=None, architecture=None, additive=None, initial_conditions=None, simulation_parameters=None):
        self.dynamics = {}
        default_graph_model = {'number_of_nodes': 10, 'type': 'rand', 'self_loop': True, 'rho': 1, 'second_order': False}
        if graph_model is None:
            graph_model = {}
        for key in default_graph_model:
            if key not in graph_model:
                graph_model[key] = default_graph_model[key]
        self.graph_initialize(graph_model)
        self.architecture = {'B': {}, 'C': {}}
        self.metric_model = {'type1': 'x', 'type2': 'scalar', 'type3': 'scalar'}
        self.simulation_parameters = {'T_sim': 200, 'T_predict': 30}
        if simulation_parameters is not None:
            for k in self.simulation_parameters:
                if k in simulation_parameters:
                    self.simulation_parameters[k] = simulation_parameters[k]
        self.trajectory = {'X0': 10*np.identity(self.dynamics['number_of_nodes']),
                           'x': [], 'x_estimate': [], 'X_enhanced': [], 'u': [], 'error_2norm': [], 'error': [],
                           'cost': {'running': 0, 'switching': 0, 'control': 0, 'stage': 0, 'predicted': [], 'true': [], 'initial': 0},
                           'P': {}, 'E': {}, 'P_hist': [], 'E_hist': [], 'Z': []}
        self.trajectory['E_hist'].append(self.trajectory['X0'])
        self.additive = {'W': np.identity(self.dynamics['number_of_nodes']),
                         'V': np.identity(self.dynamics['number_of_nodes'])}

        if additive is not None:
            for k in self.additive:
                if k in additive:
                    self.additive[k] *= additive[k]
        self.initialize_initial_conditions(initial_conditions)
        self.noise = {}
        self.noise_gen()
        if additive is not None and 'type' in additive:
            self.noise_mod(additive['type'], additive['disturbance'])
        self.initialize_architecture(architecture)
        # self.architecture['C']['cost']['R1'] = self.additive['V']
        self.architecture['C']['cost']['Q'] = self.additive['W']
        self.model_name = ''
        self.model_rename()
        self.network_plot_parameters = {}
        self.full_network_position()
        # if additive is not None:
        #     self.initialize_additive_noise(additive)
        # self.enhanced_system_matrix()

    def model_rename(self, name_append=None):
        self.model_name = "model_n" + str(int(self.dynamics['number_of_nodes']/2) if self.dynamics['second_order'] else self.dynamics['number_of_nodes'])
        if self.dynamics['rho'] is not None:
            self.model_name = self.model_name + "_rho" + str(np.round(self.dynamics['rho'], decimals=3))
        else:
            self.model_name = self.model_name + "_rhoNone"
        self.model_name = self.model_name + "_Tp" + str(self.simulation_parameters['T_predict']) + "_arch" + str(self.architecture['B']['max'])
        if 'mod' in self.noise:
            self.model_name = self.model_name + "_" + self.noise['mod']
        if self.dynamics['second_order']:
            self.model_name = self.model_name + "_secondorder"
        if name_append is not None:
            self.model_name = self.model_name + "_" + name_append

    def graph_initialize(self, graph_model):
        connected_network_check = False
        self.dynamics['number_of_nodes'] = graph_model['number_of_nodes']
        G = netx.Graph()
        while not connected_network_check:
            print('Graph network mode: ', graph_model['type'])
            if graph_model['type'] == 'ER':
                if 'p' not in graph_model:
                    graph_model['p'] = 0.3
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
            elif graph_model['type'] == 'cycle':
                G = netx.generators.classic.cycle_graph(self.dynamics['number_of_nodes'])
            elif graph_model['type'] == 'rand_eval':
                if 'p' not in graph_model:
                    graph_model['p'] = 0.2
                A = np.random.rand(self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes'])
                A = A.T + A
                _, V_mat = np.linalg.eig(A)
                e = graph_model['p']*(0.5 - np.random.rand(self.dynamics['number_of_nodes']))
                e = [i*-1 if np.random.rand(1) > 0.5 else i for i in e]
                A = V_mat @ np.diag(e) @ np.linalg.inv(V_mat)
                G = netx.from_numpy_array(A)
                if 'rho' in graph_model and graph_model['rho'] is not None:
                    print('Overriding scaling factor')
                elif 'rho' not in graph_model:
                    graph_model['rho'] = None
            else:
                raise Exception('Check graph model')
            connected_network_check = netx.algorithms.components.is_connected(G)

        self.dynamics['Adj'] = netx.to_numpy_array(G)
        if graph_model['self_loop']:
            self.dynamics['Adj'] += np.identity(self.dynamics['number_of_nodes'])

        if graph_model['second_order']:
            self.dynamics['second_order'] = True
            # self.second_order_network()
            self.dynamics['number_of_nodes'] *= 2
            # self.open_loop_stability_eval()
        else:
            self.dynamics['second_order'] = False
        self.rescale_dynamics(graph_model['rho'])

        # self.dynamics['A'] = self.dynamics['Adj'] * graph_model['rho'] / np.max(np.abs(np.linalg.eigvals(self.dynamics['Adj'])))
        # self.dynamics['ol_eig'] = np.sort(np.linalg.eigvals(self.dynamics['A']))
        # self.dynamics['n_unstable'] = sum([1 for i in self.dynamics['ol_eig'] if i >= 1])
    def second_order_network(self):
        self.dynamics['A'] = np.block([[self.dynamics['Adj'], np.zeros_like(self.dynamics['Adj'])],
                                       [0.5*np.identity(int(self.dynamics['number_of_nodes']/2)), 0.5*np.identity(int(self.dynamics['number_of_nodes']/2))]])

    def rescale_dynamics(self, rho):
        if self.dynamics['second_order']:
            self.second_order_network()
        else:
            self.dynamics['A'] = self.dynamics['Adj']
        self.dynamics['rho'] = rho
        if self.dynamics['rho'] is not None:
            self.dynamics['A'] = rho * self.dynamics['A']/(np.max(np.abs(np.linalg.eigvals(self.dynamics['A']))))
            self.dynamics['rho'] = rho
        self.open_loop_stability_eval()

    def open_loop_stability_eval(self):
        self.dynamics['ol_eig'] = np.sort(np.linalg.eigvals(self.dynamics['A']))
        self.dynamics['n_unstable'] = sum([1 for i in self.dynamics['ol_eig'] if np.abs(i) >= 1])

    def initialize_architecture(self, architecture):
        n = int(self.dynamics['number_of_nodes']/2) if self.dynamics['second_order'] else self.dynamics['number_of_nodes']
        architecture_model = {'min': 1, 'max': n,
                              'cost': {'Q': np.identity(self.dynamics['number_of_nodes']),
                                       'R1': np.identity(n),
                                       'R2': 0,
                                       'R3': 0},
                              'active': [],
                              'matrix': np.zeros((self.dynamics['number_of_nodes'], n)),
                              'indicator': np.zeros(n),
                              'available': range(0, n),
                              'set': [],
                              'history': [],
                              'change_count': 0,
                              'gain': {},
                              'gram': np.zeros_like(self.dynamics['A'])}
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
        n = int(self.dynamics['number_of_nodes']/2) if self.dynamics['second_order'] else self.dynamics['number_of_nodes']
        basis = []
        for i in range(0, n):
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
            self.architecture[parameters['architecture_type']]['active'] = np.sort(np.array(self.architecture[parameters['architecture_type']]['active'])).tolist()
        for a in ['B', 'C']:
            self.architecture[a]['history'].append(dc(self.architecture[a]['active']))
        self.architecture_active_to_matrix()

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
            self.architecture[i]['matrix'] = np.zeros((self.dynamics['number_of_nodes'], len(self.architecture[i]['active'])))
            self.architecture[i]['indicator'] = np.zeros(int(self.dynamics['number_of_nodes']/2) if self.dynamics['second_order'] else self.dynamics['number_of_nodes'])
            for k in range(0, len(self.architecture[i]['active'])):
                if i == 'B':
                    self.architecture[i]['matrix'][:, k] = self.architecture[i]['set'][self.architecture[i]['active'][k]]
                elif i == 'C':
                    self.architecture[i]['matrix'][:, k] = self.architecture[i]['set'][self.architecture[i]['active'][k]]
                else:
                    raise Exception('Check architecture type')
                self.architecture[i]['indicator'][self.architecture[i]['active'][k]] = 1
            if i == 'B':
                self.architecture['B']['cost']['R1_active'] = self.architecture['B']['cost']['R1'][self.architecture['B']['active'], :][:, self.architecture['B']['active']]
            elif i == 'C':
                self.architecture[i]['matrix'] = self.architecture[i]['matrix'].T
                self.additive['V_active'] = self.additive['V'][self.architecture[i]['active'], :][:, self.architecture[i]['active']]
                self.architecture['C']['cost']['R1_active'] = self.additive['V_active']
            else:
                raise Exception('Check architecture type')

    def random_architecture(self, architecture_type=None, n_select=None):
        if architecture_type is None:
            architecture_type = ['B', 'C']
        for i in architecture_type:
            if n_select is not None:
                n_arch = n_select
            else:
                n_arch = max((self.architecture[i]['min'] + self.architecture[i]['max']) // 2, self.architecture[i]['min'])
            self.architecture[i]['active'] = np.sort(random.sample(self.architecture[i]['available'], k=n_arch)).tolist()
            # print(i, ':', self.architecture[i]['active'])
        for i in architecture_type:
            self.architecture[i]['history'] = [self.architecture[i]['active']]
        self.architecture_active_to_matrix()

    def initialize_initial_conditions(self, initial_conditions=None):
        if initial_conditions is not None and 'X0' in initial_conditions:
            self.trajectory['X0'] = initial_conditions['X0']
        for key in ['x', 'x_estimate']:
            if initial_conditions is not None and key in initial_conditions:
                self.trajectory[key].append(initial_conditions[key])
            else:
                self.trajectory[key].append(np.squeeze(np.random.default_rng().multivariate_normal(np.zeros(self.dynamics['number_of_nodes']), self.trajectory['X0'], 1).T))
        self.trajectory['X_enhanced'].append(np.squeeze(np.concatenate((self.trajectory['x'][-1], self.trajectory['x_estimate'][-1]))))
        self.trajectory['error'].append(self.trajectory['x'][-1]-self.trajectory['x_estimate'][-1])
        self.trajectory['error_2norm'].append(np.linalg.norm(self.trajectory['error'][-1], ord=2))

    def noise_gen(self):
        self.noise['block_noise_covariance'] = scp.linalg.block_diag(self.additive['W'], self.additive['V'])
        self.noise['noise_sim'] = np.random.default_rng().multivariate_normal(np.zeros(2*self.dynamics['number_of_nodes']), self.noise['block_noise_covariance'], self.simulation_parameters['T_sim']+1)

    def noise_mod(self, noise_type, disturbance):
        if noise_type in ['process', 'sensor', 'combined']:
            for i in range(0, self.simulation_parameters['T_sim'], disturbance['step']):
                if noise_type in ['process', 'combined']:
                    self.noise['noise_sim'][i][np.random.choice(self.dynamics['number_of_nodes'], disturbance['number'], replace=False)] = disturbance['magnitude'] * np.random.choice([-1, 1], disturbance['number'])
                if noise_type in ['sensor', 'combined']:
                    self.noise['noise_sim'][i][self.dynamics['number_of_nodes']:] = disturbance['magnitude'] * np.random.choice([-1, 1], self.dynamics['number_of_nodes'])
            self.noise['mod'] = noise_type

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
        self.dynamics['A_enhanced'] = {}
        self.dynamics['enhanced_stage_cost'] = {}
        active = [self.dynamics['number_of_nodes'] + i for i in self.architecture['C']['active']] + [i for i in range(0, self.dynamics['number_of_nodes'])]
        self.noise['enhanced_noise_covariance'] = self.noise['block_noise_covariance'][active, :][:, active]
        self.dynamics['enhanced_terminal_cost'] = scp.linalg.block_diag(self.architecture['B']['cost']['Q'], np.zeros((self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes'])))
        self.noise['enhanced_noise_matrix'] = {}
        self.noise['enhanced_noise_expectation'] = {}

        T = self.simulation_parameters['T_predict']
        for t in range(0, T+1):
            BK = self.architecture['B']['matrix'] @ self.architecture['B']['gain'][t]
            LC = self.architecture['C']['gain'][t] @ self.architecture['C']['matrix']
            ALC = LC @ self.dynamics['A']
            self.dynamics['A_enhanced'][t] = np.block([
                [self.dynamics['A'], -BK],
                [ALC, self.dynamics['A'] - BK - ALC]])

            self.noise['enhanced_noise_matrix'][t] = np.block([
                [np.identity(self.dynamics['number_of_nodes']), np.zeros((self.dynamics['number_of_nodes'], len(self.architecture['C']['active'])))],
                [np.zeros((self.dynamics['number_of_nodes'], self.dynamics['number_of_nodes'])), self.dynamics['A'] @ self.architecture['C']['gain'][t]]])

            self.noise['enhanced_noise_expectation'][t] = self.noise['enhanced_noise_matrix'][t] @ self.noise['enhanced_noise_covariance'] @ self.noise['enhanced_noise_matrix'][t].T

            self.dynamics['enhanced_stage_cost'][t] = scp.linalg.block_diag(self.architecture['B']['cost']['Q'], self.architecture['B']['gain'][t].T @ self.architecture['B']['cost']['R1_active'] @ self.architecture['B']['gain'][t])

    def feedback_computations(self):
        T = self.simulation_parameters['T_predict']
        self.trajectory['P'] = {T+1: self.architecture['B']['cost']['Q']}
        self.trajectory['E'] = {0: self.trajectory['E_hist'][-1]}
        self.architecture['B']['gain'] = {}
        self.architecture['C']['gain'] = {}
        for t in range(T, -1, -1):
            self.architecture['B']['gain'][t] = np.linalg.inv((self.architecture['B']['matrix'].T @ self.trajectory['P'][t+1] @ self.architecture['B']['matrix']) + self.architecture['B']['cost']['R1_active']) @ self.architecture['B']['matrix'].T @ self.trajectory['P'][t+1] @ self.dynamics['A']

            self.trajectory['P'][t] = (self.dynamics['A'].T @ self.trajectory['P'][t+1] @ self.dynamics['A']) - (self.dynamics['A'] @ self.trajectory['P'][t+1] @ self.architecture['B']['matrix'] @ self.architecture['B']['gain'][t]) + self.architecture['B']['cost']['Q']

            if np.min(np.linalg.eigvals(self.trajectory['P'][t])) < 0:
                raise Exception('Negative control cost eigenvalues')

        for t in range(0, T+1):
            self.architecture['C']['gain'][t] = self.trajectory['E'][t] @ self.architecture['C']['matrix'].T @ np.linalg.inv((self.architecture['C']['matrix'] @ self.trajectory['E'][t] @ self.architecture['C']['matrix'].T) + self.architecture['C']['cost']['R1_active'])

            self.trajectory['E'][t+1] = (self.dynamics['A'] @ self.trajectory['E'][t] @ self.dynamics['A'].T) - (self.dynamics['A'] @ self.architecture['C']['gain'][t] @ self.architecture['C']['matrix'] @ self.trajectory['E'][t] @ self.dynamics['A'].T) + self.architecture['C']['cost']['Q']

            if np.min(np.linalg.eigvals(self.trajectory['E'][t+1])) < 0:
                for k in self.trajectory['E']:
                    print(k, '- Eigenvalues: ', np.min(np.linalg.eigvals(self.trajectory['E'][k])))
                raise Exception('Negative covariance eigenvalues')

    def enhanced_lyapunov_control_cost(self):
        T = self.simulation_parameters['T_predict']
        self.trajectory['Z'] = {T+1: self.dynamics['enhanced_terminal_cost']}
        self.trajectory['cost']['control'] = 0  # np.trace(self.trajectory['Z'][T] @ self.noise['enhanced_noise_expectation'][T-1])
        for t in range(T, -1, -1):
            # print(t+1)
            self.trajectory['cost']['control'] += np.trace(self.trajectory['Z'][t+1] @ self.noise['enhanced_noise_expectation'][t])
            self.trajectory['Z'][t] = self.dynamics['A_enhanced'][t].T @ self.trajectory['Z'][t+1] @ self.dynamics['A_enhanced'][t] + self.dynamics['enhanced_stage_cost'][t]
        estimate_vector = np.concatenate((self.trajectory['x_estimate'][-1], self.trajectory['x_estimate'][-1]))
        self.trajectory['cost']['control'] += np.squeeze(estimate_vector.T @ self.trajectory['Z'][0] @ estimate_vector)

    def enhanced_stage_control_cost(self):
        self.trajectory['cost']['stage'] = np.squeeze(self.trajectory['X_enhanced'][-1].T @ self.dynamics['enhanced_stage_cost'][0] @ self.trajectory['X_enhanced'][-1])

    def enhanced_design_cost(self):
        self.feedback_computations()
        self.architecture_costs()
        self.enhanced_system_matrix()
        self.trajectory['cost']['initial'] = np.max(np.linalg.eigvals(self.trajectory['Z'][0]))

    def system_one_step_update_enhanced(self, t):
        active = [self.dynamics['number_of_nodes']+i for i in self.architecture['C']['active']] + [i for i in range(0, self.dynamics['number_of_nodes'])]
        noise = self.noise['noise_sim'][t][active]
        self.trajectory['X_enhanced'].append((self.dynamics['A_enhanced'][0] @ self.trajectory['X_enhanced'][-1]) + (self.noise['enhanced_noise_matrix'][0] @ noise))
        self.trajectory['x'].append(self.trajectory['X_enhanced'][-1][0:self.dynamics['number_of_nodes']])
        self.trajectory['x_estimate'].append(self.trajectory['X_enhanced'][-1][self.dynamics['number_of_nodes']:])
        self.trajectory['error'].append(self.trajectory['x'][-1] - self.trajectory['x_estimate'][-1])
        self.trajectory['error_2norm'].append(np.linalg.norm(self.trajectory['error'][-1], ord=2))

    def architecture_cost_update(self, R_set):
        for arch in ['B', 'C']:
            for R in R_set:
                self.architecture[arch]['cost'][R] = R_set[R]

    def architecture_costs(self):
        self.trajectory['cost']['running'] = 0
        self.trajectory['cost']['switching'] = 0
        architecture_type = ['B', 'C']
        for a in architecture_type:
            history_vector = np.zeros(self.dynamics['number_of_nodes'])
            if len(self.architecture[a]['history']) > 0:
                for i in self.architecture[a]['history'][-1]:
                    history_vector[i] = 1
            if np.asarray((self.architecture[a]['cost']['R2'] > 0)).any():
                if self.metric_model['type2'] == 'matrix':
                    self.trajectory['cost']['running'] += self.simulation_parameters['T_predict']*np.squeeze(self.architecture[a]['indicator'].T @ self.architecture[a]['cost']['R2'] @ self.architecture[a]['indicator'])
                elif self.metric_model['type2'] == 'scalar':
                    self.trajectory['cost']['running'] += self.simulation_parameters['T_predict']*self.architecture[a]['cost']['R2'] * np.linalg.norm(self.architecture[a]['indicator'], ord=1)
                else:
                    raise Exception('Check Metric for Type 2 - Running Costs')

            if np.asarray((self.architecture[a]['cost']['R3'] > 0)).any():
                if self.metric_model['type3'] == 'matrix':
                    self.trajectory['cost']['switching'] += self.simulation_parameters['T_predict']*np.squeeze((self.architecture[a]['indicator'] - history_vector).T @ self.architecture[a]['cost']['R3'] @ (self.architecture[a]['indicator'] - history_vector))
                elif self.metric_model['type3'] == 'scalar':
                    self.trajectory['cost']['switching'] += self.simulation_parameters['T_predict']*self.architecture[a]['cost']['R3'] * np.linalg.norm(self.architecture[a]['indicator'] - history_vector, ord=1)
                else:
                    raise Exception('Check Metric for Type 2 - Running Costs')

    def cost_wrapper_enhanced_prediction(self):
        self.feedback_computations()
        self.architecture_costs()
        self.enhanced_system_matrix()
        self.enhanced_lyapunov_control_cost()
        self.trajectory['cost']['predicted'].append(0)
        for i in ['running', 'switching', 'control']:
            if self.trajectory['cost'][i] < 0:
                raise Exception('Negative cost: ', i)
            self.trajectory['cost']['predicted'][-1] += self.trajectory['cost'][i]

    def cost_wrapper_enhanced_true(self):
        self.feedback_computations()
        self.architecture_costs()
        self.enhanced_system_matrix()
        self.enhanced_stage_control_cost()
        self.trajectory['cost']['true'].append(0)
        for i in ['running', 'switching', 'stage']:
            if self.trajectory['cost'][i] < 0:
                raise Exception('Negative cost: ', i)
            self.trajectory['cost']['true'][-1] += self.trajectory['cost'][i]
        self.trajectory['P_hist'].append(self.trajectory['P'][0])
        self.trajectory['E_hist'].append(self.trajectory['E'][1])

    def gramian_wrapper(self):
        architecture_set = ['B', 'C']
        for a in architecture_set:
            self.gramian_calc(arch_type=a)

    def gramian_calc(self, arch_type):
        if arch_type not in ['B', 'C']:
            raise Exception('Check architecture type')

        g_mat = np.zeros_like(self.dynamics['A'])
        for i in range(0, self.dynamics['number_of_nodes']):
            A_pow = np.linalg.matrix_power(self.dynamics['A'], i)
            AT_pow = np.linalg.matrix_power(self.dynamics['A'].T, i)
            if arch_type == 'B':
                g_mat += (A_pow @ self.architecture['B']['matrix'] @ self.architecture['B']['matrix'].T @ AT_pow)
            elif arch_type == 'C':
                g_mat += (AT_pow @ self.architecture['C']['matrix'].T @ self.architecture['C']['matrix'] @ A_pow)
        if np.linalg.det(g_mat) == 0:
            raise Exception('Gramian is singular')
        if arch_type == 'B':
            self.architecture['B']['gram'] = dc(g_mat)
        elif arch_type == 'C':
            self.architecture['C']['gram'] = dc(g_mat)

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
        print("Sys:", self.model_name)
        for a in ['B', 'C']:
            print(a, ':', self.architecture[a]['active'])

    def count_architecture_changes(self, print_check=False):
        for a in ['B', 'C']:
            self.architecture[a]['change_count'] = 0
            for t in range(0, len(self.architecture[a]['history'])-1):
                self.architecture[a]['change_count'] += compare_lists(self.architecture[a]['history'][t], self.architecture[a]['history'][t+1])['change_count']
            if print_check:
                print(a, ': ', self.architecture[a]['change_count'])

    def network_matrix(self):
        A_mat = self.dynamics['A'] > 0
        B_mat = self.architecture['B']['matrix']
        C_mat = self.architecture['C']['matrix']
        net_matrix = np.block([[A_mat, B_mat, C_mat.T],
                               [B_mat.T, np.zeros((len(self.architecture['B']['active']), len(self.architecture['B']['active']))), np.zeros((len(self.architecture['B']['active']), len(self.architecture['C']['active'])))],
                               [C_mat, np.zeros((len(self.architecture['C']['active']), len(self.architecture['B']['active']))), np.zeros((len(self.architecture['C']['active']), len(self.architecture['C']['active'])))]])
        return net_matrix

    def architecture_at_t(self, time_step):
        self.architecture['B']['active'] = self.architecture['B']['history'][time_step]
        self.architecture['C']['active'] = self.architecture['C']['history'][time_step]
        self.architecture_active_to_matrix()

    def network_node_relabel(self, G):
        node_labels = {}
        color_map = []
        node_color = {'node': 'C9', 'actuator': 'C1', 'sensor': 'C2'}
        for i in range(0, self.dynamics['number_of_nodes']):
            node_labels[i] = str(i + 1)
            color_map.append(node_color['node'])
        for i in range(0, len(self.architecture['B']['active'])):
            node_labels[i + self.dynamics['number_of_nodes']] = "B" + str(i + 1)
            color_map.append(node_color['actuator'])
        for i in range(0, len(self.architecture['C']['active'])):
            node_labels[i + self.dynamics['number_of_nodes'] + len(self.architecture['B']['active'])] = "C" + str(i + 1)
            color_map.append(node_color['sensor'])
        netx.relabel_nodes(G, node_labels, copy=False)
        return {'G': G, 'c_map': color_map}

    def full_network_position(self):
        S_full = dc(self)
        S_full.architecture['B']['active'] = self.architecture['B']['available']
        S_full.architecture['C']['active'] = self.architecture['C']['available']
        S_full.architecture_active_to_matrix()

        G_full = netx.from_numpy_array(S_full.network_matrix())
        G_R = S_full.network_node_relabel(G_full)
        G_full = G_R['G']

        G_A = netx.from_numpy_array(S_full.dynamics['A'])
        G_A = S_full.network_node_relabel(G_A)['G']

        core_circ_pos = netx.circular_layout(G_A)
        node_pos = netx.spring_layout(G_full, pos=core_circ_pos, fixed=[str(i+1) for i in range(0, S_full.dynamics['number_of_nodes'])])
        x = [node_pos[k][0] for k in node_pos]
        y = [node_pos[k][1] for k in node_pos]

        # node_pos = {}
        # for n in network_plot_parameters:
        #     node_pos[n] = network_plot_parameters[n]

        self.network_plot_parameters['node_pos'] = core_circ_pos
        # self.network_plot_parameters['lim'] = {'x': [np.min(1.1*np.min(x), 0.9*np.min(x)), np.max(1.1*np.max(x), 0.9*np.max(x))], 'y': [np.min(1.1*np.min(y), 0.9*np.min(y)), np.max(1.1*np.max(y), 0.9*np.max(y))]}
        # self.network_plot_parameters['lim'] = {
        #     'x': [np.floor(10*np.min(x))/10, np.ceil(10*np.max(x))/10],
        #     'y': [np.floor(10*np.min(y))/10, np.ceil(10*np.max(y))/10]}
        self.network_plot_parameters['lim'] = {
            'x': [-1.5, 1.5],
            'y': [-1.5, 1.5]}
        # print(self.network_plot_parameters['lim'])

    def plot_dynamics(self, ax_in=None):
        if ax_in is None:
            fig, ax = plt.add_subplot()
        else:
            ax = ax_in

        S_0 = dc(self)
        S_0.architecture['B']['active'] = []
        S_0.architecture['C']['active'] = []
        S_0.architecture_active_to_matrix()
        G = netx.from_numpy_array(S_0.dynamics['Adj'] > 0)
        relabel = S_0.network_node_relabel(G)
        G = relabel['G']
        c_map = relabel['c_map']

        netx.draw_networkx(G, ax=ax, node_size=100, pos=S_0.network_plot_parameters['node_pos'], node_color=c_map, with_labels=True, font_size=8)

        ax.set_xlim(self.network_plot_parameters['lim']['x'])
        ax.set_ylim(self.network_plot_parameters['lim']['y'])
        ax.set_aspect('equal')

        if ax_in is None:
            f_name = image_save_folder_path + self.model_name + '_t0'
            plt.savefig(f_name)
            plt.show()

    def plot_network(self, time_step=None, ax_in=None, node_filter=None):
        if ax_in is None:
            fig, ax = plt.add_subplot()
        else:
            ax = ax_in

        S_t = dc(self)
        if time_step is not None:
            S_t.architecture_at_t(time_step)
        net_mat = S_t.network_matrix()
        G = netx.from_numpy_array(net_mat)
        Graph = S_t.network_node_relabel(G)
        G = Graph['G']
        c_map = Graph['c_map']

        # network_plot_parameters, lim = S_t.full_network_position()
        # node_pos = {}
        # for n in network_plot_parameters:
        #     node_pos[n] = network_plot_parameters[n]
        if node_filter is None:
            node_list = list(G)
        elif node_filter == 'dynamics':
            node_list = list(G)[0:self.dynamics['number_of_nodes']]
            for n in list(G):
                if n not in node_list:
                    G.remove_node(n)
            c_map = c_map[0:self.dynamics['number_of_nodes']]
        elif node_filter == 'architecture':
            node_list = list(G)[self.dynamics['number_of_nodes']:]
            for n in list(G):
                if n not in node_list and n not in G.neighbors(n):
                    G.remove_node(n)
            c_map = c_map[self.dynamics['number_of_nodes']:]
        else:
            raise Exception('Check node filter')

        # print('Node list:', node_list)
        netx.draw_networkx(G, ax=ax, node_size=100, pos=netx.spring_layout(G, pos=self.network_plot_parameters['node_pos'], fixed=[str(i+1) for i in range(0, S_t.dynamics['number_of_nodes'])]), node_color=c_map, with_labels=True, font_size=8, nodelist=node_list, edgelist=netx.edges(G, node_list))
        # netx.draw_networkx(G, ax=ax, node_color=c_map, with_labels=False)
        ax.set_xlim(self.network_plot_parameters['lim']['x'])
        ax.set_ylim(self.network_plot_parameters['lim']['y'])
        ax.set_aspect('equal')

        if ax_in is None:
            f_name = image_save_folder_path + self.model_name + '_t' + str(time_step)
            plt.savefig(f_name)
            plt.show()

    def plot_architecture_history(self, f_name=None,  ax_in=None, plt_map=None):
        if ax_in is None:
            fig = plt.figure()
            grid = fig.add_gridspec(2, 1)
            ax = {'B': fig.add_subplot(grid[0, 0])}
            ax['C'] = fig.add_subplot(grid[1, 0], sharex=ax['B'])
        else:
            ax = ax_in

        if plt_map is None:
            plt_map = {'fixed': {'c': 'C0', 'line_style': 'solid', 'alpha': 0.5, 'zorder': 1},
                       'tuning': {'c': 'C1', 'line_style': 'dashed', 'alpha': 0.5, 'zorder': 2},
                       'marker': {'B': "+", 'C': "+"}}

        architecture_map = {'B': {'active': [], 'time': []}, 'C': {'active': [], 'time': []}}
        T = len(self.architecture['B']['history'])
        for a in ['B', 'C']:
            for t in range(0, T):
                for i in self.architecture[a]['history'][t]:
                    architecture_map[a]['active'].append(i)
                    architecture_map[a]['time'].append(t)

        for a in ['B', 'C']:
            for i in self.architecture[a]['history'][0]:
                ax[a].axhline(i, marker=plt_map['marker'][a], c=plt_map['fixed']['c'], linewidth=2, alpha=0.5, zorder=plt_map['fixed']['zorder'])
            ax[a].scatter(architecture_map[a]['time'], architecture_map[a]['active'], marker=plt_map['marker'][a], s=3, c=plt_map['tuning']['c'], label=a, zorder=plt_map['fixed']['zorder'])

        ax['B'].tick_params(axis="x", labelbottom=False)
        ax['B'].set_ylabel('Nodes with \n Actuators ' + r'$(B_{S_t})$')
        ax['B'].set_title(str(self.architecture['B']['change_count']) + ' changes')
        ax['C'].set_ylabel('Nodes with \n Sensors ' + r'$(C_{S_t})$')
        ax['C'].set_title(str(self.architecture['C']['change_count']) + ' changes')

        if ax_in is None:
            ax['C'].set_xlabel('Time')
            if f_name is None:
                f_name = image_save_folder_path+self.model_name+'_architecture_history.png'
            plt.savefig(f_name)
            plt.show()

    def plot_trajectory(self, f_name=None,  ax_in=None, plt_map=None, s_type='fixed', v_norm=2):
        if ax_in is None:
            fig = plt.figure()
            grid = fig.add_gridspec(2, 1)
            ax = {'x': fig.add_subplot(grid[0, 0])}
            ax['error'] = fig.add_subplot(grid[1, 0], sharex=ax['x'])
        else:
            ax = ax_in

        if plt_map is None:
            plt_map = {'fixed': {'c': 'C0', 'line_style': 'solid', 'alpha': 0.5, 'zorder': 1},
                       'tuning': {'c': 'C1', 'line_style': 'dashed', 'alpha': 0.5, 'zorder': 2},
                       'marker': {'B': "+", 'C': "+"}}

        T = len(self.trajectory['x'])
        # ax['error'].plot(range(0, T), self.trajectory['error_2norm'], c=plt_map[s_type]['c'], linestyle=plt_map[s_type]['line_style'], label=s_type, alpha=plt_map[s_type]['alpha'])

        # for i in range(0, self.dynamics['number_of_nodes']):
        #     x = [self.trajectory['error'][t][i] for t in range(0, T)]
        #     ax['error'].plot(range(0, T), x, linewidth=2, alpha=0.2, c=plt_map[s_type]['c'], linestyle=plt_map[s_type]['line_style'])

        error_vec = [np.linalg.norm(self.trajectory['error'][t], ord=v_norm) for t in range(0, T)]
        ax['error'].plot(range(0, T), error_vec, linewidth=2, alpha=0.8, c=plt_map[s_type]['c'], linestyle=plt_map[s_type]['line_style'])

        # for i in range(0, self.dynamics['number_of_nodes']):
        #     x = [self.trajectory['x'][t][i] for t in range(0, T)]
        #     ax['x'].plot(range(0, T), x, linewidth=2, alpha=0.2, c=plt_map[s_type]['c'], linestyle=plt_map[s_type]['line_style'])

        state_vec = [np.linalg.norm(self.trajectory['x'][t], ord=v_norm) for t in range(0, T)]
        ax['x'].plot(range(0, T), state_vec, linewidth=2, alpha=0.8, c=plt_map[s_type]['c'], linestyle=plt_map[s_type]['line_style'])

        ax['x'].set_ylabel(r'$|x_t|_'+str(v_norm)+'$')
        ax['error'].set_ylabel(r'$|x_t - \hat{x}_t|_'+str(v_norm)+'$')
        ax['x'].set_yscale('log')
        ax['error'].set_yscale('log')
        ax['x'].tick_params(axis="x", labelbottom=False)

        if ax_in is None:
            ax['error'].set_xlabel('Time')
            if f_name is None:
                f_name = image_save_folder_path + self.model_name + '_trajectory.png'
            plt.savefig(f_name)
            plt.show()
        else:
            # ax['x'].tick_params(axis="x", labelbottom=False)
            ax['error'].tick_params(axis="x", labelbottom=False)

    def plot_eigvals(self, f_name=None,  ax_in=None):
        if ax_in is None:
            fig = plt.figure()
            grid = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(grid[0, 0])
        else:
            ax = ax_in

        ax.scatter(range(1, self.dynamics['number_of_nodes']+1), -np.sort(-1*np.abs(self.dynamics['ol_eig'])), s=10, marker='x')
        ax.axhline(1, ls='--', c='k')
        ax.set_xlabel(r'Mode $i$')
        ax.set_ylabel(r'$\lambda_i (A)$')

        if ax_in is None:
            if f_name is None:
                f_name = image_save_folder_path + self.model_name + '_openloopeigs.png'
            plt.savefig(f_name)
            plt.show()



def matrix_convergence_check(A, B, accuracy=10 ** (-8), check_type=None):
    np_norm_methods = ['inf', 'fro', 2, None]
    if check_type is None:
        return np.allclose(A, B, atol=accuracy, rtol=accuracy)
    elif check_type in np_norm_methods:
        return np.linalg.norm(A - B, ord=check_type) < accuracy
    else:
        raise Exception('Check Matrix Convergence')


def max_limit(architecture, algorithm):
    correction = {'select': 0, 'reject': 1}
    return len(architecture['active']) < (architecture['max'] + correction[algorithm])


def min_limit(architecture, algorithm):
    correction = {'select': 0, 'reject': 1}
    return len(architecture['active']) >= (architecture['min'] + correction[algorithm])


def compare_lists(array1, array2):
    a1 = [i for i in array1 if i not in array2]
    a2 = [i for i in array2 if i not in array1]
    c = [i for i in array1 if i in array2]
    changes = np.max([len(a1), len(a2)])
    return {'a1only': a1, 'a2only': a2, 'common': c, 'change_count': changes}


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


def greedy_architecture_selection(sys, number_of_changes=None, policy="min", no_select=False, status_check=False, t_start=time.time(), design_cost=False):
    if not isinstance(sys, System):
        raise Exception('Check data type')
    if status_check:
        print('\nSelection\n')
    work_iteration = dc(sys)
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
        if no_select and (min_limit(work_iteration.architecture['B'], 'select') and min_limit(work_iteration.architecture['C'], 'select')):
            choices.append({'architecture_type': None})
        choice_history.append(choices)
        for i in choices:
            i['algorithm'] = 'select'
            test_sys = dc(work_iteration)
            if status_check:
                test_sys.model_name = "test_model"
                # test_sys.display_active_architecture()
            test_sys.active_architecture_update(i)
            if status_check:
                test_sys.display_active_architecture()
            if not design_cost:
                test_sys.cost_wrapper_enhanced_prediction()
                i['value'] = test_sys.trajectory['cost']['predicted'][-1]
            else:
                test_sys.enhanced_design_cost()
                i['value'] = test_sys.trajectory['cost']['initial']
        if status_check:
            print_choices(choices)
        value_history.append([i['value'] for i in choices])
        target_idx = item_index_from_policy(value_history[-1], policy)
        work_iteration.active_architecture_update(choices[target_idx])
        if not design_cost:
            work_iteration.cost_wrapper_enhanced_prediction()
        else:
            work_iteration.enhanced_design_cost()
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
    if not design_cost:
        return_set.cost_wrapper_enhanced_prediction()
    else:
        return_set.enhanced_design_cost()
    if status_check:
        return_set.display_active_architecture()
    work_history.append(return_set)
    return {'work_set': return_set, 'work_history': work_history, 'choice_history': choice_history, 'value_history': value_history, 'time': time.time() - t_start, 'steps': count_of_changes}


def greedy_architecture_rejection(sys, number_of_changes=None, policy="min", no_reject=False, status_check=False, t_start=time.time(), design_cost=False):
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
            print('\n Change:', count_of_changes)
            work_iteration.display_active_architecture()
        work_history.append(work_iteration)
        choices = work_iteration.available_choices('reject')
        if len(choices) == 0:
            if status_check:
                print('No selections available')
            rejection_check = False
            break
        if no_reject and (max_limit(work_iteration.architecture['B'], 'select') and max_limit(work_iteration.architecture['C'], 'reject')):
            choices.append({'architecture_type': None})
        choice_history.append(choices)
        for i in choices:
            i['algorithm'] = 'reject'
            test_sys = dc(work_iteration)
            test_sys.model_name = "test_model"
            test_sys.active_architecture_update(i)
            if status_check:
                print('Choice i:', i)
                test_sys.display_active_architecture()
            if not design_cost:
                test_sys.cost_wrapper_enhanced_prediction()
                i['value'] = test_sys.trajectory['cost']['predicted'][-1]
            else:
                test_sys.enhanced_design_cost()
                i['value'] = test_sys.trajectory['cost']['initial']
        if status_check:
            print_choices(choices)
        target_idx = item_index_from_policy([i['value'] for i in choices], policy)
        value_history.append([i['value'] for i in choices])
        work_iteration.active_architecture_update(choices[target_idx])
        if not design_cost:
            work_iteration.cost_wrapper_enhanced_prediction()
        else:
            work_iteration.enhanced_design_cost()
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
    if not design_cost:
        return_set.cost_wrapper_enhanced_prediction()
    else:
        return_set.enhanced_design_cost()
    if status_check:
        return_set.display_active_architecture()
    work_history.append(return_set)
    return {'work_set': return_set, 'work_history': work_history, 'choice_history': choice_history, 'value_history': value_history, 'time': time.time() - t_start, 'steps': count_of_changes}


def greedy_simultaneous(sys, iterations=None, changes_per_iteration=1, fixed_set=None, failure_set=None, policy="min", t_start=time.time(), status_check=False, design_cost=False):
    if not isinstance(sys, System):
        raise Exception('Incorrect data type')
    if status_check:
        print('\nSimultaneous\n')
    work_iteration = dc(sys)
    work_history = [work_iteration]
    value_history = []
    iteration_stop = False
    iteration_count = 0
    while not iteration_stop:
        if status_check:
            print('Iteration:', iteration_count)
            work_iteration.display_active_architecture()
        # Keep same
        values = []
        iteration_cases = []

        # Select one
        select = greedy_architecture_selection(work_iteration, number_of_changes=changes_per_iteration, policy=policy, t_start=t_start, no_select=True, status_check=status_check, design_cost=design_cost)
        iteration_cases.append(select['work_set'])
        if not design_cost:
            values.append(iteration_cases[-1].trajectory['cost']['predicted'][-1])
        else:
            values.append(iteration_cases[-1].trajectory['cost']['initial'])
        if status_check:
            print('Add Value: ', values[-1])
            iteration_cases[-1].display_active_architecture()

        # Reject one
        reject = greedy_architecture_rejection(work_iteration, number_of_changes=changes_per_iteration, policy=policy, t_start=t_start, no_reject=True, status_check=status_check, design_cost=design_cost)
        iteration_cases.append(reject['work_set'])
        if not design_cost:
            values.append(iteration_cases[-1].trajectory['cost']['predicted'][-1])
        else:
            values.append(iteration_cases[-1].trajectory['cost']['initial'])
        if status_check:
            print('Subtract Value: ', values[-1])
            iteration_cases[-1].display_active_architecture()

        # Swap: add then drop
        if status_check:
            print('\nSwap\n')
        sys_swap = dc(work_iteration)
        sys_swap.architecture_limit_modifier(min_mod=changes_per_iteration, max_mod=changes_per_iteration)
        sys_swap = greedy_architecture_selection(sys_swap, number_of_changes=changes_per_iteration, policy=policy, t_start=t_start, no_select=False, status_check=status_check, design_cost=design_cost)['work_set']
        sys_swap.cost_wrapper_enhanced_prediction()
        # print('1:', sys_swap.trajectory['cost']['predicted'][-1])
        sys_swap.architecture_limit_modifier(min_mod=-changes_per_iteration, max_mod=-changes_per_iteration)
        sys_swap = greedy_architecture_rejection(sys_swap, number_of_changes=changes_per_iteration, policy=policy, t_start=t_start, status_check=status_check, design_cost=design_cost)['work_set']
        sys_swap.cost_wrapper_enhanced_prediction()
        # print('2:', sys_swap.trajectory['cost']['predicted'][-1])
        iteration_cases.append(sys_swap)
        if not design_cost:
            values.append(iteration_cases[-1].trajectory['cost']['predicted'][-1])
        else:
            values.append(iteration_cases[-1].trajectory['cost']['initial'])
        if status_check:
            print('Swap Value: ', values[-1])
            iteration_cases[-1].display_active_architecture()

        target_idx = item_index_from_policy(values, policy)
        work_iteration.active_architecture_duplicate(iteration_cases[target_idx])
        if not design_cost:
            work_iteration.cost_wrapper_enhanced_prediction()
        else:
            work_iteration.enhanced_design_cost()
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
    if not design_cost:
        return_set.cost_wrapper_enhanced_prediction()
    else:
        return_set.enhanced_design_cost()
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


def cost_plots(cost, f_name=None, ax=None, plt_map=None):
    if ax is None:
        fig = plt.figure(figsize=(6, 4), )
        grid = fig.add_gridspec(1, 1)
        ax_cost = fig.add_subplot(grid[0, 0])
    else:
        ax_cost = ax
    
    if plt_map is None:
        plt_map = {'fixed': {'c': 'C0', 'line_style': 'solid', 'alpha': 0.5, 'zorder': 1},
                   'tuning': {'c': 'C1', 'line_style': 'dashed', 'alpha': 0.5, 'zorder': 2},
                   'marker': {'B': "+", 'C': "+"}}

    # font = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 16}

    tstep_cost = {}
    T = len(cost['fixed'])
    for i in cost:
        cumulative_cost = [cost[i][0]]
        for t in range(1, T):
            cumulative_cost.append(cumulative_cost[-1] + cost[i][t])
        tstep_cost[i] = cumulative_cost[-1]
        ax_cost.plot(range(0, T), cumulative_cost, label=i+' cumulative', c=plt_map[i]['c'], linestyle='solid', zorder=plt_map[i]['zorder'])
        ax_cost.plot(range(0, T), cost[i], label=i+' stage', c=plt_map[i]['c'], linestyle='dashed', zorder=plt_map[i]['zorder'])
    ax_cost.set_ylabel('Cost')
    ax_cost.legend(loc='upper left', ncols=2)
    improvement = np.round(100*(tstep_cost['fixed']-tstep_cost['tuning'])/tstep_cost['fixed'], 2)
    improvement_str = 'Cost: '
    if improvement < 100:
        improvement_str = improvement_str + str(improvement) + r'\% improvement'
    improvement_str = improvement_str + ' : Fixed ' + np.format_float_scientific(tstep_cost['fixed'], precision=2, trim='0') + ' vs Self-Tuning ' + np.format_float_scientific(tstep_cost['tuning'], precision=2, trim='0')
    print(improvement_str)
    # if improvement > 95:
    #     ax_cost.set_yscale('log')
    # else:
    ax_cost.set_yscale('log')
    # ax_cost.ticklabel_format(axis='y', style='sci', scilimits=(-4, 4))
    ax_cost.set_xlim(-1, 1+max([len(cost[i]) for i in cost]))
    ax_cost.set_title(improvement_str)
    # x_lims = ax_cost.get_xlim()
    # y_lims = ax_cost.get_ylim()
    # ax_cost.text(x_lims[0]+(0.5*(x_lims[1]-x_lims[0])), y_lims[0]+(0.0001*(y_lims[1]-y_lims[0])), improvement_str)
    if ax is None:
        # ax_cost.set_title('Cost comparison')
        ax_cost.set_xlabel('Time')
        if f_name is None:
            f_name = image_save_folder_path + "cost_trajectory.png"
        else:
            f_name = image_save_folder_path + f_name + "_cost_trajectory.png"
        plt.savefig(f_name)
        plt.show()
    else:
        ax_cost.tick_params(axis="x", labelbottom=False)


def combined_plot(S, S_fixed, S_tuning):
    print('\n Plotting')

    fig = plt.figure(figsize=(5, 7), tight_layout=True)
    grid = fig.add_gridspec(5, 1)
    plt_map = {'fixed': {'c': 'C0', 'line_style': 'solid', 'alpha': 0.5, 'zorder': 1},
               'tuning': {'c': 'C1', 'line_style': 'dashed', 'alpha': 0.5, 'zorder': 2},
               'marker': {'B': "o", 'C': "o"}}

    ax_cost = fig.add_subplot(grid[0, 0])
    ax_trajectory = fig.add_subplot(grid[1, 0], sharex=ax_cost)
    ax_error = fig.add_subplot(grid[2, 0], sharex=ax_cost)
    ax_architecture_B = fig.add_subplot(grid[3, 0], sharex=ax_cost)
    ax_architecture_C = fig.add_subplot(grid[4, 0], sharex=ax_cost)

    cost_plots({'fixed': S_fixed.trajectory['cost']['true'], 'tuning': S_tuning.trajectory['cost']['true']}, S.model_name, ax_cost, plt_map=plt_map)
    S_fixed.plot_trajectory(ax_in={'x': ax_trajectory, 'error': ax_error}, plt_map=plt_map, s_type='fixed')
    S_tuning.plot_trajectory(ax_in={'x': ax_trajectory, 'error': ax_error}, plt_map=plt_map, s_type='tuning')
    S_tuning.plot_architecture_history(ax_in={'B': ax_architecture_B, 'C': ax_architecture_C}, plt_map=plt_map)

    # ax_cost.tick_params(axis="x", labelbottom=False)
    # ax_error.tick_params(axis="x", labelbottom=False)
    # ax_trajectory.tick_params(axis="x", labelbottom=False)

    ax_cost.set_title(S.model_name)
    ax_architecture_C.set_xlabel('Time')

    save_file = image_save_folder_path + S.model_name+'_fixed_vs_tuning.png'
    plt.savefig(save_file)
    print('Figure saved:', save_file)
    plt.show()


def simulate_fixed_architecture(S, print_check=True, multiprocess_check=False):
    if not isinstance(S, System):
        raise Exception('Check data type')
    T_sim = dc(S.simulation_parameters['T_sim']) + 1
    if print_check:
        print('\n     Fixed architecture simulation')
    S_fixed = dc(S)
    S_fixed.model_rename(S.model_name + "_fixed")
    if multiprocess_check:
        process_number = int(multiprocessing.current_process().name[-1])
        for t in tqdm(range(0, T_sim), ncols=100, position=process_number, leave=False, desc='Process '+str(process_number)):
            S_fixed.cost_wrapper_enhanced_true()
            S_fixed.system_one_step_update_enhanced(t)
    else:
        for t in tqdm(range(0, T_sim), ncols=100, leave=False):
            S_fixed.cost_wrapper_enhanced_true()
            S_fixed.system_one_step_update_enhanced(t)
        print('         Simulation Done')
    return S_fixed


def simulate_selftuning_architecture(S, iterations_per_step=1, changes_per_iteration=1, print_check=True, multiprocess_check=False):
    if print_check:
        print('\n     Self-Tuning architecture simulation')
    S_tuning = dc(S)
    S_tuning.model_rename(S.model_name + "_selftuning")
    T_sim = dc(S.simulation_parameters['T_sim']) + 1
    if multiprocess_check:
        process_number = int(multiprocessing.current_process().name[-1])
        for t in tqdm(range(0, T_sim), ncols=100, position=process_number, leave=False, desc='Process '+str(process_number)):
            S_tuning.cost_wrapper_enhanced_true()
            S_tuning.system_one_step_update_enhanced(t)
            S_tuning = dc(greedy_simultaneous(S_tuning, iterations=iterations_per_step, changes_per_iteration=changes_per_iteration)['work_set'])
    else:
        for t in tqdm(range(0, T_sim), ncols=100, leave=False):
            S_tuning.cost_wrapper_enhanced_true()
            S_tuning.system_one_step_update_enhanced(t)
            S_tuning = dc(greedy_simultaneous(S_tuning, iterations=iterations_per_step, changes_per_iteration=changes_per_iteration)['work_set'])
        print('         Simulation Done')
    S_tuning.count_architecture_changes()
    return S_tuning


def greedy_architecture_initialization(S):
    if not isinstance(S, System):
        raise Exception('Data type check')
    S_test = dc(S)
    for a in ['B', 'C']:
        # S_test.architecture[a]['active'] = [S_test.architecture[a]['active'][0]]
        S_test.architecture[a]['cost']['R3'] = 0
    # S_test.simulation_parameters['T_predict'] *= 2
    S_test.model_name = 'Test model'
    S_test.display_active_architecture()
    # S_test = dc(greedy_architecture_selection(S_test)['work_set'])
    S_test = dc(greedy_simultaneous(S_test, iterations=S_test.dynamics['number_of_nodes'])['work_set'])
    print('Optimal architecture:')
    S_test.display_active_architecture()
    S_test.model_name = S.model_name
    for a in ['B', 'C']:
        S.architecture[a]['active'] = dc(S_test.architecture[a]['active'])
    return S_test


def statistics_shelving_initialize(fname):
    folder_path = datadump_folder_path + 'statistics/' + fname
    if os.path.isdir(folder_path):
        rmtree(folder_path)
    os.makedirs(folder_path)


def data_shelving_statistics(S, S_fixed, S_tuning, i, print_check=False):
    shelve_filename = datadump_folder_path + 'statistics/' + S.model_name + '/model_' + str(i)
    shelve_data = shelve.open(shelve_filename)
    if print_check:
        print('\nShelving to:', shelve_filename)
    shelve_data['System'] = S
    shelve_data['Fixed'] = S_fixed
    shelve_data['SelfTuning'] = S_tuning
    shelve_data.close()
    if print_check:
        print('\nModel shelve done: ', shelve_filename)


def data_reading_statistics(model_type, model_id, print_check=False):
    shelve_filename = datadump_folder_path + 'statistics/' + model_type + '/model_' + str(model_id)
    if print_check:
        print('\nShelving from:', shelve_filename)
    shelve_data = shelve.open(shelve_filename)
    S = shelve_data['System']
    S_fixed = shelve_data['Fixed']
    S_tuning = shelve_data['SelfTuning']
    shelve_data.close()
    return S, S_fixed, S_tuning


def data_shelving_gen_model(S):
    shelve_filename = datadump_folder_path + 'gen_' + S.model_name
    del_check = False
    for f_type in ['.bak', '.dat', '.dir']:
        if os.path.exists(shelve_filename + f_type):
            os.remove(shelve_filename + f_type)
            del_check = True
    if del_check:
        print('\nOld data deleted for overwriting')
    shelve_data = shelve.open(shelve_filename)
    shelve_data['System'] = S
    shelve_data.close()
    print('\nModel shelve done: ', shelve_filename)


def model_namer(n, rho, Tp, n_arch, test_model=None, second_order=False):
    model = 'model_n' + str(n)
    if rho is not None:
        model = model + '_rho' + str(rho)
    else:
        model = model + '_rhoNone'
    model = model + '_Tp' + str(Tp) + '_arch' + str(n_arch)
    if test_model is not None:
        model = model + '_' + test_model
    if second_order:
        model = model + '_secondorder'
    return model


def data_reading_gen_model(model):
    print('\nReading generated model...')
    modelname = 'gen_' + model
    print(modelname)
    shelve_file = datadump_folder_path + modelname
    shelve_data = shelve.open(shelve_file)
    S = shelve_data['System']
    shelve_data.close()
    if not isinstance(S, System):
        raise Exception('System model error')
    return S


def data_shelving_sim_model(S, S_fixed, S_tuning):
    print('\nTrajectory data shelving')
    shelve_file = datadump_folder_path + 'sim_' + S.model_name
    print('\nDeleting old data...')
    for f_type in ['.bak', '.dat', '.dir']:
        if os.path.exists(shelve_file + f_type):
            os.remove(shelve_file + f_type)
    print('\nShelving new data...')
    shelve_data = shelve.open(shelve_file)
    for k in ['System', 'Fixed', 'SelfTuning']:
        if k in shelve_data:
            del shelve_data[k]
    shelve_data['System'] = S
    shelve_data['Fixed'] = S_fixed
    shelve_data['SelfTuning'] = S_tuning
    shelve_data.close()
    print('\nShelving done:', shelve_file)


def data_reading_sim_model(model):
    print('\nData reading')
    shelve_file = datadump_folder_path + 'sim_' + model
    print(shelve_file)
    try:
        shelve_data = shelve.open(shelve_file)
    except (FileNotFoundError, IOError):
        raise Exception('test file not found')
    for k in ['System', 'Fixed', 'SelfTuning']:
        if k not in shelve_data:
            raise Exception('Check data file')
    S = shelve_data['System']
    S_fixed = shelve_data['Fixed']
    S_tuning = shelve_data['SelfTuning']
    shelve_data.close()
    if not isinstance(S, System) or not isinstance(S_tuning, System) or not isinstance(S_fixed, System):
        raise Exception('Data type mismatch')
    print('\nModel: ', S.model_name)
    return S, S_fixed, S_tuning


def time_axis_plot(model):
    S, S_fixed, S_tuning = data_reading_sim_model(model)
    plt_map = {'fixed': {'c': 'C0', 'line_style': 'solid', 'alpha': 0.5, 'zorder': 1},
               'tuning': {'c': 'C1', 'line_style': 'dashed', 'alpha': 0.5, 'zorder': 2},
               'marker': {'B': "o", 'C': "o"}}

    t_step = S.simulation_parameters['T_sim'] + 1

    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    grid = fig.add_gridspec(6, 2)

    ax_cost = fig.add_subplot(grid[0, 0])
    ax_trajectory = fig.add_subplot(grid[1, 0], sharex=ax_cost)
    ax_error = fig.add_subplot(grid[2, 0], sharex=ax_cost)
    ax_architecture_B = fig.add_subplot(grid[3, 0], sharex=ax_cost)
    ax_architecture_C = fig.add_subplot(grid[4, 0], sharex=ax_cost)
    ax_eigvals = fig.add_subplot(grid[5, 0])

    ax_tstep_architecture = fig.add_subplot(grid[0:4, 1], frameon=False, zorder=-1)
    ax_tstep_architecture.tick_params(axis='both', labelbottom=False, labelleft=False, bottom=False, top=False, left=False, right=False)
    # ax_tstep_architecture.patch.set_alpha(0.1)

    ax_network_nodes = fig.add_subplot(grid[0:4, 1], zorder=0)
    ax_network_nodes.tick_params(axis='both', labelbottom=False, labelleft=False, bottom=False, top=False, left=False, right=False)
    ax_network_nodes.patch.set_alpha(0.1)

    ax_timeline = fig.add_subplot(grid[0:5, 0], sharex=ax_cost, frameon=False)
    ax_timeline.tick_params(axis='both', labelbottom=False, labelleft=False, bottom=False, top=False, left=False, right=False)

    cost_plots({'fixed': S_fixed.trajectory['cost']['true'], 'tuning': S_tuning.trajectory['cost']['true']}, S.model_name, ax_cost, plt_map=plt_map)
    S_fixed.plot_trajectory(ax_in={'x': ax_trajectory, 'error': ax_error}, plt_map=plt_map, s_type='fixed')
    S_tuning.plot_trajectory(ax_in={'x': ax_trajectory, 'error': ax_error}, plt_map=plt_map, s_type='tuning')
    S_tuning.plot_architecture_history(ax_in={'B': ax_architecture_B, 'C': ax_architecture_C}, plt_map=plt_map)
    S_fixed.plot_eigvals(ax_in=ax_eigvals)

    S_tuning.plot_network(ax_in=ax_network_nodes, node_filter='dynamics')
    ax_network_nodes.set_title(S.model_name + '\nUnstable modes:' + str(S_tuning.dynamics['n_unstable']))
    # S_tuning.plot_dynamics(ax_in=ax_network_nodes)
    S_tuning.plot_network(ax_in=ax_tstep_architecture, node_filter='architecture')

    time_slider_dim = [0.3, 0.025]
    reset_button_dim = [0.1, 0.05]
    next_button_dim = [0.1, 0.05]
    prev_button_dim = [0.1, 0.05]
    # text_box_dim = [0.1, 0.05]

    ax_timeslide = fig.add_axes([((0.95 + 0.55 - time_slider_dim[0]) / 2), 0.1, time_slider_dim[0], time_slider_dim[1]])
    timeslider = Slider(ax=ax_timeslide, label='', valmin=0, valmax=S.simulation_parameters['T_sim'] + 1, valinit=S.simulation_parameters['T_sim'] + 1, valstep=1)
    t_text = ax_timeslide.text(0.38, 2.6, 'Time: '+str(t_step), transform=ax_timeslide.transAxes, fontsize='large')
    # valfmt=' time: %d'
    timeslider.valtext.set_visible(False)

    ax_reset_button = fig.add_axes([((0.95 + 0.55 - reset_button_dim[0]) / 2), 0.05, reset_button_dim[0], reset_button_dim[1]])
    reset_button = Button(ax=ax_reset_button, label='Reset', hovercolor='0.975')

    ax_prev_button = fig.add_axes([((0.95 + 0.55 - 3*prev_button_dim[0]) / 2), 0.15, prev_button_dim[0], prev_button_dim[1]])
    prev_button = Button(ax=ax_prev_button, label='t-', hovercolor='0.975')

    ax_next_button = fig.add_axes([((0.95 + 0.55 + next_button_dim[0]) / 2), 0.15, next_button_dim[0], next_button_dim[1]])
    next_button = Button(ax=ax_next_button, label='t+', hovercolor='0.975')

    # ax_text_box = fig.add_axes([((0.95 + 0.55 - text_box_dim[0]) / 2), 0.15, text_box_dim[0], text_box_dim[1]])
    # time_text = TextBox(ax=ax_text_box, label='', textalignment='center', initial='t:'+str(S.simulation_parameters['T_sim'] + 1))

    def slider_update(t):
        nonlocal t_step
        t_step = t
        # time_text.set_val(t_step)
        ax_timeslide.text(time_slider_dim[0] / 2, time_slider_dim[1] + 0.05, 'time: ' + str(t_step))
        t_text.set_text('time: ' + str(t_step))
        ax_timeline.clear()
        ax_tstep_architecture.clear()
        ax_timeline.axvline(t, alpha=0.2, c='k', linestyle='dashed')
        # ax_timeline.tick_params(axis='both', labelbottom=False, labelleft=False, bottom=False, top=False, left=False, right=False)
        S_tuning.plot_network(ax_in=ax_tstep_architecture, time_step=t, node_filter='architecture')

    def reset_button_press(event):
        nonlocal t_step
        t_step = S.simulation_parameters['T_sim'] + 1
        # time_text.set_val(t_step)
        timeslider.reset()

    def next_button_press(event):
        nonlocal t_step
        t_step += 1
        # time_text.set_val(t_step)
        timeslider.set_val(t_step)

    def prev_button_press(event):
        nonlocal t_step
        t_step -= 1
        # time_text.set_val(t_step)
        timeslider.set_val(t_step)

    # def text_box_update(event):
    #     nonlocal t_step
    #     # time_text.set_val('t='+str(t_step))

    timeslider.on_changed(slider_update)
    reset_button.on_clicked(reset_button_press)
    next_button.on_clicked(next_button_press)
    prev_button.on_clicked(prev_button_press)
    # time_text.on_submit(text_box_update)
    plt.show()


def statistics_plot(test_model):
    plt_map = {'fixed': {'c': 'C0', 'line_style': 'solid', 'alpha': 0.5, 'zorder': 1},
               'tuning': {'c': 'C1', 'line_style': 'dashed', 'alpha': 0.5, 'zorder': 2},
               'marker': {'B': "o", 'C': "o"}}
    S, _, _ = data_reading_statistics(test_model, 1)
    if not isinstance(S, System):
        raise Exception('Incorrect model')

    unstable_modes = []
    B_changes = []
    C_changes = []
    cost_min_fix = np.inf * np.ones(S.simulation_parameters['T_sim'] + 1)
    cost_min_tuning = dc(cost_min_fix)
    cost_max_fix = np.zeros(S.simulation_parameters['T_sim'] + 1)
    cost_max_tuning = dc(cost_max_fix)

    rand_model = np.random.choice(range(1, 101), 1)
    cost_fix = []
    cost_tuning = []

    for model_id in tqdm(range(1, 101), ncols=100):
        S_i, S_fixed_i, S_tuning_i = data_reading_statistics(S.model_name, model_id)

        if not isinstance(S_i, System):
            raise Exception('Incorrect system model')
        if not isinstance(S_fixed_i, System):
            raise Exception('Incorrect system model')
        if not isinstance(S_tuning_i, System):
            raise Exception('Incorrect system model')

        cumulative_cost_fixed = [S_fixed_i.trajectory['cost']['true'][0]]
        cumulative_cost_tuning = [S_tuning_i.trajectory['cost']['true'][0]]
        for j in range(1, len(S_fixed_i.trajectory['cost']['true'])):
            cumulative_cost_fixed.append(cumulative_cost_fixed[-1] + S_fixed_i.trajectory['cost']['true'][j])
            cumulative_cost_tuning.append(cumulative_cost_tuning[-1] + S_tuning_i.trajectory['cost']['true'][j])

        if model_id == rand_model:
            cost_fix = cumulative_cost_fixed
            cost_tuning = cumulative_cost_tuning

        unstable_modes.append(S_i.dynamics['n_unstable'])
        B_changes.append(S_tuning_i.architecture['B']['change_count'])
        C_changes.append(S_tuning_i.architecture['C']['change_count'])
        # print(model_id, ' : ', S_i.dynamics['n_unstable'])
        cost_min_fix = [min(cost_min_fix[i], cumulative_cost_fixed[i]) for i in range(0, len(cost_min_fix))]
        cost_min_tuning = [min(cost_min_tuning[i], cumulative_cost_tuning[i]) for i in range(0, len(cost_min_tuning))]
        cost_max_fix = [max(cost_max_fix[i], cumulative_cost_fixed[i]) for i in range(0, len(cost_max_fix))]
        cost_max_tuning = [max(cost_max_tuning[i], cumulative_cost_tuning[i]) for i in range(0, len(cost_max_tuning))]

    # print('Min fix: ', cost_min_fix)
    # print('Min tuning: ', cost_min_tuning)
    # print('Max fix: ', cost_max_fix)
    # print('Max tuning: ', cost_max_tuning)

    fig = plt.figure(constrained_layout=True)
    grid = fig.add_gridspec(4, 2)

    unstable_plot = fig.add_subplot(grid[1, :])
    cost_plot = fig.add_subplot(grid[0, :])
    B_arch_plot = fig.add_subplot(grid[2, 0])
    C_arch_plot = fig.add_subplot(grid[3, 0])
    arch_scatter_plot = fig.add_subplot(grid[2:4, 1])

    unstable_plot.hist(unstable_modes, bins=range(0, max(unstable_modes) + 1), align='right', density=True)
    cost_plot.fill_between(range(0, len(cost_min_fix)), cost_min_fix, cost_max_fix, color=plt_map['fixed']['c'], alpha=plt_map['fixed']['alpha'])
    cost_plot.fill_between(range(0, len(cost_min_tuning)), cost_min_tuning, cost_max_tuning, color=plt_map['tuning']['c'], alpha=plt_map['tuning']['alpha'])
    cost_plot.plot(range(0, len(cost_fix)), cost_fix, color=plt_map['fixed']['c'])
    cost_plot.plot(range(0, len(cost_tuning)), cost_tuning, color=plt_map['tuning']['c'])

    B_arch_plot.hist(B_changes, density=True)
    C_arch_plot.hist(C_changes, density=True)
    arch_scatter_plot.scatter(B_changes, C_changes)

    unstable_plot.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    unstable_plot.set_xlabel('Number of unstable modes')
    unstable_plot.set_ylabel('Fraction of\nrealizations')
    cost_plot.set_yscale('log')
    # cost_plot.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    cost_plot.set_xlabel('Time')
    cost_plot.set_ylabel('Cost')
    cost_plot.legend(handles=[patches.Patch(color=plt_map['fixed']['c'], label='Fixed Architecture'), patches.Patch(color=plt_map['tuning']['c'], label='SelfTuning')], loc='upper left')
    # arch_plot.legend(['B', 'C'])
    B_arch_plot.set_xlabel('Number of changes to actuators')
    B_arch_plot.set_ylabel('Fraction of\nrealizations')
    C_arch_plot.set_xlabel('Number of changes to sensors')
    C_arch_plot.set_ylabel('Fraction of\nrealizations')
    arch_scatter_plot.set_xlabel('Number of changes to actuators')
    arch_scatter_plot.set_ylabel('Number of changes to sensors')

    plt.savefig(image_save_folder_path + test_model + '_statistics.png')
    plt.show()


if __name__ == "__main__":
    print('Check function file')
