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
import matplotlib.lines as mlines
# from matplotlib.ticker import MaxNLocator
import matplotlib.animation
import multiprocessing
import pandas as pd

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


class ArchitectureError(Exception):
    """Raise when architecture is not B or C"""
    def __init__(self, value='Check architecture type'):
        self.value = value

    def __str__(self):
        return repr(self.value)


class SecondOrderError(Exception):
    """Raise when second order or type is not specified accurately"""
    def __init__(self, value='Check second order type'):
        self.value = value

    def __str__(self):
        return repr(self.value)


class ClassError(Exception):
    """Raise when variable is not of the correct class/variable type"""
    def __init__(self, value='Check class data type'):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Experiment:
    def __init__(self):
        self.save_filename = "experiment_parameters.csv"

        self.parameter_keys = ['number_of_nodes', 'network_model', 'second_order', 'second_order_network', 'initial_architecture_size', 'second_order_architecture', 'W_scaling', 'V_scaling', 'disturbance_model', 'disturbance_step', 'disturbance_number', 'disturbance_magnitude', 'simulation_model', 'architecture_constraint', 'rho', 'network_parameter', 'prediction_time_horizon', 'X0_scaling']

        self.parameter_table = None
        self.parameters = None
        self.experiment_number = None

    def read_table_from_file(self):
        if not os.path.exists(datadump_folder_path + self.save_filename):
            raise Warning('File does not exist')
        else:
            self.parameter_table = pd.read_csv(datadump_folder_path + self.save_filename)

    def read_parameters_from_table(self, experiment_no):
        if experiment_no not in self.parameter_table:
            raise Exception('Experiment parameters not in table')
        if not isinstance(self.parameter_table, pd.DataFrame):
            raise Exception('Not a pandas frame')

        self.parameters = zip(self.parameter_table.columns, [k for k in self.parameter_table.iloc(experiment_no)])

    def write_to_table(self, parameter_values=None):
        if parameter_values is None:
            raise Exception('No experiment parameters provided')
        # parameter_values = [20, 'rand', False, 0, 2, None, 1, 1, None, None, None, None, None, 3, 3, None, 10, 1]
        self.parameter_table = self.parameter_table.append(parameter_values)

    def write_to_file(self):
        self.parameter_table.to_csv(datadump_folder_path + self.save_filename)
        print('Printing done')


class System:

    class Dynamics:
        def __init__(self):
            self.number_of_nodes = 20
            self.number_of_states = 20

            self.rho = 1
            self.network_model = 'rand'
            self.network_parameter = None
            self.self_loop = True

            self.second_order = False
            self.second_order_scaling_factor = 1
            self.second_order_network_type = 1

            self.open_loop_eig_vals = np.zeros(self.number_of_nodes)
            self.adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes))
            self.number_of_non_stable_modes = 0
            self.A = np.zeros((self.number_of_nodes, self.number_of_nodes))

        def adjacency_matrix_initialize(self):
            if self.network_model not in ['rand', 'ER', 'BA', 'path', 'cycle', 'eval_squeeze', 'eval_bound']:
                raise Exception('Network model not defined')

            connected_graph_check = False
            G = netx.Graph()

            while not connected_graph_check:

                if self.network_model == 'ER':
                    if self.network_parameter is None:
                        self.network_parameter = 0.3
                    elif self.network_parameter < 0 or self.network_parameter > 1:
                        raise Exception('Check network model parameter')
                    G = netx.generators.random_graphs.erdos_renyi_graph(self.number_of_nodes,
                                                                        self.network_parameter)

                elif self.network_model == 'BA':
                    if self.network_parameter is None:
                        self.network_parameter = 2
                    elif self.network_parameter <= 0 or self.network_parameter >= self.number_of_nodes:
                        raise Exception('Check network model parameter')
                    G = netx.generators.random_graphs.barabasi_albert_graph(self.number_of_nodes,
                                                                            self.network_parameter)

                elif self.network_model == 'rand':
                    self.network_parameter = None
                    A = np.random.rand(self.number_of_nodes, self.number_of_nodes)
                    G = netx.from_numpy_array(A)

                elif self.network_model == 'path':
                    G = netx.generators.classic.path_graph(self.number_of_nodes)

                elif self.network_model == 'cycle':
                    G = netx.generators.classic.cycle_graph(self.number_of_nodes)

                elif self.network_model in ['eval_squeeze', 'eval_bound']:
                    if self.network_parameter is None:
                        self.network_parameter = 0.1
                    elif self.network_parameter < 0 or self.network_parameter > 1:
                        raise Exception('Check network model parameter')

                    A = np.random.rand(self.number_of_nodes, self.number_of_nodes)
                    A = A.T + A
                    _, V_mat = np.linalg.eig(A)

                    if self.network_model == 'eval_squeeze':
                        e = 2 * self.network_parameter * (
                                    0.5 - np.random.default_rng().random(self.number_of_nodes))
                    elif self.network_model == 'eval_bound':
                        e = 1 - self.network_parameter * np.random.default_rng().random(self.number_of_nodes)
                    else:
                        raise Exception('Check Network model')

                    self.self_loop = None
                    self.rho = None

                    e = np.array([i * -1 if coin_toss() else i for i in e])
                    A = V_mat @ np.diag(e) @ np.linalg.inv(V_mat)
                    G = netx.from_numpy_array(A)

                else:
                    raise Exception('Check Network model')

                connected_graph_check = netx.algorithms.components.is_connected(G)

            self.adjacency_matrix = netx.to_numpy_array(G)

            if self.self_loop:
                self.adjacency_matrix += np.identity(self.number_of_nodes)

        def rescale(self):
            if self.rho is not None:
                self.A = self.rho * self.adjacency_matrix/np.max(np.abs(np.linalg.eigvals(self.adjacency_matrix)))

        def evaluate_modes(self):
            self.open_loop_eig_vals = np.sort(np.linalg.eigvals(self.A))
            self.number_of_non_stable_modes = len([e for e in self.open_loop_eig_vals if e >= 1])

        def second_order_matrix(self):
            if self.second_order:
                if self.second_order_network_type == 1:
                    self.A = np.block([
                        [self.A, np.zeros_like(self.A)],
                        [self.second_order_scaling_factor * np.identity(self.number_of_nodes), self.second_order_scaling_factor * np.identity(self.number_of_nodes)]
                    ])
                elif self.second_order_network_type == 2:
                    self.A = np.block([
                        [np.identity(self.number_of_nodes), np.zeros_like(self.A)],
                        [self.second_order_scaling_factor * self.A, self.second_order_scaling_factor * np.identity(self.number_of_nodes)]
                    ])
                else:
                    raise SecondOrderError()

        def rescale_wrapper(self):
            self.rescale()
            self.second_order_matrix()
            self.evaluate_modes()

    class Architecture:
        def __init__(self, architecture_type):
            self.architecture_type = architecture_type if architecture_type in ['B', 'C'] else False
            if not architecture_type:
                raise ArchitectureError

            self.number_of_nodes = 0
            self.number_of_states = 0
            self.second_order_architecture_type = False

            self.min = 0
            self.max = 0

            self.Q = 0
            self.R1 = 0
            self.R1_reference = 0
            self.R2 = 0
            self.R3 = 0

            self.available_vectors = {}
            self.number_of_available = 0
            self.available_indices = []
            self.active_set = []

            self.active_matrix = 0
            self.indicator_vector_current = 0
            self.indicator_vector_history = 0

            self.history_active_set = []
            self.change_count = 0
            self.gain = {}

        def initialize_active_matrix(self):
            if self.architecture_type == 'B':
                self.active_matrix = np.zeros((self.number_of_nodes, self.number_of_available))
            elif self.architecture_type == 'C':
                self.active_matrix = np.zeros((self.number_of_available, self.number_of_nodes))
            else:
                raise ArchitectureError

        def initialize_available_vectors_as_basis_vectors(self, number_of_nodes, number_of_states):
            set_mat = np.identity(number_of_states)
            if self.second_order_architecture_type == 1:
                set_mat = set_mat[:number_of_nodes, :]
            elif self.second_order_architecture_type == 2:
                set_mat = set_mat[number_of_nodes:, :]
            else:
                raise SecondOrderError

            if self.architecture_type == 'B':
                self.available_vectors = {i+1: np.expand_dims(set_mat[:, i], axis=1) for i in range(0, number_of_nodes)}
            elif self.architecture_type == 'C':
                self.available_vectors = {i+1: np.expand_dims(set_mat[:, i], axis=0) for i in range(0, number_of_nodes)}
            else:
                raise ArchitectureError

        def initialize_random_architecture_active_set(self, initialize_random):
            self.active_set = np.random.default_rng().choice(self.available_indices, size=initialize_random, replace=False)
        
        def architecture_update_to_matrix_from_active_set(self):
            self.initialize_active_matrix()
            if len(self.active_set) > 0:
                if self.architecture_type == 'B':
                    self.active_matrix[:, self.active_set] = self.available_vectors[self.active_set]
                    self.R1 = np.zeros_like(self.R1_reference)
                    self.R1[self.active_set, :][:, self.active_set] = self.R1_reference[self.active_set, :][:, self.active_set]

                elif self.architecture_type == 'C':
                    self.active_matrix[self.active_set, :] = self.available_vectors[self.active_set]

                else:
                    raise ArchitectureError

        def architecture_update_to_indicator_vector_from_active_set(self):
            self.indicator_vector_history = self.indicator_vector_current
            self.indicator_vector_current = np.zeros_like(self.available_indices)
            self.indicator_vector_current[self.active_set] = 1

        def architecture_update_wrapper_from_active_set(self):
            self.architecture_update_to_indicator_vector_from_active_set()
            self.architecture_update_to_matrix_from_active_set()
        
        def duplicate_architecture(self, reference_set, history_update_check=False):
            self.active_set = reference_set
            if history_update_check:
                self.architecture_update_to_indicator_vector_from_active_set()
            else:
                self.indicator_vector_current = np.zeros_like(self.available_indices)
                self.indicator_vector_current[self.active_set] = 1
            self.architecture_update_to_matrix_from_active_set()

    class Disturbance:
        def __init__(self):
            self.number_of_nodes = 0
            self.number_of_states = 0
            self.W = 1
            self.V = 1
            self.W_scaling = 1
            self.V_scaling = 1
            self.w_gen = 0
            self.v_gen = 0

            self.noise_model = None
            self.disturbance_step = None
            self.disturbance_number = None
            self.disturbance_magnitude = None

        def initialize_disturbance(self, simulation_time_horizon):
            self.W = np.identity(self.number_of_states) * self.W_scaling
            self.V = np.identity(self.number_of_nodes) * self.V_scaling

            self.w_gen = np.random.default_rng().multivariate_normal(np.zeros(self.number_of_states), self.W, simulation_time_horizon)
            self.v_gen = np.random.default_rng().multivariate_normal(np.zeros(self.number_of_nodes), self.V, simulation_time_horizon)

            if self.noise_model in ['process', 'measurement', 'combined']:
                for t in range(0, simulation_time_horizon, self.disturbance_step):
                    if self.noise_model in ['process', 'combined']:
                        self.w_gen[t][np.random.default_rng().choice(self.number_of_nodes, self.disturbance_number, replace=False)] = self.disturbance_magnitude * [coin_toss() for _ in range(0, self.disturbance_number)]
                    if self.noise_model in ['measurement', 'combined']:
                        self.v_gen[t] = self.disturbance_magnitude * [coin_toss() for _ in range(0, self.number_of_nodes)]

    class Simulation:
        def __init__(self):
            self.t_simulate = int(100)
            self.t_predict = int(10)
            self.model = None

    class Trajectory:
        def __init__(self):
            self.number_of_nodes = 0
            self.number_of_states = 0
            self.X0_covariance = 0
            self.X0_scaling = 1
            self.x = []
            self.x_estimate = []
            self.X_enhanced = []
            self.u = []
            self.error = []
            self.cost = System.Cost()

        def initialize_trajectory(self):
            self.X0_covariance = np.identity(self.number_of_states) * self.X0_scaling
            self.x = [np.random.default_rng().multivariate_normal(np.zeros(self.number_of_nodes), self.X0_covariance)]
            self.x_estimate = [np.random.default_rng().multivariate_normal(np.zeros(self.number_of_nodes), self.X0_covariance)]

    class Cost:
        def __init__(self):
            self.running = 0
            self.switching = 0
            self.control = 0
            self.stage = 0
            self.predicted = []
            self.true = []
            self.initial = []
            self.metric_control = 'x'
            self.metric_running = 1
            self.metric_switching = 1

    def __init__(self):
        self.A = self.Dynamics()
        self.B = self.Architecture('B')
        self.C = self.Architecture('C')
        self.disturbance = self.Disturbance()
        self.sim = self.Simulation()
        self.trajectory = self.Trajectory()

        self.model_name = ''

    def model_namer(self):
        self.model_name = 'model_n' + str(int(self.A.number_of_nodes)) + '_net' + self.A.network_model

        if self.A.rho is None:
            self.model_name = self.model_name + '_rhoNone'
        else:
            self.model_name = self.model_name + '_rho' + str(np.round(self.A.rho, decimals=2))

        if self.A.second_order:
            self.model_name = self.model_name + '_secondorder'

        self.model_name = self.model_name + '_arch' + str(self.B.max) + '_Tp' + str(self.sim.t_predict)

        if self.disturbance.noise_model is not None:
            self.model_name = self.model_name + '_' + self.disturbance.noise_model

        if self.sim.model is not None:
            self.model_name = self.model_name + '_' + self.sim.model

    def initialize_system_from_experiment_parameters(self, experiment_parameters, experiment_keys):
        # parameter_keys = ['number_of_nodes', 'network_model', 'second_order', 'second_order_network', 'initial_architecture_size', 'second_order_architecture', 'disturbance_model', 'disturbance_step', 'disturbance_number', 'disturbance_magnitude', 'simulation_model', 'architecture_constraint', 'rho', 'network_parameter', 'disturbance_step', 'prediction_time_horizon']

        parameters = zip(experiment_keys, experiment_parameters)
        self.A.number_of_nodes = parameters['number_of_nodes']
        self.A.rho = parameters['rho']
        self.A.network_model = parameters['network_model']
        self.A.network_parameter = parameters['network_parameter']

        self.A.adjacency_matrix_initialize()

        self.A.second_order = parameters['second_order']
        if self.A.second_order:
            self.A.number_of_states = self.A.number_of_nodes * 2
        self.A.second_order_network_type = parameters['second_order_network']
        self.A.rescale_wrapper()

        self.update_network_size_parameters()

        self.sim.t_predict = parameters['t_sim']

        self.B.initialize_available_vectors_as_basis_vectors(self.A.number_of_nodes, self.A.number_of_states)
        self.C.initialize_available_vectors_as_basis_vectors(self.A.number_of_nodes, self.A.number_of_states)

        self.B.initialize_random_architecture_active_set(parameters['initial_architecture_size'])
        self.C.initialize_random_architecture_active_set(parameters['initial_architecture_size'])

        self.disturbance.W_scaling = parameters['W_scaling']
        self.disturbance.V_scaling = parameters['V_scaling']

        self.disturbance.noise_model = parameters['disturbance_model']
        if self.disturbance.noise_model is not None:
            self.disturbance.disturbance_step = parameters['disturbance_step']
            self.disturbance.disturbance_number = parameters['disturbance_number']
            self.disturbance.disturbance_magnitude = parameters['disturbance_magnitude']

        self.disturbance.initialize_disturbance(self.sim.t_simulate)

        self.trajectory.X0_scaling = parameters['X0_scaling']

        self.trajectory.initialize_trajectory()

    def update_network_size_parameters(self):
        self.B.number_of_nodes = self.A.number_of_nodes
        self.C.number_of_nodes = self.A.number_of_nodes
        self.disturbance.number_of_nodes = self.A.number_of_nodes
        self.trajectory.number_of_nodes = self.A.number_of_nodes

        self.B.number_of_states = self.A.number_of_states
        self.C.number_of_states = self.A.number_of_states
        self.disturbance.number_of_states = self.A.number_of_states
        self.trajectory.number_of_states = self.A.number_of_states
    
    def architecture_matrix_update(self, update_list=None):
        update_list = [update_list] if update_list in ['A', 'B'] else ['A', 'B'] if update_list is None else False
        if not update_list:
            raise ArchitectureError

        for arch in update_list:
            if arch == 'B':
                self.B.initialize_active_matrix()
                self.B.architecture_update_wrapper_from_active_set()
            elif arch == 'C':
                self.C.initialize_active_matrix()
                self.C.architecture_update_wrapper_from_active_set()
    
    def duplicate_architecture(self, reference_system):
        if not isinstance(reference_system, System):
            raise ClassError
        
        for arch in ['B', 'C']:
            if arch == 'B':
                self.B.duplicate_architecture(reference_system.B)
            elif arch == 'C':
                self.C.duplicate_architecture(reference_system.C)
            else:
                raise ArchitectureError


def coin_toss():
    return np.random.default_rng().random() > 0
