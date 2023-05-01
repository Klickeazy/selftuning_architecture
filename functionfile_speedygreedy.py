import numpy as np
import networkx as netx
import random
import time
import scipy as scp
from copy import deepcopy as dc
import shelve
import os
import socket
from itertools import product
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

        self.parameter_keys = ['experiment_no', 'number_of_nodes', 'network_model', 'second_order', 'second_order_network', 'initial_architecture_size', 'second_order_architecture', 'W_scaling', 'V_scaling', 'disturbance_model', 'disturbance_step', 'disturbance_number', 'disturbance_magnitude', 'simulation_model', 'architecture_constraint_min', 'architecture_constraint_max', 'rho', 'network_parameter', 'prediction_time_horizon', 'X0_scaling']
        self.parameter_dtypes = [int, int, str, bool, int, int, int, float, float, str, int, int, int, str, int, int, float, float, int, float]
        self.dtype_map = dict(zip(self.parameter_keys, self.parameter_dtypes))
        self.sample_values = [1, 20, 'rand', False, 0, 2, 0, 1, 1, None, 0, 0, 0, None, 2, 2, 3, 0, 10, 1]

        self.parameter_table = pd.DataFrame()
        self.experiments_list = []
        self.parameter_values = []

        self.read_table_from_file()

    def initialize_table(self):
        print('Initializing table with default parameters')
        self.parameter_values = [[k] for k in self.sample_values]
        self.parameter_table = pd.DataFrame(dict(zip(self.parameter_keys, self.parameter_values)))
        self.parameter_table.set_index(self.parameter_keys[0], inplace=True)
        self.write_table_to_file()

    def check_dimensions(self, print_check=False):
        if len(self.parameter_values) == len(self.parameter_dtypes) == len(self.parameter_keys):
            if print_check:
                print('Dimensions agree')
        else:
            raise Exception("Dimension mismatch - values: {}, dtype: {}, keys: {}".format(len(self.parameter_values), len(self.parameter_dtypes), len(self.parameter_keys)))

    def read_table_from_file(self):
        if not os.path.exists(datadump_folder_path + self.save_filename):
            raise Warning('File does not exist')
        else:
            self.parameter_table = pd.read_csv(datadump_folder_path + self.save_filename, index_col=0, dtype=self.dtype_map)
            self.parameter_table.replace({np.nan: None}, inplace=True)
            self.experiments_list = self.parameter_table.index

    def read_parameters_from_table(self, experiment_no=1):
        if experiment_no not in self.experiments_list:
            raise Exception('Experiment parameters not in table')
        if not isinstance(self.parameter_table, pd.DataFrame):
            raise Exception('Not a pandas frame')

        self.parameter_values = [k for k in self.parameter_table.loc[experiment_no]]

    def parameter_value_map(self):
        self.parameter_values = [list(map(d, [v]))[0] if v is not None else None for d, v in zip(self.parameter_dtypes, self.parameter_values)]

    def write_parameters_to_table(self):
        if len(self.parameter_values) == 0:
            raise Exception('No experiment parameters provided')
        self.check_dimensions()
        self.parameter_value_map()
        append_check = True
        for i in self.experiments_list:
            if [k for k in self.parameter_table.loc[i]][:] == self.parameter_values[1:]:
                print('Duplicate experiment at :', i)
                append_check = False
                break
        if append_check:
            self.parameter_table.loc[max(self.experiments_list)+1] = self.parameter_values[1:]
            self.experiments_list = self.parameter_table.index
            self.write_table_to_file()

    def write_table_to_file(self):
        self.parameter_table.to_csv(datadump_folder_path + self.save_filename)
        print('Printing done')

    def return_keys_values(self):
        return self.parameter_values, self.parameter_keys

    def display_test_parameters(self):
        print(self.parameter_table)


def test_all_experiments():
    Exp = Experiment()
    print('Experiments:', Exp.experiments_list)
    for i in Exp.experiments_list:
        S_test = initialize_system_from_experiment_number(i)


def initialize_system_from_experiment_number(exp_no=1):
    print('Exp No:', exp_no)
    exp = Experiment()
    exp.read_parameters_from_table(exp_no)
    print('Parameters:', exp.parameter_values)
    S = System()
    S.initialize_system_from_experiment_parameters(exp.parameter_values, exp.parameter_keys[1:])
    return S



class System:

    class Dynamics:
        def __init__(self):

            self.rho = 1
            self.network_model = 'rand'
            self.network_parameter = 0
            self.self_loop = True

            self.second_order = False
            self.second_order_scaling_factor = 1
            self.second_order_network_type = 1

            self.open_loop_eig_vals = 0
            self.adjacency_matrix = 0
            self.number_of_non_stable_modes = 0
            self.A_mat = np.zeros((0, 0))

            self.A_enhanced_mat = {}

    class Architecture:
        def __init__(self, architecture_type):
            self.architecture_type = architecture_type if architecture_type in ['B', 'C'] else 'Error'
            if architecture_type == 'Error':
                raise ArchitectureError

            self.second_order_architecture_type = 0

            self.min = 0
            self.max = 0

            self.Q = 0
            self.R1 = 0
            self.R1_reference = 0
            self.R2 = 0
            self.R3 = 0
            self.recursion_matrix = {}

            self.available_vectors = {}
            self.number_of_available = 0
            self.available_indices = []
            self.active_set = []

            self.active_matrix = np.zeros((0, 0))
            self.indicator_vector_current = 0
            self.indicator_vector_history = 0

            self.history_active_set = []
            self.change_count = 0
            self.gain = {}

    class Disturbance:
        def __init__(self):
            self.W = 1
            self.V = 1
            self.W_scaling = 1
            self.V_scaling = 1
            self.w_gen = 0
            self.v_gen = 0

            self.noise_model = None
            self.disturbance_step = 0
            self.disturbance_number = 0
            self.disturbance_magnitude = 0

    class Simulation:
        def __init__(self):
            self.t_simulate = int(100)
            self.t_predict = int(10)
            self.test_model = None

    class Trajectory:
        def __init__(self):
            self.X0_covariance = 0
            self.X0_scaling = 1
            self.x = []
            self.x_estimate = []
            self.X_enhanced = []
            self.u = []
            self.error = []
            self.control_matrix = []
            self.estimation_matrix = []
            self.error_2norm = []
            self.cost = System.Cost()

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
        self.number_of_nodes = 20
        self.number_of_states = 20
        self.A = self.Dynamics()
        self.B = self.Architecture('B')
        self.C = self.Architecture('C')
        self.disturbance = self.Disturbance()
        self.sim = self.Simulation()
        self.trajectory = self.Trajectory()

        self.model_name = ''

    def model_namer(self):
        self.model_name = 'model_n' + str(int(self.number_of_nodes)) + '_net' + self.A.network_model

        if self.A.rho is None:
            self.model_name = self.model_name + '_rhoNone'
        else:
            self.model_name = self.model_name + '_rho' + str(np.round(self.A.rho, decimals=2))

        if self.A.second_order:
            self.model_name = self.model_name + '_secondorder'

        self.model_name = self.model_name + '_arch' + str(self.B.max) + '_Tp' + str(self.sim.t_predict)

        if self.disturbance.noise_model is not None:
            self.model_name = self.model_name + '_' + self.disturbance.noise_model

        if self.sim.test_model is not None:
            self.model_name = self.model_name + '_' + self.sim.test_model

    def initialize_system_from_experiment_parameters(self, experiment_parameters, experiment_keys):
        # parameter_keys = ['number_of_nodes', 'network_model', 'second_order', 'second_order_network', 'initial_architecture_size', 'second_order_architecture', 'disturbance_model', 'disturbance_step', 'disturbance_number', 'disturbance_magnitude', 'simulation_model', 'architecture_constraint_min', 'architecture_constraint_max', 'rho', 'network_parameter', 'prediction_time_horizon']

        parameters = dict(zip(experiment_keys, experiment_parameters))

        self.number_of_nodes = parameters['number_of_nodes']
        self.number_of_states = parameters['number_of_nodes']
        self.A.rho = parameters['rho']
        self.A.network_model = parameters['network_model']
        self.A.network_parameter = parameters['network_parameter']

        self.adjacency_matrix_initialize()

        self.A.second_order = parameters['second_order']
        if self.A.second_order:
            self.number_of_states = self.number_of_nodes * 2
            self.A.second_order_network_type = parameters['second_order_network']
            self.B.second_order_architecture_type = parameters['second_order_architecture']
            self.C.second_order_architecture_type = parameters['second_order_architecture']
        self.rescale_wrapper()

        self.sim.t_predict = parameters['prediction_time_horizon']
        self.sim.test_model = 'simulation_model'

        self.B.min = parameters['architecture_constraint_min']
        self.B.max = parameters['architecture_constraint_max']
        self.C.min = parameters['architecture_constraint_min']
        self.C.max = parameters['architecture_constraint_max']

        self.initialize_available_vectors_as_basis_vectors()
        self.initialize_random_architecture_active_set(parameters['initial_architecture_size'])

        self.disturbance.W_scaling = parameters['W_scaling']
        self.disturbance.V_scaling = parameters['V_scaling']

        self.disturbance.noise_model = parameters['disturbance_model']
        if self.disturbance.noise_model is not None:
            self.disturbance.disturbance_step = parameters['disturbance_step']
            self.disturbance.disturbance_number = parameters['disturbance_number']
            self.disturbance.disturbance_magnitude = parameters['disturbance_magnitude']
        self.initialize_disturbance()

        self.C.Q = self.disturbance.W
        self.C.R1_reference = self.disturbance.V

        self.trajectory.X0_scaling = parameters['X0_scaling']
        self.initialize_trajectory()

    def adjacency_matrix_initialize(self):
        if self.A.network_model not in ['rand', 'ER', 'BA', 'path', 'cycle', 'eval_squeeze', 'eval_bound']:
            raise Exception('Network model not defined: {}'.format(self.A.network_model))

        connected_graph_check = False
        G = netx.Graph()

        while not connected_graph_check:

            if self.A.network_model == 'ER':
                if self.A.network_parameter is None:
                    self.A.network_parameter = 0.3
                elif self.A.network_parameter < 0 or self.A.network_parameter > 1:
                    raise Exception('Check network model parameter')
                G = netx.generators.random_graphs.erdos_renyi_graph(self.number_of_nodes, self.A.network_parameter)

            elif self.A.network_model == 'BA':
                if self.A.network_parameter is None:
                    self.A.network_parameter = 2
                elif self.A.network_parameter <= 0 or self.A.network_parameter >= self.number_of_nodes:
                    raise Exception('Check network model parameter')
                G = netx.generators.random_graphs.barabasi_albert_graph(self.number_of_nodes, self.A.network_parameter)

            elif self.A.network_model == 'rand':
                self.A.network_parameter = None
                A = np.random.rand(self.number_of_nodes, self.number_of_nodes)
                G = netx.from_numpy_array(A)

            elif self.A.network_model == 'path':
                G = netx.generators.classic.path_graph(self.number_of_nodes)

            elif self.A.network_model == 'cycle':
                G = netx.generators.classic.cycle_graph(self.number_of_nodes)

            elif self.A.network_model in ['eval_squeeze', 'eval_bound']:
                if self.A.network_parameter is None:
                    self.A.network_parameter = 0.1
                elif self.A.network_parameter < 0 or self.A.network_parameter > 1:
                    raise Exception('Check network model parameter')

                A = np.random.rand(self.number_of_nodes, self.number_of_nodes)
                A = A.T + A
                _, V_mat = np.linalg.eig(A)

                if self.A.network_model == 'eval_squeeze':
                    e = 2 * self.A.network_parameter * (
                            0.5 - np.random.default_rng().random(self.number_of_nodes))
                elif self.A.network_model == 'eval_bound':
                    e = 1 - self.A.network_parameter * np.random.default_rng().random(self.number_of_nodes)
                else:
                    raise Exception('Check Network model')

                self.A.self_loop = None
                self.A.rho = None

                e = np.array([i * -1 if coin_toss() else i for i in e])
                A = V_mat @ np.diag(e) @ np.linalg.inv(V_mat)
                G = netx.from_numpy_array(A)

            else:
                raise Exception('Check Network model')

            connected_graph_check = netx.algorithms.components.is_connected(G)

        self.A.adjacency_matrix = netx.to_numpy_array(G)

        if self.A.self_loop:
            self.A.adjacency_matrix += np.identity(self.number_of_nodes)

    def rescale(self):
        if self.A.rho is not None:
            self.A.A_mat = self.A.rho * self.A.adjacency_matrix / np.max(np.abs(np.linalg.eigvals(self.A.adjacency_matrix)))
        else:
            self.A.A_mat = self.A.adjacency_matrix

    def evaluate_modes(self):
        self.A.open_loop_eig_vals = np.sort(np.linalg.eigvals(self.A.A_mat))
        self.A.number_of_non_stable_modes = len([e for e in self.A.open_loop_eig_vals if e >= 1])

    def second_order_matrix(self):
        if self.A.second_order:
            if self.A.second_order_network_type == 1:
                self.A.A_mat = np.block([
                    [self.A.A_mat, np.zeros_like(self.A.A_mat)],
                    [self.A.second_order_scaling_factor * np.identity(self.number_of_nodes), self.A.second_order_scaling_factor * np.identity(self.number_of_nodes)]
                ])
            elif self.A.second_order_network_type == 2:
                self.A.A_mat = np.block([
                    [np.identity(self.number_of_nodes), np.zeros_like(self.A.A_mat)],
                    [self.A.second_order_scaling_factor * self.A.A_mat, self.A.second_order_scaling_factor * np.identity(self.number_of_nodes)]])
            else:
                raise SecondOrderError()

    def rescale_wrapper(self):
        self.rescale()
        self.second_order_matrix()
        self.evaluate_modes()

    @staticmethod
    def architecture_iterator(arch=None):
        arch = [arch] if arch in ['B', 'C'] else ['B', 'C'] if arch is None else 'Error'
        if arch == 'Error':
            raise ArchitectureError
        return arch

    def initialize_active_matrix(self, arch=None):
        arch = self.architecture_iterator(arch)
        for a in arch:
            if a == 'B':
                self.B.active_matrix = np.zeros((self.number_of_nodes, self.B.number_of_available))
            else:  # self.architecture_type == 'C':
                self.C.active_matrix = np.zeros((self.C.number_of_available, self.number_of_nodes))

    def initialize_available_vectors_as_basis_vectors(self, arch=None):
        arch = self.architecture_iterator(arch)

        for a in arch:
            set_mat = np.identity(self.number_of_states)

            if a == 'B':
                if self.A.second_order:
                    if self.B.second_order_architecture_type == 1:
                        set_mat = set_mat[:, :self.number_of_nodes]
                    elif self.B.second_order_architecture_type == 2:
                        set_mat = set_mat[:, self.number_of_nodes:]
                    else:
                        raise SecondOrderError
                self.B.available_vectors = {i+1: np.expand_dims(set_mat[:, i], axis=1) for i in range(0, self.number_of_nodes)}
                self.B.available_indices = [i for i in self.B.available_vectors]

            else:  # a == 'C'
                if self.A.second_order:
                    if self.C.second_order_architecture_type == 1:
                        set_mat = set_mat[:self.number_of_nodes, :]
                    elif self.C.second_order_architecture_type == 2:
                        set_mat = set_mat[self.number_of_nodes:, :]
                    else:
                        raise SecondOrderError
                self.C.available_vectors = {i + 1: np.expand_dims(set_mat[:, i], axis=0) for i in range(0, self.number_of_nodes)}
                self.C.available_indices = [i for i in self.C.available_vectors]

    def initialize_random_architecture_active_set(self, initialize_random, arch=None):
        arch = self.architecture_iterator(arch)

        for a in arch:
            if a == 'B':
                self.B.active_set = np.random.default_rng().choice(self.B.available_indices, size=initialize_random, replace=False)
            else:  # a == 'C'
                self.C.active_set = np.random.default_rng().choice(self.C.available_indices, size=initialize_random, replace=False)

    def architecture_limit_set(self, arch=None, min_set=None, max_set=None):
        arch = self.architecture_iterator(arch)
        for a in arch:
            if a == 'B':
                self.B.min = self.B.number_of_available if self.B.min is None else min_set if min_set is not None else self.B.min
                self.B.max = self.B.number_of_available if self.B.max is None else max_set if max_set is not None else self.B.max
            else:  # a == 'C'
                self.C.min = self.C.number_of_available if self.C.min is None else min_set if min_set is not None else self.C.min
                self.C.max = self.C.number_of_available if self.C.max is None else max_set if max_set is not None else self.C.max

    def architecture_limit_mod(self, min_mod=None, max_mod=None, arch=None):
        arch = self.architecture_iterator(arch)
        for a in arch:
            if a == 'B':
                self.B.min += min_mod if min_mod is not None else 1
                self.B.max += max_mod if max_mod is not None else 1
            else:  # a == 'C'
                self.C.min += min_mod if min_mod is not None else 1
                self.C.max += max_mod if max_mod is not None else 1

    def architecture_update_to_matrix_from_active_set(self, arch=None):
        arch = self.architecture_iterator(arch)
        self.initialize_active_matrix(arch)

        for a in arch:
            if a == 'B':
                if len(self.B.active_set) > 0:
                    self.B.active_set = np.sort(self.B.active_set)
                    self.B.active_matrix[:, self.B.active_set] = self.B.available_vectors[self.B.active_set]
                    self.B.R1 = np.zeros_like(self.B.R1_reference)
                    self.B.R1[self.B.active_set, :][:, self.B.active_set] = self.B.R1_reference[self.B.active_set, :][:, self.B.active_set]
            else:  # a == 'C'
                if len(self.C.active_set) > 0:
                    self.C.active_set = np.sort(self.C.active_set)
                    self.C.active_matrix[self.C.active_set, :] = self.C.available_vectors[self.C.active_set]

    def architecture_update_to_indicator_vector_from_active_set(self, arch=None):
        arch = self.architecture_iterator(arch)

        for a in arch:
            if a == 'B':
                self.B.indicator_vector_history = self.B.indicator_vector_current
                self.B.indicator_vector_current = np.zeros_like(self.B.available_indices)
                self.B.indicator_vector_current[self.B.active_set] = 1
            else:  # a == 'C'
                self.C.indicator_vector_history = self.C.indicator_vector_current
                self.C.indicator_vector_current = np.zeros_like(self.C.available_indices)
                self.C.indicator_vector_current[self.C.active_set] = 1

    def architecture_update_wrapper_from_active_set(self, arch=None):
        self.architecture_update_to_indicator_vector_from_active_set(arch)
        self.architecture_update_to_matrix_from_active_set(arch)

    def architecture_duplicate(self, reference_system, history_update_check=False):
        if not isinstance(reference_system, System):
            raise ClassError

        for arch in ['B', 'C']:
            if arch == 'B':
                self.B.active_set = reference_system.B.active_set
            else:
                self.C.active_set = reference_system.C.active_set

        if history_update_check:
            self.architecture_update_wrapper_from_active_set()

    def architecture_active_set_update(self):
        arch = self.architecture_iterator(None)
        for a in arch:
            if a == 'B':
                self.B.history_active_set.append(self.B.active_set)
            else:  # a == 'C'
                self.C.history_active_set.append(self.C.active_set)
        self.architecture_update_wrapper_from_active_set(None)

    def initialize_disturbance(self):
        self.disturbance.W = np.identity(self.number_of_states) * self.disturbance.W_scaling
        self.disturbance.V = np.identity(self.number_of_nodes) * self.disturbance.V_scaling

        self.disturbance.w_gen = np.random.default_rng().multivariate_normal(np.zeros(self.number_of_states), self.disturbance.W, self.sim.t_simulate)
        self.disturbance.v_gen = np.random.default_rng().multivariate_normal(np.zeros(self.number_of_nodes), self.disturbance.V, self.sim.t_simulate)

        if self.disturbance.noise_model in ['process', 'measurement', 'combined']:
            if self.disturbance.disturbance_number == 0 or self.disturbance.disturbance_magnitude == 0 or self.disturbance.disturbance_step == 0:
                raise Exception('Check disturbance parameters')
            for t in range(0, self.sim.t_simulate, self.disturbance.disturbance_step):
                if self.disturbance.noise_model in ['process', 'combined']:
                    self.disturbance.w_gen[t][np.random.default_rng().choice(self.number_of_nodes, self.disturbance.disturbance_number, replace=False)] = self.disturbance.disturbance_magnitude * np.array([coin_toss() for _ in range(0, self.disturbance.disturbance_number)])
                if self.disturbance.noise_model in ['measurement', 'combined']:
                    self.disturbance.v_gen[t] = self.disturbance.disturbance_magnitude * np.array([coin_toss() for _ in range(0, self.number_of_nodes)])

    def initialize_trajectory(self):
        self.trajectory.X0_covariance = np.identity(self.number_of_states) * self.trajectory.X0_scaling

        self.trajectory.x = [np.random.default_rng().multivariate_normal(np.zeros(self.number_of_states), self.trajectory.X0_covariance)]

        self.trajectory.x_estimate = [np.random.default_rng().multivariate_normal(np.zeros(self.number_of_states), self.trajectory.X0_covariance)]

        self.trajectory.X_enhanced = [np.squeeze(np.concatenate((self.trajectory.x[-1], self.trajectory.x_estimate[-1])))]

        self.trajectory.control_matrix = {}
        self.trajectory.estimation_matrix = {0: np.identity(self.number_of_states)}

        self.trajectory.error = [self.trajectory.x[-1] - self.trajectory.x_estimate[-1]]
        self.trajectory.error_2norm = [np.linalg.norm(self.trajectory.error[-1])]

    def architecture_check_min_limits(self, architecture_set, arch):
        if arch not in ['B', 'C']:
            raise ArchitectureError
        if arch == 'B':
            return self.B.min <= len(architecture_set)
        else:  # arch == 'B':
            return self.C.min <= len(architecture_set)

    def architecture_check_max_limits(self, architecture_set, arch):
        if arch not in ['B', 'C']:
            raise ArchitectureError
        if arch == 'B':
            return len(architecture_set) <= self.B.max
        else:  # arch == 'B':
            return len(architecture_set) <= self.C.max

    def available_selection_choices(self):
        choices = []
        if len(self.B.active_set) <= self.B.min and len(self.C.active_set) <= self.C.min:
            choices = [(None, None, None)]
        if len(self.B.active_set) < self.B.max:
            for i in compare_lists(self.B.active_set, self.B.available_indices)['only2']:
                choices.append(('B', i, '+'))
        if len(self.C.active_set) < self.C.max:
            for i in compare_lists(self.C.active_set, self.C.available_indices)['only2']:
                choices.append(('C', i, '+'))
        return choices

    def available_rejection_choices(self):
        choices = []
        if len(self.B.active_set) <= self.B.max and len(self.C.active_set) <= self.C.max:
            choices = [(None, None, None)]
        if len(self.B.active_set) > self.B.min:
            for i in self.B.active_set:
                choices.append(('B', i, '-'))
        if len(self.C.active_set) > self.C.min:
            for i in self.C.active_set:
                choices.append(('C', i, '-'))
        return choices

    def update_active_set_from_choices(self, choice):
        if choice(0) == 'B':
            if choice(2) == '+':
                self.B.active_set.append(choice(1))
            elif choice(2) == '-':
                self.B.active_set = [k for k in self.B.active_set if k != choice(1)]
        elif choice(0) == 'C':
            if choice(2) == '+':
                self.C.active_set.append(choice(1))
            elif choice(2) == '-':
                self.C.active_set = [k for k in self.C.active_set if k != choice(1)]
        elif choice == (None, None, None):
            pass
        else:
            raise Exception('Invalid update choice')

    def prediction_control_gain(self):
        self.B.recursion_matrix = {self.sim.t_predict: dc(self.B.Q)}
        self.B.R1 = self.B.R1_reference[:, self.B.active_set][self.B.active_set, :]
        for t in range(self.sim.t_predict-1, -1, -1):
            self.B.gain[t] = np.linalg.inv((self.B.active_matrix.T @ self.B.recursion_matrix[t+1] @ self.B.active_matrix) + self.B.R1) @ self.B.active_matrix.T @ self.B.recursion_matrix[t+1] @ self.A.A_mat
            self.B.recursion_matrix[t] = (self.A.A_mat.T @ self.B.recursion_matrix[t+1] @ self.A.A_mat) - (self.A.A_mat.T @ self.B.recursion_matrix[t+1] @ self.B.active_matrix @ self.B.gain[t]) + self.B.Q
        self.trajectory.control_matrix.append(self.B.recursion_matrix[0])

    def prediction_estimation_gain(self):
        self.C.recursion_matrix = {0: self.trajectory.estimation_matrix[-1]}
        for t in range(0, self.sim.t_predict):
            self.C.gain[t] = self.C.recursion_matrix[t] @ self.C.active_matrix.T @ np.linalg.inv((self.C.active_matrix @ self.C.recursion_matrix[t] @ self.C.active_matrix.T) + self.C.R1)
            self.C.recursion_matrix[t+1] = (self.A.A_mat @ self.C.recursion_matrix[t] @ self.A.A_mat.T) - (self.A.A_mat @ self.C.gain[t] @ self.C.active_matrix @ self.C.recursion_matrix[t] @ self.A.A_mat.T) + self.C.Q
        self.trajectory.estimation_matrix.append(self.C.recursion_matrix[1])

    def prediction_cost_calculation(self):
        A_enhanced = {}
        W_enhanced = {}
        W_mat = np.block([
            [self.disturbance.W, np.zeros_like(self.disturbance.W)],
            [np.zeros_like(self.disturbance.W), self.disturbance.V]])
        for t in range(0, self.sim.t_predict):
            BKt = self.B.active_matrix @ self.B.gain[t]
            ALtC = self.A.A_mat @ self.C.gain[t] @ self.C.active_matrix
            A_enhanced[t] = np.block([
                [self.A.A_mat, -BKt],
                [ALtC, self.A.A_mat - ALtC - BKt]])
            F_enhanced = np.block([
                [np.identity(self.number_of_states), np.zeros(self.number_of_states)],
                [np.zeros(self.number_of_states), ALtC]])
            W_enhanced[t] = F_enhanced @ W_mat @ F_enhanced.T
        P = np.block_diag([self.B.Q, np.zeros(np.number_of_states)])
        # for t in range(0, self.sim.t_predict):





def coin_toss():
    return np.random.default_rng().random() > 0


def compare_lists(list1, list2):
    return {'only1': [k for k in list1 if k not in list2], 'only2': [k for k in list2 if k not in list1], 'both': [k for k in list1 if k in list2]}
