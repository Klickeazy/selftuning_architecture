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
from multiprocessing import Pool
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

        self.default_parameter_dtype_map = {'experiment_no': int(1),
                                            'number_of_nodes': int(20),
                                            'network_model': 'rand',
                                            'network_parameter': float(0),
                                            'rho': float(1),
                                            'second_order': False,
                                            'second_order_network': int(0),
                                            'initial_architecture_size': int(5),
                                            'architecture_constraint_min': int(5),
                                            'architecture_constraint_max': int(5),
                                            'second_order_architecture': int(0),
                                            'Q_cost_scaling': float(1),
                                            'R_cost_scaling': float(1),
                                            'B_run_cost': float(1),
                                            'C_run_cost': float(1),
                                            'B_switch_cost': float(1),
                                            'C_switch_cost': float(1),
                                            'W_scaling': float(1),
                                            'V_scaling': float(1),
                                            'disturbance_model': str(None),
                                            'disturbance_step': int(0),
                                            'disturbance_number': int(0),
                                            'disturbance_magnitude': int(0),
                                            'prediction_time_horizon': int(10),
                                            'simulation_model': str(None),
                                            'X0_scaling': float(1)}

        self.parameter_keys = list(self.default_parameter_dtype_map.keys())
        self.parameter_dtypes = {k: type(self.default_parameter_dtype_map[k]) for k in self.default_parameter_dtype_map}

        self.parameter_table = pd.DataFrame()
        self.experiments_list = []
        self.parameter_values = []
        self.read_table_from_file()

    def initialize_table(self):
        print('Initializing table with default parameters')
        self.parameter_values = [[k] for k in self.default_parameter_dtype_map.values()]
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
            self.parameter_table = pd.read_csv(datadump_folder_path + self.save_filename, index_col=0, dtype=self.parameter_dtypes)
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
    print('Testing all experiment models')
    Exp = Experiment()
    print('Experiments No_s:', [k for k in Exp.experiments_list])
    for i in tqdm(Exp.experiments_list):
        _ = initialize_system_from_experiment_number(i)
    print('Testing done')


def initialize_system_from_experiment_number(exp_no=1):
    exp = Experiment()
    exp.read_parameters_from_table(exp_no)
    S = System()
    S.initialize_system_from_experiment_parameters(exp.parameter_values, exp.parameter_keys[1:])
    return S


class System:

    class Dynamics:
        def __init__(self):
            # Parameters assigned from function file
            self.rho = 1                            # Scaling factor for eigenvalues
            self.network_model = 'rand'             # Network adjacency matrix
            self.network_parameter = 0              # Parameter for network adjacency matrix
            self.second_order = False               # Check for second order states - each node has 2 states associated with it
            self.second_order_scaling_factor = 1    # Scaling factor for second order equation
            self.second_order_network_type = 1      # Network type of second order states

            # Default parameter
            self.self_loop = True                   # Self-loops within the adjacency matrix

            # Evaluated
            self.open_loop_eig_vals = 0             # Vector of second order states
            self.adjacency_matrix = 0               # Adjacency matrix
            self.number_of_non_stable_modes = 0     # Number of unstable modes with magnitude >= 1
            self.A_mat = np.zeros((0, 0))           # Open-loop dynamics matrix

    class Architecture:
        def __init__(self):
            # Parameters assigned from function file
            self.second_order_architecture_type = 0  # Type of second-order architecture - which states do it control or estimate
            self.min = 1                             # Minimum architecture bounds
            self.max = 1                             # Maximum architecture bounds
            self.R2 = 0                              # Cost on running architecture
            self.R3 = 0                              # Cost on switching architecture

            # Calculated terms
            self.available_indices = []             # Indices of available architecture
            self.available_vectors = {}             # Vectors associated with available architecture
            self.number_of_available = 0            # Count of available architecture
            self.active_set = []                    # Set of indices of active architecture
            self.active_matrix = np.zeros((0, 0))   # Matrix of active architecture
            self.Q = np.zeros((0, 0))               # Cost on states/Process noise covariance
            self.R1 = np.zeros((0, 0))              # Cost on active actuators/Measurement noise covariance
            self.R1_reference = np.zeros((0, 0))    # Cost on available actuators
            self.indicator_vector_current = np.zeros(self.number_of_available)  # {1, 0} binary vector of currently active architecture - to compute running/switching costs
            self.indicator_vector_history = np.zeros(self.number_of_available)  # {1, 0} binary vector of previously active architecture - to compute switching costs
            self.history_active_set = []            # Record of active architecture over simulation horizon
            self.change_count = 0                   # Count of number of changes in architecture over simulation horizon
            self.recursion_matrix = {}              # Recursive cost matrix/estimation error covariance over the prediction horizon
            self.gain = {}                          # Gains calculated over the prediction horizon for the fixed architecture

    class Disturbance:
        def __init__(self):
            # Parameters assigned from function file
            self.W_scaling = 1                      # Scaling factor for process noise covariance
            self.V_scaling = 1                      # Scaling factor for measurement noise covariance
            self.noise_model = None                 # Noise model for targeted un-modelled disturbances
            self.disturbance_step = 0               # Number of steps between un-modelled disturbances
            self.disturbance_number = 0             # Number of states affected by un-modelled disturbances
            self.disturbance_magnitude = 0          # Scaling factor of un-modelled disturbances

            # Calculated terms
            self.W = np.zeros((0, 0))                # Process noise covariance
            self.V = np.zeros((0, 0))                # Measurement noise covariance
            self.w_gen = 0                          # Realization of process noise
            self.v_gen = 0                          # Realization of measurement noise

    class Simulation:
        def __init__(self):
            # Parameters assigned from func
            self.t_predict = int(10)                # Prediction time horizon
            self.test_model = None                  # Simulation model of actuators

            # Constant parameters
            self.t_simulate = int(100)              # Simulation time horizon
            self.plot_parameters = {'fixed': {'c': 'tab:blue', 'ls': 'dotted'},
                                    'tuning': {'c': 'tab:orange', 'ls': 'solid'}}
            self.network_matrix = np.zeros((0, 0))

    class Trajectory:
        def __init__(self):
            # Parameters assigned from function file
            self.X0_scaling = 1                     # Scaling factor of initial state

            # Calculated terms
            self.X0_covariance = np.zeros((0, 0))   # Initial state covariance
            self.x = []                             # True state trajectory
            self.x_estimate = []                    # Estimates state trajectory
            self.X_enhanced = []                    # Enhanced state trajectory
            self.u = []                             # Control input trajectory
            self.error = []                         # Estimation error trajectory
            self.control_matrix = []                # Control cost matrix at each timestep
            self.estimation_matrix = []             # Estimation error covariance matrix at each timestep
            self.error_2norm = []                   # 2-norm of estimation error trajectory
            self.cost = System.Cost()               # Cost variables
            self.computation_time = []              # Computation time at each simulation timestep
            self.number_of_choices = []             # Sum of number of choices over all iterations

    class Cost:
        def __init__(self):
            # Parameters assigned from function file
            self.metric_control = 1                 # Metric function to evaluate control costs
            self.metric_running = 1                 # Metric function to evaluate running costs
            self.metric_switching = 1               # Metric function to evaluate switching costs

            # Calculated terms
            self.running = 0                        # Running cost at current timestep
            self.switching = 0                      # Switching cost at current timestep
            self.control = 0                        # Control cost at current timestep
            self.predicted = []                     # Predicted total stage cost trajectory
            self.true = []                          # True total stage cost trajectory
            self.initial = []                       # Costs for initial architecture optimization
            self.predicted_matrix = {}              # Cost matrix over the prediction horizon

    def __init__(self):
        self.number_of_nodes = 20                   # Number of nodes in the network
        self.number_of_states = 20                  # Number of state in the network (affected by second_order dynamics)
        self.A = self.Dynamics()
        self.B = self.Architecture()
        self.C = self.Architecture()
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

        self.disturbance.W_scaling = parameters['W_scaling']
        self.disturbance.V_scaling = parameters['V_scaling']
        self.disturbance.noise_model = parameters['disturbance_model']
        if self.disturbance.noise_model is not None:
            self.disturbance.disturbance_step = parameters['disturbance_step']
            self.disturbance.disturbance_number = parameters['disturbance_number']
            self.disturbance.disturbance_magnitude = parameters['disturbance_magnitude']
        self.initialize_disturbance()

        self.B.R2 = parameters['B_run_cost']
        self.C.R2 = parameters['C_run_cost']
        self.B.R3 = parameters['B_switch_cost']
        self.C.R3 = parameters['C_switch_cost']

        self.architecture_limit_set(min_set=parameters['architecture_constraint_min'], max_set=parameters['architecture_constraint_max'])

        self.initialize_available_vectors_as_basis_vectors()

        self.B.Q = np.identity(self.number_of_states)*parameters['Q_cost_scaling']
        self.B.R1_reference = np.identity(self.B.number_of_available) * parameters['R_cost_scaling']
        self.C.Q = self.disturbance.W
        self.C.R1_reference = self.disturbance.V

        self.initialize_random_architecture_active_set(parameters['initial_architecture_size'])
        self.architecture_update_all_from_active_set()

        self.trajectory.X0_scaling = parameters['X0_scaling']
        self.initialize_trajectory()

        self.prediction_control_gain()
        self.prediction_estimation_gain()

        self.model_namer()

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
                    e = 2 * self.A.network_parameter * (0.5 - np.random.default_rng().random(self.number_of_nodes))
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
        if type(arch) == list and len(arch) == 1:
            arch = arch[0]
        arch = [arch] if arch in ['B', 'C'] else ['B', 'C'] if (arch is None or arch == ['B', 'C']) else 'Error'
        if arch == 'Error':
            raise ArchitectureError
        return arch

    def initialize_active_matrix(self, arch=None):
        arch = self.architecture_iterator(arch)
        for a in arch:
            if a == 'B':
                self.B.active_matrix = np.zeros((self.number_of_nodes, len(self.B.active_set)))
            else:  # self.architecture_type == 'C':
                self.C.active_matrix = np.zeros((len(self.C.active_set), self.number_of_nodes))

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
                self.B.available_vectors = {i: set_mat[:, i] for i in range(0, self.number_of_nodes)}
                self.B.available_indices = [i for i in self.B.available_vectors]
                self.B.number_of_available = len(self.B.available_indices)
                self.B.indicator_vector_history = np.zeros(self.B.number_of_available, dtype=int)
                self.B.indicator_vector_current = np.zeros(self.B.number_of_available, dtype=int)

            else:  # a == 'C'
                if self.A.second_order:
                    if self.C.second_order_architecture_type == 1:
                        set_mat = set_mat[:self.number_of_nodes, :]
                    elif self.C.second_order_architecture_type == 2:
                        set_mat = set_mat[self.number_of_nodes:, :]
                    else:
                        raise SecondOrderError
                self.C.available_vectors = {i: set_mat[:, i] for i in range(0, self.number_of_nodes)}
                self.C.available_indices = [i for i in self.C.available_vectors]
                self.C.number_of_available = len(self.B.available_indices)
                self.C.indicator_vector_history = np.zeros(self.C.number_of_available, dtype=int)
                self.C.indicator_vector_current = np.zeros(self.C.number_of_available, dtype=int)

    def initialize_random_architecture_active_set(self, initialize_random, arch=None):
        arch = self.architecture_iterator(arch)
        for a in arch:
            if a == 'B':
                self.B.active_set = list(np.sort(np.random.default_rng().choice(self.B.available_indices, size=initialize_random, replace=False)))
            else:  # a == 'C'
                self.C.active_set = list(np.sort(np.random.default_rng().choice(self.C.available_indices, size=initialize_random, replace=False)))

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

        self.trajectory.control_matrix = []
        self.trajectory.estimation_matrix = [np.identity(self.number_of_states)]

        self.trajectory.error = [self.trajectory.x[-1] - self.trajectory.x_estimate[-1]]
        self.trajectory.error_2norm = [np.linalg.norm(self.trajectory.error[-1])]

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
                    self.B.active_set = list(np.sort(self.B.active_set))
                    for k in range(0, len(self.B.active_set)):
                        self.B.active_matrix[:, k] = self.B.available_vectors[self.B.active_set[k]]
                    self.B.R1 = self.B.R1_reference[self.B.active_set, :][:, self.B.active_set]
            else:  # a == 'C'
                if len(self.C.active_set) > 0:
                    self.C.active_set = list(np.sort(self.C.active_set))
                    for k in range(0, len(self.C.active_set)):
                        self.C.active_matrix[k, :] = self.C.available_vectors[self.C.active_set[k]]
                    self.C.R1 = self.C.R1_reference[self.C.active_set, :][:, self.C.active_set]

    def architecture_update_to_indicator_vector_from_active_set(self, arch=None):
        arch = self.architecture_iterator(arch)
        for a in arch:
            if a == 'B':
                self.B.indicator_vector_history = dc(self.B.indicator_vector_current)
                self.B.indicator_vector_current = np.zeros_like(self.B.available_indices, dtype=int)
                self.B.indicator_vector_current[self.B.active_set] = int(1)
            else:  # a == 'C'
                self.C.indicator_vector_history = dc(self.C.indicator_vector_current)
                self.C.indicator_vector_current = np.zeros_like(self.C.available_indices, dtype=int)
                self.C.indicator_vector_current[self.C.active_set] = int(1)

    def architecture_update_to_history_from_active_set(self, arch=None):
        arch = self.architecture_iterator(arch)
        for a in arch:
            if a == 'B':
                self.B.history_active_set.append(self.B.active_set)
            else:  # a == 'C'
                self.C.history_active_set.append(self.C.active_set)

    def architecture_update_all_from_active_set(self, arch=None):
        self.architecture_update_to_history_from_active_set(arch)
        self.architecture_update_to_indicator_vector_from_active_set(arch)
        self.architecture_update_to_matrix_from_active_set(arch)

    def architecture_duplicate_active_set_from_system(self, reference_system, update_check=True):
        if not isinstance(reference_system, System):
            raise ClassError
        for a in ['B', 'C']:
            if a == 'B':
                self.B.active_set = reference_system.B.active_set
            else:
                self.C.active_set = reference_system.C.active_set
        if update_check:
            self.architecture_update_all_from_active_set()

    def architecture_compare_active_set_to_system(self, reference_system):
        if not isinstance(reference_system, System):
            raise ClassError
        return set(self.B.active_set) == set(reference_system.B.active_set) and set(self.C.active_set) == set(reference_system.C.active_set)

    def architecture_display_active_set(self):
        print('B:', self.B.active_set)
        print('C:', self.C.active_set)

    def prediction_gains(self, arch=None, update_trajectory_check=False):
        arch = self.architecture_iterator(arch)
        for a in arch:
            if a == 'B':
                self.prediction_control_gain(update_trajectory_check=update_trajectory_check)
            else:  # if a == 'C'
                self.prediction_estimation_gain(update_trajectory_check=update_trajectory_check)

    def prediction_control_gain(self, update_trajectory_check=False):
        self.B.recursion_matrix = {self.sim.t_predict: self.B.Q}
        for t in range(self.sim.t_predict-1, -1, -1):
            self.B.gain[t] = np.linalg.inv((self.B.active_matrix.T @ self.B.recursion_matrix[t+1] @ self.B.active_matrix) + self.B.R1) @ self.B.active_matrix.T @ self.B.recursion_matrix[t+1] @ self.A.A_mat
            self.B.recursion_matrix[t] = (self.A.A_mat.T @ self.B.recursion_matrix[t+1] @ self.A.A_mat) - (self.A.A_mat.T @ self.B.recursion_matrix[t+1] @ self.B.active_matrix @ self.B.gain[t]) + self.B.Q
        if update_trajectory_check:
            self.trajectory.control_matrix.append(self.B.recursion_matrix[0])

    def prediction_estimation_gain(self, update_trajectory_check=False):
        self.C.recursion_matrix = {0: self.trajectory.estimation_matrix[-1]}
        for t in range(0, self.sim.t_predict):
            self.C.gain[t] = self.C.recursion_matrix[t] @ self.C.active_matrix.T @ np.linalg.inv((self.C.active_matrix @ self.C.recursion_matrix[t] @ self.C.active_matrix.T) + self.C.R1)
            self.C.recursion_matrix[t+1] = (self.A.A_mat @ self.C.recursion_matrix[t] @ self.A.A_mat.T) - (self.A.A_mat @ self.C.gain[t] @ self.C.active_matrix @ self.C.recursion_matrix[t] @ self.A.A_mat.T) + self.C.Q
        if update_trajectory_check:
            self.trajectory.estimation_matrix.append(self.C.recursion_matrix[1])

    def cost_architecture_running(self):
        if self.trajectory.cost.metric_running == 0 or (self.B.R2 == 0 and self.C.R2 == 0):
            self.trajectory.cost.running = 0
        elif self.trajectory.cost.metric_running == 1 and type(self.B.R2) == int and type(self.C.R2) == int:
            self.trajectory.cost.running = np.linalg.norm(self.B.indicator_vector_current) * self.B.R2 + np.linalg.norm(self.C.indicator_vector_current) * self.C.R2
        elif self.trajectory.cost.metric_running == 2 and np.shape(self.B.R2) == (len(self.B.active_set), len(self.B.active_set)) and np.shape(self.C.R2) == (len(self.C.active_set), len(self.C.active_set)):
            self.trajectory.cost.running = self.B.indicator_vector_current.T @ self.B.R2 @ self.B.indicator_vector_current + self.C.indicator_vector_current.T @ self.C.R2 @ self.C.indicator_vector_current
        else:
            raise Exception('Check running cost metric')

    def cost_architecture_switching(self):
        if self.trajectory.cost.metric_switching == 0 or (self.B.R3 == 0 and self.C.R3 == 0):
            self.trajectory.cost.switching = 0
        elif self.trajectory.cost.metric_switching == 1 and type(self.B.R3) == int and type(self.C.R3) == int:
            self.trajectory.cost.switching = np.linalg.norm(self.B.indicator_vector_current - self.B.indicator_vector_history) * self.B.R3 + np.linalg.norm(self.C.indicator_vector_current - self.C.indicator_vector_history) * self.C.R3
        elif self.trajectory.cost.metric_switching == 2 and np.shape(self.B.R3) == (len(self.B.active_set), len(self.B.active_set)) and np.shape(self.C.R3) == (len(self.C.active_set), len(self.C.active_set)):
            self.trajectory.cost.switching = (self.B.indicator_vector_current - self.B.indicator_vector_history).T @ self.B.R2 @ (self.B.indicator_vector_current - self.B.indicator_vector_history) + (self.C.indicator_vector_current - self.C.indicator_vector_history).T @ self.C.R2 @ (self.C.indicator_vector_current - self.C.indicator_vector_history)
        else:
            raise Exception('Check switching cost metric')

    def cost_prediction(self):
        A_enhanced_mat = {}
        W_enhanced = {}
        Q_enhanced = {}
        W_mat = np.block([[self.disturbance.W, np.zeros_like(self.disturbance.W)],
                          [np.zeros_like(self.disturbance.W), self.disturbance.V]])
        for t in range(0, self.sim.t_predict):
            BKt = self.B.active_matrix @ self.B.gain[t]
            ALtC = self.A.A_mat @ self.C.gain[t] @ self.C.active_matrix
            A_enhanced_mat[t] = np.block([[self.A.A_mat, -BKt],
                                          [ALtC, self.A.A_mat - ALtC - BKt]])
            F_enhanced = np.block([[np.identity(self.number_of_states), np.zeros((self.number_of_states, self.C.number_of_available))],
                                   [np.zeros((self.C.number_of_available, self.number_of_states)), ALtC]])
            W_enhanced[t] = F_enhanced @ W_mat @ F_enhanced.T
            Q_enhanced[t] = np.block([[self.B.Q, np.zeros((self.number_of_states, self.B.number_of_available))],
                                      [np.zeros((self.B.number_of_available, self.number_of_states)), self.B.gain[t].T @ self.B.R1 @ self.B.gain[t]]])

        Q_enhanced[self.sim.t_predict] = np.block([[self.B.Q, np.zeros((self.number_of_states, self.number_of_states))],
                                                   [np.zeros((self.number_of_states, self.number_of_states)), np.zeros((self.number_of_states, self.number_of_states))]])

        self.trajectory.cost.predicted_matrix = {self.sim.t_predict: Q_enhanced[self.sim.t_predict]}
        self.trajectory.cost.control = 0

        for t in range(self.sim.t_predict-1, -1, -1):
            self.trajectory.cost.control += np.trace(self.trajectory.cost.predicted_matrix[t+1] @ W_enhanced[t])
            self.trajectory.cost.predicted_matrix[t] = A_enhanced_mat[t].T @ self.trajectory.cost.predicted_matrix[t+1] @ A_enhanced_mat[t]
        if self.trajectory.cost.metric_control == 1:
            x_estimate_stack = np.squeeze(np.tile(self.trajectory.x_estimate[-1], (1, 2)))
            self.trajectory.cost.control += (x_estimate_stack.T @ self.trajectory.cost.predicted_matrix[0] @ x_estimate_stack)
        elif self.trajectory.cost.metric_control == 2:
            self.trajectory.cost.control += np.max(np.linalg.eigvals(self.trajectory.cost.predicted_matrix[0]))
        else:
            raise Exception('Check control cost metric')

    def cost_prediction_wrapper(self, arch=None, gain_update_check=True):
        if gain_update_check:
            self.prediction_gains(arch)

        self.cost_prediction()
        self.cost_architecture_running()
        self.cost_architecture_switching()
        self.trajectory.cost.predicted.append(self.trajectory.cost.control + self.trajectory.cost.running + self.trajectory.cost.switching)

    def cost_display_stage_components(self):
        print('Running:', self.trajectory.cost.running)
        print('Switching:', self.trajectory.cost.switching)
        print('Predicted control:', self.trajectory.cost.control)

        print('Predicted total:', self.trajectory.cost.predicted[-1])
        print('Predicted array:', self.trajectory.cost.predicted)
        # print('True total:', self.trajectory.cost.true[-1])

    def available_selection_choices(self):
        if len(self.B.active_set) >= self.B.min and len(self.C.active_set) >= self.C.min:
            choices = [[None, None, None]]
        else:
            choices = []
        if len(self.B.active_set) < self.B.max:
            for i in compare_lists(self.B.active_set, self.B.available_indices)['only2']:
                choices.append(['B', i, '+'])
        if len(self.C.active_set) < self.C.max:
            for i in compare_lists(self.C.active_set, self.C.available_indices)['only2']:
                choices.append(['C', i, '+'])
        return choices

    def available_rejection_choices(self):
        if len(self.B.active_set) <= self.B.max and len(self.C.active_set) <= self.C.max:
            choices = [[None, None, None]]
        else:
            choices = []
        if len(self.B.active_set) > self.B.min:
            for i in self.B.active_set:
                choices.append(['B', i, '-'])
        if len(self.C.active_set) > self.C.min:
            for i in self.C.active_set:
                choices.append(['C', i, '-'])
        return choices

    def architecture_update_active_set_from_choices(self, architecture_change_parameters):
        if architecture_change_parameters[0] == 'B':
            if architecture_change_parameters[2] == '+':
                self.B.active_set.append(architecture_change_parameters[1])
                self.B.active_set = list(np.sort(np.array(self.B.active_set)))
            elif architecture_change_parameters[2] == '-':
                self.B.active_set = [k for k in self.B.active_set if k != architecture_change_parameters[1]]
        elif architecture_change_parameters[0] == 'C':
            if architecture_change_parameters[2] == '+':
                self.C.active_set.append(architecture_change_parameters[1])
                self.C.active_set = list(np.sort(np.array(self.C.active_set)))
            elif architecture_change_parameters[2] == '-':
                self.C.active_set = [k for k in self.C.active_set if k != architecture_change_parameters[1]]
        elif architecture_change_parameters == [None, None, None]:
            pass
        else:
            raise Exception('Invalid update choice')

    def cost_prediction_for_architecture_change(self, architecture_change_parameters, print_check=False):
        self.architecture_update_active_set_from_choices(architecture_change_parameters)
        self.architecture_update_all_from_active_set()
        self.cost_prediction_wrapper()
        if print_check:
            print('\nChoice:', architecture_change_parameters)
            self.architecture_display_active_set()
            self.cost_display_stage_components()
        return_val = [self.trajectory.cost.predicted[-1]] + architecture_change_parameters
        return return_val

    def cost_prediction_over_possible_architecture_changes(self, architecture_change_parameters):
        S_temp = dc(self)
        return S_temp.cost_prediction_for_architecture_change(architecture_change_parameters)

    def visualizer_network_matrix(self):
        A_mat = self.A.A_mat > 0
        B_mat = self.B.active_matrix
        C_mat = self.C.active_matrix

        self.sim.network_matrix = np.block([[A_mat, B_mat, C_mat.T],
                                            [B_mat.T, np.zeros((len(self.B.active_set), len(self.B.active_set))), np.zeros((len(self.B.active_set), len(self.C.active_set)))],
                                            [C_mat, np.zeros((len(self.C.active_set), len(self.B.active_set))), np.zeros((len(self.C.active_set), len(self.C.active_set)))]])


def coin_toss():
    return np.random.default_rng().random() > 0


def compare_lists(list1, list2):
    return {'only1': [k for k in list1 if k not in list2], 'only2': [k for k in list2 if k not in list1], 'both': [k for k in list1 if k in list2]}


def cost_eval_mapper(S, choices, multiprocess_check=True):
    if multiprocess_check:
        with Pool(processes=os.cpu_count()-4) as P:
            prediction_costs = list(P.map(S.cost_prediction_over_possible_architecture_changes, choices))
    else:
        prediction_costs = list(map(S.cost_prediction_over_possible_architecture_changes, choices))
    return prediction_costs


def greedy_selection(S, number_of_changes_limit=None, multiprocess_check=True, print_check=False, t_start=time.time()):
    if not isinstance(S, System):
        raise ClassError
    work_sys = dc(S)
    if print_check:
        print('Initial architecture')
        work_sys.architecture_display_active_set()
    number_of_changes = 0
    number_of_choices = 0
    selection_check = True
    while selection_check:
        work_iteration = dc(work_sys)
        choices = work_iteration.available_selection_choices()
        if print_check:
            print('Choices:', choices)
        if len(choices) == 0 or (len(choices) == 1 and choices[0] == [None, None, None]):
            selection_check = False
            if print_check:
                print('Selection exit 1: No available selections')
        else:
            number_of_choices += len(choices)
            iterations = cost_eval_mapper(work_iteration, choices, multiprocess_check=multiprocess_check)
            iteration_costs = np.array([k[0] for k in iterations])
            if print_check:
                print('Iterations:', iterations)
                print('Iteration costs:', iteration_costs)
            idx = np.argmin(iteration_costs)
            if print_check:
                print('Min change:', iterations[idx])
            if iterations[idx][1:] == [None, None, None]:
                selection_check = False
                if print_check:
                    print('Selection exit 2: No valuable selections')
            else:
                number_of_changes += 1
                work_sys.cost_prediction_for_architecture_change(iterations[idx][1:])
                if print_check:
                    print('Cost improvement with architecture change:', work_sys.trajectory.cost.predicted)

                if print_check:
                    work_sys.architecture_display_active_set()
                if number_of_changes_limit is not None and number_of_changes >= number_of_changes_limit:
                    selection_check = False
                    if print_check:
                        print('Selection exit 3: Maximum selections done')
    return_sys = dc(S)
    return_sys.architecture_duplicate_active_set_from_system(work_sys)
    return_sys.architecture_update_all_from_active_set()
    return_sys.cost_prediction_wrapper()
    return_sys.trajectory.computation_time.append(time.time() - t_start)
    return_sys.trajectory.number_of_choices.append(number_of_choices)
    if print_check:
        print('Final architecture')
        return_sys.architecture_display_active_set()
        print('Computation time: ', return_sys.trajectory.computation_time[-1])
    return return_sys


def greedy_rejection(S, number_of_changes_limit=None, multiprocess_check=True, print_check=False, t_start=time.time()):
    if not isinstance(S, System):
        raise ClassError
    work_sys = dc(S)
    if print_check:
        print('Initial architecture')
        work_sys.architecture_display_active_set()
    number_of_changes = 0
    number_of_choices = 0
    rejection_check = True
    while rejection_check:
        work_iteration = dc(work_sys)
        choices = work_iteration.available_rejection_choices()
        if print_check:
            print('Choices:', choices)
        if len(choices) == 0 or (len(choices) == 1 and choices[0] == [None, None, None]):
            rejection_check = False
            if print_check:
                print('Rejection exit 1: No available rejections')
        else:
            number_of_choices += len(choices)
            iterations = cost_eval_mapper(work_iteration, choices, multiprocess_check=multiprocess_check)
            iteration_costs = np.array([k[0] for k in iterations])
            if print_check:
                print('Iterations:', iterations)
                print('Iteration costs:', iteration_costs)
            idx = np.argmin(iteration_costs)
            if print_check:
                print('Min change:', iterations[idx])
            if iterations[idx][1:] == [None, None, None]:
                rejection_check = False
                if print_check:
                    print('Rejection exit 2: No valuable rejections ')
            else:
                number_of_changes += 1
                work_sys.cost_prediction_for_architecture_change(iterations[idx][1:])
                if print_check:
                    print('Cost improvement with architecture change:', work_sys.trajectory.cost.predicted)

                if print_check:
                    work_sys.architecture_display_active_set()
                if number_of_changes_limit is not None and number_of_changes >= number_of_changes_limit:
                    rejection_check = False
                    if print_check:
                        print('Selection exit 3: Maximum rejections done')
    return_sys = dc(S)
    return_sys.architecture_duplicate_active_set_from_system(work_sys)
    return_sys.architecture_update_all_from_active_set()
    return_sys.cost_prediction_wrapper()
    return_sys.trajectory.computation_time.append(time.time() - t_start)
    return_sys.trajectory.number_of_choices.append(number_of_choices)
    if print_check:
        print('Final architecture')
        return_sys.architecture_display_active_set()
        print('Computation time: ', return_sys.trajectory.computation_time[-1])
    return return_sys


def greedy_simultaneous(S, number_of_changes_limit=None, number_of_changes_per_iteration=1, multiprocess_check=True, print_check=False, print_check_inner=False, t_start=time.time(), swap_only=True):
    if not isinstance(S, System):
        raise ClassError
    work_sys = dc(S)
    if print_check:
        print('Initial architecture')
        work_sys.architecture_display_active_set()
    number_of_changes = 0
    number_of_choices = 0
    simultaneous_check = True
    while simultaneous_check:
        if print_check:
            print('\n     ITERATION:', number_of_changes+1)
            print('Current starting Architecture:')
            work_sys.architecture_display_active_set()
        work_iteration = dc(work_sys)
        cost_iteration = []

        # Swap
        if print_check:
            print('\nSwap')
        swap_force_select = dc(work_iteration)
        swap_limit_mod = 1 if number_of_changes_per_iteration is None else number_of_changes_per_iteration
        swap_force_select.architecture_limit_mod(min_mod=swap_limit_mod, max_mod=swap_limit_mod)
        swap_selection_result = greedy_selection(swap_force_select, number_of_changes_limit=2*swap_limit_mod, multiprocess_check=multiprocess_check, print_check=print_check_inner)
        number_of_choices += swap_selection_result.trajectory.number_of_choices[-1]
        if print_check:
            print(' Swap force select')
            swap_selection_result.architecture_display_active_set()
            swap_selection_result.cost_display_stage_components()
        swap_force_reject = swap_selection_result
        swap_force_reject.architecture_limit_mod(min_mod=-swap_limit_mod, max_mod=-swap_limit_mod)
        swap_result = greedy_rejection(swap_force_reject, number_of_changes_limit=2*swap_limit_mod, multiprocess_check=multiprocess_check, print_check=print_check_inner)
        cost_iteration.append(swap_result.trajectory.cost.predicted[-1])
        number_of_choices += swap_result.trajectory.number_of_choices[-1]
        if print_check:
            print(' Swap force reject')
            swap_result.architecture_display_active_set()
            swap_result.cost_display_stage_components()

        if not swap_only:
            # Selection
            selection_result = greedy_selection(work_iteration, number_of_changes_limit=number_of_changes_per_iteration,
                                                multiprocess_check=multiprocess_check, print_check=print_check_inner)
            cost_iteration.append(selection_result.trajectory.cost.predicted[-1])
            number_of_choices += selection_result.trajectory.number_of_choices[-1]
            if print_check:
                print('\nSelection')
                selection_result.architecture_display_active_set()
                selection_result.cost_display_stage_components()

            # Rejection
            rejection_result = greedy_rejection(work_iteration, number_of_changes_limit=number_of_changes_per_iteration,
                                                multiprocess_check=multiprocess_check, print_check=print_check_inner)
            cost_iteration.append(rejection_result.trajectory.cost.predicted[-1])
            number_of_choices += rejection_result.trajectory.number_of_choices[-1]
            if print_check:
                print('\nRejection')
                rejection_result.architecture_display_active_set()
                rejection_result.cost_display_stage_components()

        else:
            selection_result = None
            rejection_result = None

        if print_check:
            print('\nCosts at iteration:', cost_iteration)
        idx = np.argmin(np.array(cost_iteration))
        if idx == 0:
            simultaneous_check, work_sys = arch_update_or_terminate(work_sys, swap_result)
        elif idx == 1 and not swap_only:
            simultaneous_check, work_sys = arch_update_or_terminate(work_sys, selection_result)
        elif idx == 2 and not swap_only:
            simultaneous_check, work_sys = arch_update_or_terminate(work_sys, rejection_result)
        else:
            raise Exception('Check index of min cost or swap parameters')
        if simultaneous_check:
            number_of_changes += 1
            if number_of_changes_limit is not None and number_of_changes >= number_of_changes_limit:
                simultaneous_check = False
                if print_check:
                    print('Simultaneous exit 1: Maximum changes done')
        else:
            if print_check:
                print('Simultaneous exit 2: No more valuable changes')

    return_sys = dc(S)
    return_sys.architecture_duplicate_active_set_from_system(work_sys)
    return_sys.architecture_update_all_from_active_set()
    return_sys.cost_prediction_wrapper()
    return_sys.trajectory.computation_time.append(time.time() - t_start)
    return_sys.trajectory.number_of_choices.append(number_of_choices)
    if print_check:
        print('Final architecture')
        return_sys.architecture_display_active_set()
        print('Computation time: ', return_sys.trajectory.computation_time[-1])
    return return_sys


def arch_update_or_terminate(S, S_ref):
    update_check = True
    if S.architecture_compare_active_set_to_system(S_ref):
        update_check = False
    else:
        S.architecture_duplicate_active_set_from_system(S_ref)
        S.architecture_update_all_from_active_set()
        S.cost_prediction_wrapper()
    return update_check, S
