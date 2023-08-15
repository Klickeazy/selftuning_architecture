import numpy as np
import networkx as netx

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import time
from copy import deepcopy as dc
import pandas as pd
import shelve
import itertools
import multiprocessing
# from multiprocessing import Pool
import concurrent.futures

import os
import socket

from tqdm import tqdm

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

# Save folder for data dump
if socket.gethostname() == 'melap257805':
    datadump_folder_path = 'C:/Users/kxg161630/Box/KarthikGanapathy_Research/SpeedyGreedyAlgorithm/DataDump/'
else:
    datadump_folder_path = 'D:/Box/KarthikGanapathy_Research/SpeedyGreedyAlgorithm/DataDump/'

# Save folder for images within the same git folder
image_save_folder_path = 'Images/'


# Error handling
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
    """
    Class to manage experiment parameters - save/load from csv file
    """
    def __init__(self):
        self.save_filename = "experiment_parameters.csv"            # File name for experiment parameters

        self.default_parameter_datatype_map = {'experiment_no'              : int(1),
                                               'test_model'                 : str(None),
                                               'test_parameter'             : int(0),
                                               'number_of_nodes'            : int(20),
                                               'network_model'              : str('rand'),
                                               'network_parameter'          : float(0),
                                               'rho'                        : float(1),
                                               'second_order'               : False,
                                               'second_order_network'       : int(0),
                                               'initial_architecture_size'  : int(5),
                                               'architecture_constraint_min': int(5),
                                               'architecture_constraint_max': int(5),
                                               'second_order_architecture'  : int(0),
                                               'Q_cost_scaling'             : float(1),
                                               'R_cost_scaling'             : float(1),
                                               'B_run_cost'                 : float(1),
                                               'C_run_cost'                 : float(1),
                                               'B_switch_cost'              : float(1),
                                               'C_switch_cost'              : float(1),
                                               'W_scaling'                  : float(1),
                                               'V_scaling'                  : float(1),
                                               'disturbance_model'          : str(None),
                                               'disturbance_step'           : int(0),
                                               'disturbance_number'         : int(0),
                                               'disturbance_magnitude'      : int(0),
                                               'prediction_time_horizon'    : int(10),
                                               'X0_scaling'                 : float(1),
                                               'multiprocessing'            : False}                 # Dictionary of parameter names and default value with data-type

        self.parameter_keys = list(self.default_parameter_datatype_map.keys())      # Strip parameter names from dict
        self.parameter_datatypes = {k: type(self.default_parameter_datatype_map[k]) for k in self.default_parameter_datatype_map}      # Strip parameter data types from dict

        self.parameter_table = pd.DataFrame()       # Parameter table from csv
        self.experiments_list = []                  # List of experiments
        self.parameter_values = []                  # Parameters for target experiment
        self.read_table_from_file()

    def initialize_table(self):
        # Initialize parameter csv file from nothing
        print('Initializing table with default parameters')
        self.parameter_values = [[k] for k in self.default_parameter_datatype_map.values()]
        self.parameter_table = pd.DataFrame(dict(zip(self.parameter_keys, self.parameter_values)))
        self.parameter_table.set_index(self.parameter_keys[0], inplace=True)
        self.write_table_to_file()

    def check_dimensions(self, print_check=False):
        # Ensure dimensions match
        if len(self.parameter_values) == len(self.parameter_datatypes) == len(self.parameter_keys):
            if print_check:
                print('Dimensions agree: {} elements'.format(len(self.parameter_keys)))
        else:
            raise Exception("Dimension mismatch - values: {}, datatype: {}, keys: {}".format(len(self.parameter_values), len(self.parameter_datatypes), len(self.parameter_keys)))

    def read_table_from_file(self):
        # Read table from file
        if not os.path.exists(self.save_filename):
            raise Warning('File does not exist')
        else:
            self.parameter_table = pd.read_csv(self.save_filename, index_col=0, dtype=self.parameter_datatypes)
            self.parameter_table.replace({np.nan: None}, inplace=True)
            self.experiments_list = self.parameter_table.index

    def read_parameters_from_table(self, experiment_no=1):
        if experiment_no not in self.experiments_list:
            raise Exception('Experiment parameters not in table')
        if not isinstance(self.parameter_table, pd.DataFrame):
            raise Exception('Not a pandas frame')

        self.parameter_values = [experiment_no] + [k for k in self.parameter_table.loc[experiment_no]]

    def parameter_value_map(self):
        self.parameter_values = [list(map(d, [v]))[0] if v is not None else None for d, v in zip(self.parameter_datatypes, self.parameter_values)]

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
            self.parameter_table.loc[max(self.experiments_list) + 1] = self.parameter_values[1:]
            self.experiments_list = self.parameter_table.index
            self.write_table_to_file()

    def write_table_to_file(self):
        self.parameter_table.to_csv(self.save_filename)
        print('Printing done')

    def return_keys_values(self):
        return self.parameter_values, self.parameter_keys

    def display_test_parameters(self):
        print(self.parameter_table)


def coin_toss():
    # Generate True/False with equal probability
    return np.random.default_rng().random() > 0.5


def compare_lists(list1: list, list2: list):
    return {'only1': [k for k in list1 if k not in list2], 'only2': [k for k in list2 if k not in list1], 'both': [k for k in list1 if k in list2]}


def architecture_iterator(arch=None):
    if type(arch) == list and len(arch) == 1:
        arch = arch[0]
    arch = [arch] if arch in ['B', 'C'] else ['B', 'C'] if (arch is None or arch == ['B', 'C']) else [] if arch == 'skip' else 'Error'
    if arch == 'Error':
        raise ArchitectureError
    return arch


def initialize_system_from_experiment_number(exp_no: int = 1):
    exp = Experiment()
    exp.read_parameters_from_table(exp_no)
    S = System()
    S.initialize_system_from_experiment_parameters(exp.parameter_values, exp.parameter_keys)
    return S


class PlotParameters:
    def __init__(self, sys_stage: int = 0):
        self.plot_system = sys_stage
        self.predicted_cost, self.true_cost = [], []
        self.x_2norm, self.x_estimate_2norm, self.error_2norm = [], [], []
        self.network_state_graph, self.network_state_locations = netx.Graph(), {}
        self.network_architecture_graph, self.network_architecture_locations = netx.Graph(), {}
        self.plot_parameters = \
            {1: {'node': 'tab:blue', 'B': 'tab:orange', 'C': 'tab:green', 'm': 'o', 'c': 'tab:blue', 'ms': 20, 'ls': 'solid'},
             2: {'node': 'tab:blue', 'B': 'tab:orange', 'C': 'tab:green', 'm': 'x', 'c': 'tab:orange', 'ms': 20, 'ls': 'dashed'}}
        self.network_plot_limits = []
        self.B_history = [[], []]
        self.C_history = [[], []]


class System:
    class Dynamics:
        def __init__(self):
            # Parameters assigned from function file
            self.rho = 1  # Scaling factor for eigenvalues
            self.network_model = 'rand'  # Network adjacency matrix
            self.network_parameter = 0  # Parameter for network adjacency matrix
            self.second_order = False  # Check for second order states - each node has 2 states associated with it
            self.second_order_scaling_factor = 1  # Scaling factor for second order equation
            self.second_order_network_type = 1  # Network type of second order states

            # Default parameter
            self.self_loop = True  # Self-loops within the adjacency matrix

            # Evaluated
            self.open_loop_eig_vals = 0  # Vector of second order states
            self.open_loop_eig_vecs = np.zeros((0, 0))
            self.adjacency_matrix = 0  # Adjacency matrix
            self.number_of_non_stable_modes = 0  # Number of unstable modes with magnitude >= 1
            self.A_mat = np.zeros((0, 0))  # Open-loop dynamics matrix
            self.A_enhanced_mat = np.zeros((0, 0))  # Open-loop enhanced dynamics matrix for current t

        def display_values(self):
            for var, value in vars(self).items():
                print(f"{var} = {value}")

    class Architecture:
        def __init__(self):
            # Parameters assigned from function file
            self.second_order_architecture_type = 0  # Type of second-order architecture - which states do it control or estimate
            self.min = 1  # Minimum architecture bounds
            self.max = 1  # Maximum architecture bounds
            self.R2 = 0  # Cost on running architecture
            self.R3 = 0  # Cost on switching architecture

            # Calculated terms
            self.available_indices = []  # Indices of available architecture
            self.available_vectors = {}  # Vectors associated with available architecture
            self.number_of_available = 0  # Count of available architecture
            self.active_set = []  # Set of indices of active architecture
            self.active_matrix = np.zeros((0, 0))  # Matrix of active architecture
            self.Q = np.zeros((0, 0))  # Cost on states/Process noise covariance
            self.R1 = np.zeros((0, 0))  # Cost on active actuators/Measurement noise covariance
            self.R1_reference = np.zeros((0, 0))  # Cost on available actuators
            self.indicator_vector_current = np.zeros(self.number_of_available)  # {1, 0} binary vector of currently active architecture - to compute running/switching costs
            self.indicator_vector_history = np.zeros(self.number_of_available)  # {1, 0} binary vector of previously active architecture - to compute switching costs
            self.history_active_set = {}  # Record of active architecture over simulation horizon
            self.change_count = 0  # Count of number of changes in architecture over simulation horizon
            self.recursion_matrix = {}  # Recursive cost matrix/estimation error covariance over the prediction horizon
            self.gain = {}  # Gains calculated over the prediction horizon for the fixed architecture

        def display_values(self):
            for var, value in vars(self).items():
                print(f"{var} = {value}")

    class Disturbance:
        def __init__(self):
            # Parameters assigned from function file
            self.W_scaling = 1  # Scaling factor for process noise covariance
            self.V_scaling = 1  # Scaling factor for measurement noise covariance
            self.noise_model = None  # Noise model for targeted un-modelled disturbances
            self.disturbance_step = 0  # Number of steps between un-modelled disturbances
            self.disturbance_number = 0  # Number of states affected by un-modelled disturbances
            self.disturbance_magnitude = 0  # Scaling factor of un-modelled disturbances

            # Calculated terms
            self.W = np.zeros((0, 0))  # Process noise covariance
            self.V = np.zeros((0, 0))  # Measurement noise covariance
            self.w_gen = {}  # Realization of process noise
            self.v_gen = {}  # Realization of measurement noise
            self.F_enhanced = np.zeros((0, 0))  # Enhanced matrix for noise at current time

        def display_values(self):
            for var, value in vars(self).items():
                print(f"{var} = {value}")

    class Simulation:
        def __init__(self):
            # Parameters assigned from function file
            self.experiment_number = 0  # Experiment number based on parameter sheet
            self.t_predict = int(10)  # Prediction time horizon
            self.sim_model = None  # Simulation model of architecture
            self.test_model = None  # Test case
            self.test_parameter = None  # Test case
            self.multiprocess_check = False  # Boolean to determine if multiprocess mapping is used or not for choices

            # Constant parameters
            self.t_simulate = int(100)  # Simulation time horizon
            self.t_current = 0  # Current time-step of simulation

        def display_values(self):
            for var, value in vars(self).items():
                print(f"{var} = {value}")

    class Trajectory:
        def __init__(self):
            # Parameters assigned from function file
            self.X0_scaling = 1  # Scaling factor of initial state

            # Calculated terms
            self.X0_covariance = np.zeros((0, 0))  # Initial state covariance
            self.x = {}  # True state trajectory
            self.x_estimate = {}  # Estimates state trajectory
            self.X_enhanced = {}  # Enhanced state trajectory
            self.error = {}  # Estimation error trajectory
            self.control_cost_matrix = {}  # Control cost matrix at each timestep
            self.estimation_matrix = {}  # Estimation error covariance matrix at each timestep
            self.error_2norm = {}  # 2-norm of estimation error trajectory
            self.cost = System.Cost()  # Cost variables
            self.computation_time = {}  # Computation time for greedy optimization at each simulation timestep

        def display_values(self):
            for var, value in vars(self).items():
                print(f"{var} = {value}")

    class Cost:
        def __init__(self):
            # Parameters assigned from function file
            self.metric_control = 1  # Metric function to evaluate control costs
            self.metric_running = 1  # Metric function to evaluate running costs
            self.metric_switching = 1  # Metric function to evaluate switching costs

            # Calculated terms
            self.running = 0  # Running cost at current timestep
            self.switching = 0  # Switching cost at current timestep
            self.control = 0  # Control cost at current timestep
            self.predicted = {}  # Predicted total stage cost trajectory
            self.true = {}  # True total stage cost trajectory
            self.initial = []  # Costs for initial architecture optimization
            self.predicted_matrix = {}  # Cost matrix over the prediction horizon

        def display_values(self):
            for var, value in vars(self).items():
                print(f"{var} = {value}")

    def __init__(self):
        self.number_of_nodes = 20  # Number of nodes in the network
        self.number_of_states = 20  # Number of state in the network (affected by second_order dynamics)
        self.A = self.Dynamics()
        self.B = self.Architecture()
        self.C = self.Architecture()
        self.disturbance = self.Disturbance()
        self.sim = self.Simulation()
        self.trajectory = self.Trajectory()
        self.plot = None

        self.model_name = ''
        self.plot_name = None

    def initialize_system_from_experiment_parameters(self, experiment_parameters, experiment_keys):

        parameters = dict(zip(experiment_keys, experiment_parameters))

        self.sim.experiment_number = parameters['experiment_no']
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
        self.sim.multiprocess_check = parameters['multiprocessing']

        self.sim.test_model = None if parameters['test_model'] == 'None' else parameters['test_model']
        self.sim.test_parameter = None if parameters['test_parameter'] == 0 else int(parameters['test_parameter'])

        parameters['disturbance_model'] = None if parameters['disturbance_model'] == 'None' else parameters['disturbance_model']

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

        self.B.Q = np.identity(self.number_of_states) * parameters['Q_cost_scaling']
        self.B.R1_reference = np.identity(self.B.number_of_available) * parameters['R_cost_scaling']
        self.C.Q = self.disturbance.W
        self.C.R1_reference = self.disturbance.V

        self.initialize_random_architecture_active_set(parameters['initial_architecture_size'])
        self.architecture_update_to_history_indicator_matrix_from_active_set()

        self.trajectory.X0_scaling = parameters['X0_scaling']
        self.initialize_trajectory()
        self.prediction_gains()
        self.cost_prediction_wrapper()

        self.model_namer()

    def model_namer(self, namer_type=1, name_extension: str = None):
        if namer_type == 1:
            self.model_name = 'model_exp' + str(self.sim.experiment_number)
        elif namer_type == 2:
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
            if self.sim.sim_model is not None:
                self.model_name = self.model_name + '_' + self.sim.sim_model
            if self.sim.test_model is not None:
                self.model_name = self.model_name + '_' + self.sim.test_model
        else:
            raise Exception('Not Implemented')
        if name_extension is not None:
            self.model_name = self.model_name + '_' + name_extension

    def display_all_parameters(self):
        print('Model name = ', self.model_name)
        print('Number of Nodes = ', self.number_of_nodes)
        print('Number of States = ', self.number_of_states)
        print('A')
        self.A.display_values()
        print('B')
        self.B.display_values()
        print('C')
        self.C.display_values()
        print('disturbances')
        self.disturbance.display_values()
        print('sim')
        self.sim.display_values()
        self.trajectory.display_values()

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
                    e = [1 + k if coin_toss() else 1 - k for k in e]
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

    def evaluate_modes(self):
        self.A.open_loop_eig_vals = np.sort(np.abs(np.linalg.eigvals(self.A.A_mat)))
        _, self.A.open_loop_eig_vecs = np.linalg.eig(self.A.A_mat)
        for i in range(0, self.number_of_states):
            if np.linalg.norm(self.A.open_loop_eig_vecs[:, i]) != 0:
                self.A.open_loop_eig_vecs[:, i] /= np.linalg.norm(self.A.open_loop_eig_vecs[:, i])
        self.A.number_of_non_stable_modes = len([e for e in self.A.open_loop_eig_vals if e >= 1])

    def rescale(self):
        if self.A.rho is not None:
            self.A.A_mat = self.A.rho * self.A.adjacency_matrix / np.max(np.abs(np.linalg.eigvals(self.A.adjacency_matrix)))
        else:
            self.A.A_mat = self.A.adjacency_matrix

    def second_order_matrix(self):
        if self.A.second_order:
            if self.A.second_order_network_type == 1:
                self.A.A_mat = np.block([
                    [self.A.A_mat, np.zeros_like(self.A.A_mat)],
                    [self.A.second_order_scaling_factor * np.identity(self.number_of_nodes), self.A.second_order_scaling_factor * np.identity(self.number_of_nodes)]])
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

    def initialize_active_matrix(self, arch=None):
        arch = architecture_iterator(arch)
        for a in arch:
            if a == 'B':
                self.B.active_matrix = np.zeros((self.number_of_states, len(self.B.active_set)))
            else:  # self.architecture_type == 'C':
                self.C.active_matrix = np.zeros((len(self.C.active_set), self.number_of_states))

    def initialize_available_vectors_as_basis_vectors(self, arch=None):
        arch = architecture_iterator(arch)

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
                self.C.available_vectors = {i: set_mat[i, :] for i in range(0, self.number_of_nodes)}
                self.C.available_indices = [i for i in self.C.available_vectors]
                self.C.number_of_available = len(self.B.available_indices)
                self.C.indicator_vector_history = np.zeros(self.C.number_of_available, dtype=int)
                self.C.indicator_vector_current = np.zeros(self.C.number_of_available, dtype=int)

    def initialize_random_architecture_active_set(self, initialize_random: int, arch=None):
        arch = architecture_iterator(arch)
        for a in arch:
            if a == 'B':
                self.B.active_set = list(np.sort(np.random.default_rng().choice(self.B.available_indices, size=initialize_random, replace=False)))
            else:  # a == 'C'
                self.C.active_set = list(np.sort(np.random.default_rng().choice(self.C.available_indices, size=initialize_random, replace=False)))

    def initialize_disturbance(self):
        self.disturbance.W = np.identity(self.number_of_states) * self.disturbance.W_scaling
        self.disturbance.V = np.identity(self.number_of_states) * self.disturbance.V_scaling

        # self.disturbance.w_gen = np.random.default_rng().multivariate_normal(np.zeros(self.number_of_states), self.disturbance.W, self.sim.t_simulate)
        # self.disturbance.v_gen = np.random.default_rng().multivariate_normal(np.zeros(self.number_of_nodes), self.disturbance.V, self.sim.t_simulate)

        self.disturbance.w_gen = {t: np.random.default_rng().multivariate_normal(np.zeros(self.number_of_states), self.disturbance.W) for t in range(0, self.sim.t_simulate)}
        self.disturbance.v_gen = {t: np.random.default_rng().multivariate_normal(np.zeros(self.number_of_states), self.disturbance.V) for t in range(0, self.sim.t_simulate)}

        if self.disturbance.noise_model in ['process', 'measurement', 'combined']:
            if self.disturbance.disturbance_number == 0 or self.disturbance.disturbance_magnitude == 0 or self.disturbance.disturbance_step == 0:
                raise Exception('Check disturbance parameters')
            for t in range(0, self.sim.t_simulate, self.disturbance.disturbance_step):
                if self.disturbance.noise_model in ['process', 'combined']:
                    self.disturbance.w_gen[t][np.random.default_rng().choice(self.number_of_states, self.disturbance.disturbance_number, replace=False)] = self.disturbance.disturbance_magnitude * np.array([coin_toss() for _ in range(0, self.disturbance.disturbance_number)])
                if self.disturbance.noise_model in ['measurement', 'combined']:
                    self.disturbance.v_gen[t] = self.disturbance.disturbance_magnitude * np.array([coin_toss() for _ in range(0, self.number_of_states)])

    def initialize_trajectory(self, x0_idx=None):
        self.trajectory.X0_covariance = np.identity(self.number_of_states) * self.trajectory.X0_scaling

        if x0_idx is None:
            self.trajectory.x = {0: np.random.default_rng().multivariate_normal(np.zeros(self.number_of_states), self.trajectory.X0_covariance)}
        else:
            self.trajectory.x = {0: self.A.open_loop_eig_vecs[:, x0_idx]}

        self.trajectory.x_estimate = {0: np.random.default_rng().multivariate_normal(np.zeros(self.number_of_states), self.trajectory.X0_covariance)}

        self.trajectory.X_enhanced = {0: np.concatenate((self.trajectory.x[0], self.trajectory.x_estimate[0]))}

        self.trajectory.control_cost_matrix = {}
        self.trajectory.estimation_matrix = {0: np.identity(self.number_of_states)}

        self.trajectory.error = {0: self.trajectory.x[0] - self.trajectory.x_estimate[0]}
        self.trajectory.error_2norm = {0: np.linalg.norm(self.trajectory.error[0])}

    def architecture_limit_set(self, arch=None, min_set: int = None, max_set: int = None):
        arch = architecture_iterator(arch)
        for a in arch:
            if a == 'B':
                self.B.min = self.B.number_of_available if self.B.min is None else min_set if min_set is not None else self.B.min
                self.B.max = self.B.number_of_available if self.B.max is None else max_set if max_set is not None else self.B.max
            else:  # a == 'C'
                self.C.min = self.C.number_of_available if self.C.min is None else min_set if min_set is not None else self.C.min
                self.C.max = self.C.number_of_available if self.C.max is None else max_set if max_set is not None else self.C.max

    def architecture_limit_mod(self, arch=None, min_mod: int = None, max_mod: int = None):
        arch = architecture_iterator(arch)
        for a in arch:
            if a == 'B':
                self.B.min += (min_mod if min_mod is not None else 0)
                self.B.max += (max_mod if max_mod is not None else 0)
            else:  # a == 'C'
                self.C.min += (min_mod if min_mod is not None else 0)
                self.C.max += (max_mod if max_mod is not None else 0)

    def architecture_update_to_matrix_from_active_set(self, arch=None):
        arch = architecture_iterator(arch)
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

    def architecture_update_to_indicator_from_active_set(self, arch=None):
        arch = architecture_iterator(arch)
        for a in arch:
            if a == 'B':
                self.B.indicator_vector_history = np.zeros(self.B.number_of_available, dtype=int)
                self.B.indicator_vector_current = np.zeros(self.B.number_of_available, dtype=int)
                self.B.indicator_vector_current[self.B.history_active_set[self.sim.t_current]] = 1
                if self.sim.t_current >= 1:
                    self.B.indicator_vector_history[self.B.history_active_set[self.sim.t_current-1]] = 1
                else:
                    self.B.indicator_vector_history = dc(self.B.indicator_vector_current)

            else:  # a == 'C'
                self.C.indicator_vector_history = np.zeros(self.C.number_of_available, dtype=int)
                self.C.indicator_vector_current = np.zeros(self.C.number_of_available, dtype=int)
                self.C.indicator_vector_current[self.C.history_active_set[self.sim.t_current]] = 1
                if self.sim.t_current >= 1:
                    self.C.indicator_vector_history[self.C.history_active_set[self.sim.t_current - 1]] = 1
                else:
                    self.C.indicator_vector_history = dc(self.C.indicator_vector_current)

    def architecture_update_to_history_from_active_set(self, arch=None):
        arch = architecture_iterator(arch)
        for a in arch:
            if a == 'B':
                self.B.history_active_set[self.sim.t_current] = self.B.active_set
            else:  # a == 'C'
                self.C.history_active_set[self.sim.t_current] = self.C.active_set

    def architecture_update_to_history_indicator_matrix_from_active_set(self, arch=None):
        self.architecture_update_to_history_from_active_set(arch)
        self.architecture_update_to_matrix_from_active_set(arch)
        self.architecture_update_to_indicator_from_active_set()

    def architecture_duplicate_active_set_from_system(self, reference_system, update_check=True):
        if not isinstance(reference_system, System):
            raise ClassError
        for a in ['B', 'C']:
            if a == 'B':
                self.B.active_set = dc(reference_system.B.active_set)
            else:
                self.C.active_set = dc(reference_system.C.active_set)
        if update_check:
            self.architecture_update_to_history_indicator_matrix_from_active_set()

    def architecture_compare_active_set_to_system(self, reference_system):
        if not isinstance(reference_system, System):
            raise ClassError
        return set(self.B.active_set) == set(reference_system.B.active_set) and set(self.C.active_set) == set(reference_system.C.active_set)

    def architecture_compute_active_set_changes(self, reference_system):
        if not isinstance(reference_system, System):
            raise ClassError
        B_compare = compare_lists(self.B.active_set, reference_system.B.active_set)
        C_compare = compare_lists(self.C.active_set, reference_system.C.active_set)
        number_of_changes = max(len(B_compare['only2']), len(B_compare['only2'])) + max(len(C_compare['only2']), len(C_compare['only2']))
        return number_of_changes

    def architecture_count_number_of_sim_changes(self):
        self.B.change_count, self.C.change_count = 0, 0
        if self.sim.sim_model == "selftuning":
            for t in range(1, self.sim.t_simulate):
                compare_B = compare_lists(self.B.history_active_set[t-1], self.B.history_active_set[t])
                compare_C = compare_lists(self.C.history_active_set[t-1], self.C.history_active_set[t])
                self.B.change_count += max(len(compare_B['only2']), len(compare_B['only1']))
                self.C.change_count += max(len(compare_C['only2']), len(compare_C['only1']))

    def architecture_display(self):
        print('B: ', self.B.active_set)
        print('C: ', self.C.active_set)

    def cost_architecture_running(self):
        if self.trajectory.cost.metric_running == 0 or (self.B.R2 == 0 and self.C.R2 == 0):
            self.trajectory.cost.running = 0
        elif self.trajectory.cost.metric_running == 1 or (type(self.B.R2) == int and type(self.C.R2) == int):
            self.trajectory.cost.running = np.linalg.norm(self.B.indicator_vector_current, ord=0) * self.B.R2 + np.linalg.norm(self.C.indicator_vector_current, ord=0) * self.C.R2
        elif self.trajectory.cost.metric_running == 2 or (np.shape(self.B.R2) == (len(self.B.active_set), len(self.B.active_set)) and np.shape(self.C.R2) == (len(self.C.active_set), len(self.C.active_set))):
            self.trajectory.cost.running = self.B.indicator_vector_current.T @ self.B.R2 @ self.B.indicator_vector_current + self.C.indicator_vector_current.T @ self.C.R2 @ self.C.indicator_vector_current
        else:
            print(self.B.R2)
            raise Exception('Check running cost metric')

    def cost_architecture_switching(self):
        if self.trajectory.cost.metric_switching == 0 or (self.B.R3 == 0 and self.C.R3 == 0):
            self.trajectory.cost.switching = 0
        elif self.trajectory.cost.metric_switching == 1 or (type(self.B.R3) == int and type(self.C.R3) == int):
            self.trajectory.cost.switching = np.linalg.norm(self.B.indicator_vector_current - self.B.indicator_vector_history, ord=0) * self.B.R3 + np.linalg.norm(self.C.indicator_vector_current - self.C.indicator_vector_history, ord=0) * self.C.R3
        elif self.trajectory.cost.metric_switching == 2 or (np.shape(self.B.R3) == (len(self.B.active_set), len(self.B.active_set)) and np.shape(self.C.R3) == (len(self.C.active_set), len(self.C.active_set))):
            self.trajectory.cost.switching = (self.B.indicator_vector_current - self.B.indicator_vector_history).T @ self.B.R2 @ (self.B.indicator_vector_current - self.B.indicator_vector_history) + (self.C.indicator_vector_current - self.C.indicator_vector_history).T @ self.C.R2 @ (self.C.indicator_vector_current - self.C.indicator_vector_history)
        else:
            raise Exception('Check switching cost metric')

    def prediction_gains(self, arch=None):  # , update_trajectory_check=False):
        arch = architecture_iterator(arch)
        for a in arch:
            if a == 'B':
                # self.prediction_control_gain(update_trajectory_check=update_trajectory_check)
                self.prediction_control_gain()
            else:  # if a == 'C'
                # self.prediction_estimation_gain(update_trajectory_check=update_trajectory_check)
                self.prediction_estimation_gain()

    def prediction_control_gain(self):  # , update_trajectory_check=False):
        self.B.recursion_matrix = {self.sim.t_predict: self.B.Q}
        for t in range(self.sim.t_predict - 1, -1, -1):
            self.B.gain[t] = np.linalg.inv((self.B.active_matrix.T @ self.B.recursion_matrix[t + 1] @ self.B.active_matrix) + self.B.R1) @ self.B.active_matrix.T @ self.B.recursion_matrix[t + 1] @ self.A.A_mat
            self.B.recursion_matrix[t] = (self.A.A_mat.T @ self.B.recursion_matrix[t + 1] @ self.A.A_mat) - (self.A.A_mat.T @ self.B.recursion_matrix[t + 1] @ self.B.active_matrix @ self.B.gain[t]) + self.B.Q
        # if update_trajectory_check:
        #     self.trajectory.control_cost_matrix.append(self.B.recursion_matrix[0])

    def prediction_estimation_gain(self):  # , update_trajectory_check=False):
        self.C.recursion_matrix = {0: self.trajectory.estimation_matrix[self.sim.t_current]}
        for t in range(0, self.sim.t_predict):
            self.C.gain[t] = self.C.recursion_matrix[t] @ self.C.active_matrix.T @ np.linalg.inv((self.C.active_matrix @ self.C.recursion_matrix[t] @ self.C.active_matrix.T) + self.C.R1)
            self.C.recursion_matrix[t + 1] = (self.A.A_mat @ self.C.recursion_matrix[t] @ self.A.A_mat.T) - (self.A.A_mat @ self.C.gain[t] @ self.C.active_matrix @ self.C.recursion_matrix[t] @ self.A.A_mat.T) + self.C.Q
        # if update_trajectory_check:
        #     self.trajectory.estimation_matrix.append(self.C.recursion_matrix[1])

    def cost_prediction(self):
        A_enhanced_mat = {}
        W_enhanced = {}
        F_enhanced = {}
        Q_enhanced = {}
        W_mat = np.block([[self.disturbance.W, np.zeros((self.number_of_states, len(self.C.active_set)))],
                          [np.zeros((len(self.C.active_set), self.number_of_states)), self.disturbance.V[:, self.C.active_set][self.C.active_set, :]]])
        for t in range(0, self.sim.t_predict):
            BKt = self.B.active_matrix @ self.B.gain[t]
            ALtC = self.A.A_mat @ self.C.gain[t] @ self.C.active_matrix
            A_enhanced_mat[t] = np.block([[self.A.A_mat, -BKt],
                                          [ALtC, self.A.A_mat - ALtC - BKt]])
            F_enhanced[t] = np.block([[np.identity(self.number_of_states), np.zeros((self.number_of_states, len(self.C.active_set)))],
                                      [np.zeros((self.number_of_states, self.number_of_states)), self.A.A_mat @ self.C.gain[t]]])
            W_enhanced[t] = F_enhanced[t] @ W_mat @ F_enhanced[t].T
            Q_enhanced[t] = np.block([[self.B.Q, np.zeros((self.number_of_states, self.number_of_states))],
                                      [np.zeros((self.number_of_states, self.number_of_states)), self.B.gain[t].T @ self.B.R1 @ self.B.gain[t]]])

        Q_enhanced[self.sim.t_predict] = np.block([[self.B.Q, np.zeros((self.number_of_states, self.number_of_states))],
                                                   [np.zeros((self.number_of_states, self.number_of_states)), np.zeros((self.number_of_states, self.number_of_states))]])

        self.A.A_enhanced_mat = A_enhanced_mat[0]
        self.disturbance.F_enhanced = F_enhanced[0]

        self.trajectory.cost.predicted_matrix = {self.sim.t_predict: Q_enhanced[self.sim.t_predict]}
        self.trajectory.cost.control = 0

        for t in range(self.sim.t_predict - 1, -1, -1):
            self.trajectory.cost.control += np.trace(self.trajectory.cost.predicted_matrix[t + 1] @ W_enhanced[t])
            self.trajectory.cost.predicted_matrix[t] = A_enhanced_mat[t].T @ self.trajectory.cost.predicted_matrix[t + 1] @ A_enhanced_mat[t]

        if self.trajectory.cost.metric_control == 1:
            x_estimate_stack = np.squeeze(np.tile(self.trajectory.x_estimate[self.sim.t_current], (1, 2)))
            self.trajectory.cost.control += (x_estimate_stack.T @ self.trajectory.cost.predicted_matrix[0] @ x_estimate_stack)
        elif self.trajectory.cost.metric_control == 2:
            self.trajectory.cost.control += np.max(np.linalg.eigvals(self.trajectory.cost.predicted_matrix[0]))
        else:
            raise Exception('Check control cost metric')

    def cost_true(self):
        Q_mat = np.block([[self.B.Q, np.zeros((self.number_of_states, self.number_of_states))],
                          [np.zeros((self.number_of_states, self.number_of_states)), self.B.gain[0].T @ self.B.R1 @ self.B.gain[0]]])
        self.trajectory.cost.control = 0
        if self.trajectory.cost.metric_control == 1:
            x_estimate_stack = np.squeeze(np.tile(self.trajectory.x_estimate[self.sim.t_current], (1, 2)))
            self.trajectory.cost.control += (x_estimate_stack.T @ Q_mat @ x_estimate_stack)
        elif self.trajectory.cost.metric_control == 2:
            self.trajectory.cost.control += np.max(np.linalg.eigvals(self.trajectory.cost.predicted_matrix[0]))
        else:
            raise Exception('Check control cost metric')

    def cost_prediction_wrapper(self, evaluate_gains='skip'):
        self.prediction_gains(evaluate_gains)
        self.cost_prediction()
        self.cost_architecture_running()
        self.cost_architecture_switching()
        self.trajectory.cost.predicted[self.sim.t_current] = self.trajectory.cost.control + self.trajectory.cost.running + self.trajectory.cost.switching

    def cost_true_wrapper(self):
        self.cost_true()
        self.cost_architecture_running()
        self.cost_architecture_switching()
        self.trajectory.cost.true[self.sim.t_current] = self.trajectory.cost.control + self.trajectory.cost.running

    # choice format: {target_architecture, target_node, +/- (select/reject), resultant_system}
    def available_choices_selection(self):
        if len(self.B.active_set) >= self.B.min and len(self.C.active_set) >= self.C.min:
            # If minimum number of actuators AND sensors are active
            choices = [{'arch': 'skip', 'idx': None, 'change': None}]
        else:
            choices = []
        if len(self.B.active_set) < self.B.max:
            # If maximum number of actuators are not active
            for i in compare_lists(self.B.active_set, self.B.available_indices)['only2']:
                choices.append({'arch': 'B', 'idx': i, 'change': '+'})
        if len(self.C.active_set) < self.C.max:
            # If maximum number of sensors are not active
            for i in compare_lists(self.C.active_set, self.C.available_indices)['only2']:
                choices.append({'arch': 'C', 'idx': i, 'change': '+'})
        return choices

    def available_choices_rejection(self):
        if len(self.B.active_set) <= self.B.max and len(self.C.active_set) <= self.C.max:
            # If maximum number of actuators AND sensors are active
            choices = [{'arch': 'skip', 'idx': None, 'change': None}]
        else:
            choices = []
        if len(self.B.active_set) > self.B.min:
            # If minimum number of actuators are not active
            for i in self.B.active_set:
                choices.append({'arch': 'B', 'idx': i, 'change': '-'})
        if len(self.C.active_set) > self.C.min:
            # If minimum number of sensors are not active
            for i in self.C.active_set:
                choices.append({'arch': 'C', 'idx': i, 'change': '-'})
        return choices

    def architecture_update_active_set_from_choices(self, architecture_change_parameters):
        if architecture_change_parameters['arch'] == 'B':
            if architecture_change_parameters['change'] == '+':
                self.B.active_set.append(architecture_change_parameters['idx'])
                self.B.active_set = list(np.sort(np.array(self.B.active_set)))
            elif architecture_change_parameters['change'] == '-':
                self.B.active_set = [k for k in self.B.active_set if k != architecture_change_parameters['idx']]
        elif architecture_change_parameters['arch'] == 'C':
            if architecture_change_parameters['change'] == '+':
                self.C.active_set.append(architecture_change_parameters['idx'])
                self.C.active_set = list(np.sort(np.array(self.C.active_set)))
            elif architecture_change_parameters['change'] == '-':
                self.C.active_set = [k for k in self.C.active_set if k != architecture_change_parameters['idx']]
        elif architecture_change_parameters == {'arch': 'skip', 'idx': None, 'change': None}:
            pass
        else:
            raise Exception('Invalid update choice')

    def evaluate_cost_for_choice(self, choice_parameter):
        S_choice = dc(self)
        S_choice.architecture_update_active_set_from_choices(choice_parameter)
        S_choice.architecture_update_to_history_indicator_matrix_from_active_set()
        S_choice.cost_prediction_wrapper(choice_parameter['arch'])
        return_values = dc(choice_parameter)
        return_values.update({'system': S_choice, 'cost': S_choice.trajectory.cost.predicted[S_choice.sim.t_current]})
        return return_values

    def one_step_system_update(self):
        # print('1:', np.shape(self.disturbance.F_enhanced), np.shape(self.disturbance.w_gen[self.sim.t_current]), np.shape(self.disturbance.v_gen[self.sim.t_current]))
        # print('2:', np.shape(self.A.A_enhanced_mat), np.shape(self.trajectory.X_enhanced[self.sim.t_current]))
        self.trajectory.X_enhanced[self.sim.t_current + 1] = (self.A.A_enhanced_mat @ self.trajectory.X_enhanced[self.sim.t_current]) + (self.disturbance.F_enhanced @ np.concatenate((self.disturbance.w_gen[self.sim.t_current], self.disturbance.v_gen[self.sim.t_current][self.C.active_set])))

        self.trajectory.x[self.sim.t_current + 1] = self.trajectory.X_enhanced[self.sim.t_current + 1][0:self.number_of_states]

        self.trajectory.x_estimate[self.sim.t_current + 1] = self.trajectory.X_enhanced[self.sim.t_current + 1][self.number_of_states:]

        self.trajectory.error[self.sim.t_current + 1] = self.trajectory.x[self.sim.t_current + 1] - self.trajectory.x_estimate[self.sim.t_current + 1]

        self.trajectory.error_2norm[self.sim.t_current + 1] = np.linalg.norm(self.trajectory.error[self.sim.t_current + 1])

        self.trajectory.estimation_matrix[self.sim.t_current + 1] = self.C.recursion_matrix[1]
        self.trajectory.control_cost_matrix[self.sim.t_current] = self.B.recursion_matrix[0]

        self.sim.t_current += 1
        self.architecture_update_to_history_indicator_matrix_from_active_set()
        self.prediction_gains()
        self.cost_prediction_wrapper()

    def generate_network_graph_state_locations(self):
        S_A = dc(self)
        S_A.generate_network_architecture_graph_matrix(mB=0, mC=0)
        S_A.plot.network_architecture_graph.remove_nodes_from(list(netx.isolates(S_A.plot.network_architecture_graph)))
        self.plot.network_state_graph = dc(S_A.plot.network_architecture_graph)
        self.plot.network_state_locations = netx.circular_layout(S_A.plot.network_architecture_graph)

    def generate_network_plot_limits(self):
        S_full = dc(self)
        S_full.B.active_set = dc(S_full.B.available_indices)
        S_full.C.active_set = dc(S_full.C.available_indices)
        S_full.architecture_update_to_matrix_from_active_set()
        S_full.generate_network_architecture_graph_matrix()
        full_pos = netx.spring_layout(S_full.plot.network_architecture_graph,
                                      pos=self.plot.network_state_locations,
                                      fixed=[str(i) for i in range(1, 1 + self.number_of_states)])
        x = [full_pos[k][0] for k in full_pos]
        y = [full_pos[k][1] for k in full_pos]
        scale = 1.5
        self.plot.network_plot_limits = [[min(x)*scale, max(x)*scale], [min(y)*scale, max(y)*scale]]

    def generate_network_architecture_graph_matrix(self, mA=1, mB=1, mC=1):
        A_mat = (self.A.adjacency_matrix > 0) * mA
        B_mat = (self.B.active_matrix > 0) * mB
        C_mat = (self.C.active_matrix > 0) * mC
        net_matrix = np.block([[A_mat, B_mat, C_mat.T],
                               [B_mat.T, np.zeros((len(self.B.active_set), len(self.B.active_set))), np.zeros((len(self.B.active_set), len(self.C.active_set)))],
                               [C_mat, np.zeros((len(self.C.active_set), len(self.B.active_set))), np.zeros((len(self.C.active_set), len(self.C.active_set)))]])
        self.plot.network_architecture_graph = netx.from_numpy_array(net_matrix)

        node_label_map = {}
        for i in range(0, self.number_of_states):
            node_label_map[i] = str(i+1)
        for i in range(0, len(self.B.active_set)):
            node_label_map[self.number_of_states + i] = "B" + str(i + 1)
        for i in range(0, len(self.C.active_set)):
            node_label_map[self.number_of_states + len(self.B.active_set) + i] = "C" + str(i + 1)
        netx.relabel_nodes(self.plot.network_architecture_graph, node_label_map, copy=False)

    def plot_network_states(self, ax_in=None):
        if ax_in is None:
            fig = plt.figure()
            grid = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(grid[0, 0])
        else:
            ax = ax_in

        node_color_array = [self.plot.plot_parameters[self.plot.plot_system]['node']] * self.number_of_states
        netx.draw_networkx(self.plot.network_state_graph,
                           ax=ax, node_color=node_color_array,
                           pos=self.plot.network_state_locations)
        ax.set_xlim(self.plot.network_plot_limits[0])
        ax.set_ylim(self.plot.network_plot_limits[1])

        if ax_in is None:
            plt.show()

    def architecture_at_timestep_t(self, t: int = None):
        if t is not None:
            self.B.active_set = dc(self.B.history_active_set[t])
            self.C.active_set = dc(self.C.history_active_set[t])
            self.architecture_update_to_matrix_from_active_set()

    def generate_network_graph_architecture_locations(self, t: int = None):
        self.architecture_at_timestep_t(t)
        self.generate_network_architecture_graph_matrix()
        self.plot.network_architecture_locations = netx.spring_layout(self.plot.network_architecture_graph, pos=self.plot.network_state_locations, fixed=[str(i) for i in range(1, 1 + self.number_of_states)])

    def generate_architecture_history_points(self):
        for t in self.B.history_active_set:
            for a in self.B.history_active_set[t]:
                self.plot.B_history[0].append(t)
                self.plot.B_history[1].append(a+1)
        for t in self.C.history_active_set:
            for a in self.C.history_active_set[t]:
                self.plot.C_history[0].append(t)
                self.plot.C_history[1].append(a+1)

    def plot_network_architecture(self, t: int = None, ax_in=None):
        if ax_in is None:
            fig = plt.figure()
            grid = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(grid[0, 0])
        else:
            ax = ax_in

        self.generate_network_graph_architecture_locations(t)

        self.generate_network_architecture_graph_matrix(mA=0)
        node_color_array = [self.plot.plot_parameters[self.plot.plot_system]['B']] * len(self.B.active_set)
        node_color_array.extend([self.plot.plot_parameters[self.plot.plot_system]['C']] * len(self.C.active_set))

        architecture_list = ["B"+str(i) for i in range(1, 1+len(self.B.active_set))] + ["C"+str(i) for i in range(1, 1+len(self.C.active_set))]
        architecture_labels = {"B"+str(i): "B"+str(i) for i in range(1, 1+len(self.B.active_set))}
        architecture_labels.update({"C"+str(i): "C"+str(i) for i in range(1, 1+len(self.C.active_set))})

        netx.draw_networkx_nodes(self.plot.network_architecture_graph, ax=ax, pos=self.plot.network_architecture_locations, nodelist=architecture_list, node_color=node_color_array)

        netx.draw_networkx_labels(self.plot.network_architecture_graph, ax=ax, pos=self.plot.network_architecture_locations, labels=architecture_labels)

        netx.draw_networkx_edges(self.plot.network_architecture_graph, ax=ax, pos=self.plot.network_architecture_locations)

        if ax_in is None:
            plt.show()

    def plot_network(self, t: int = None, ax1_in=None, ax2_in=None):
        if ax1_in is None and ax2_in is None:
            fig = plt.figure()
            grid = fig.add_gridspec(1, 1)
            ax1 = fig.add_subplot(grid[0, 0], frameon=False, zorder=1.1, aspect='equal')
            ax1.tick_params(axis='both', labelbottom=False, labelleft=False, bottom=False, top=False, left=False, right=False)
            # ax1.patch.set_alpha(0.1)
            ax2 = fig.add_subplot(grid[0, 0], sharex=ax1, sharey=ax1, zorder=1.2, aspect='equal')
            ax2.tick_params(axis='both', labelbottom=False, labelleft=False, bottom=False, top=False, left=False, right=False)
            ax2.patch.set_alpha(0.1)
        else:
            ax1 = ax1_in
            ax2 = ax2_in

        self.generate_network_graph_state_locations()
        self.generate_network_plot_limits()

        self.architecture_display()
        self.plot_network_states(ax_in=ax2)
        self.plot_network_architecture(t=t, ax_in=ax1)

        ax1.set_xlim(self.plot.network_plot_limits[0])
        ax1.set_ylim(self.plot.network_plot_limits[1])

        if ax1_in is None and ax2_in is None:
            plt.show()

    def plot_architecture_history(self, arch=None, ax_in=None):
        if ax_in is None:
            fig = plt.figure()
            grid = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(grid[0, 0])
        else:
            ax = ax_in

        self.generate_architecture_history_points()
        arch = architecture_iterator(arch)
        for a in arch:
            if a == 'B':
                if self.sim.sim_model == 'selftuning':
                    ax.scatter(self.plot.B_history[0], self.plot.B_history[1], label=self.plot_name,
                               s=10, alpha=0.7,
                               marker=self.plot.plot_parameters[self.plot.plot_system]['m'],
                               c=self.plot.plot_parameters[self.plot.plot_system]['c'])
                elif self.sim.sim_model == "fixed":
                    ax.hlines([i+1 for i in self.B.history_active_set[0]], 0, self.sim.t_simulate, alpha=1,
                              colors=[self.plot.plot_parameters[self.plot.plot_system]['c']]*len(self.B.history_active_set[0]), linestyles='dashed', linewidth=1)
                else:
                    raise Exception('Invalid test model')
            else:  # a=='C'
                if self.sim.sim_model == 'selftuning':
                    ax.scatter(self.plot.C_history[0], self.plot.C_history[1], label=self.plot_name,
                               s=10, alpha=0.7,
                               marker=self.plot.plot_parameters[self.plot.plot_system]['m'],
                               c=self.plot.plot_parameters[self.plot.plot_system]['c'])
                elif self.sim.sim_model == "fixed":
                    ax.hlines([i+1 for i in self.C.history_active_set[0]], 0, self.sim.t_simulate, alpha=1,
                              colors=[self.plot.plot_parameters[self.plot.plot_system]['c']] * len(self.C.history_active_set[0]), linestyles='dashed', linewidth=1)
                else:
                    raise Exception('Invalid test model')

        ax.set_ylim([0, self.number_of_states + 2])

        if ax_in is None:
            plt.show()

    def plot_cost(self, cost_type='true', ax_in=None):
        if ax_in is None:
            fig = plt.figure()
            grid = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(grid[0, 0])
        else:
            ax = ax_in

        if cost_type == 'true':
            cost = list(itertools.accumulate(self.vector_from_dict_key_cost(self.trajectory.cost.true)))
            ls = 'solid'
        elif cost_type == 'predict':
            cost = self.vector_from_dict_key_cost(self.trajectory.cost.predicted)
            ls = 'dashed'
        else:
            raise Exception('Check argument')

        ax.plot(range(0, self.sim.t_simulate), cost, label=self.plot_name + ' ' + cost_type,
                c=self.plot.plot_parameters[self.plot.plot_system]['c'], linestyle=ls, alpha=0.7)

        if ax_in is None:
            ax.set_xlabel(r'Time $t$')
            ax.set_ylabel(r'Cost')
            plt.show()

    def plot_openloop_eigvals(self, ax_in=None):
        if ax_in is None:
            fig = plt.figure()
            grid = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(grid[0, 0])
        else:
            ax = ax_in

        ax.scatter(range(1, self.number_of_states+1), np.sort(np.abs(self.A.open_loop_eig_vals)),
                   marker='x', c='black', alpha=0.7)

        if ax_in is None:
            ax.set_xlabel('Mode')
            ax.set_ylabel(r'$|\lambda_i(A)|$')
            plt.show()

    def plot_states(self, ax_in=None):
        if ax_in is None:
            fig = plt.figure()
            grid = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(grid[0, 0])
        else:
            ax = ax_in

        x = self.vector_from_dict_key_states(self.trajectory.x)
        for s in range(0, self.number_of_states):
            ax.stairs(x[s, 0:self.sim.t_simulate-1], range(0, self.sim.t_simulate), color=self.plot.plot_parameters[self.plot.plot_system]['c'], alpha=0.1)

        if ax_in is None:
            ax.set_xlabel('Mode')
            ax.set_ylabel(r'$|\lambda_i(A)|$')
            plt.show()

    def plot_states_estimates_norm(self, ax_in=None):
        if ax_in is None:
            fig = plt.figure()
            grid = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(grid[0, 0])
        else:
            ax = ax_in

        x = self.vector_from_dict_key_states(self.trajectory.x)
        x_norm = [np.linalg.norm(x[:, i]) for i in range(0, self.sim.t_simulate)]
        x_estimate = self.vector_from_dict_key_states(self.trajectory.x_estimate)
        x_estimate_norm = [np.linalg.norm(x_estimate[:, i]) for i in range(0, self.sim.t_simulate)]

        ax.plot(range(0, self.sim.t_simulate), x_norm, color=self.plot.plot_parameters[self.plot.plot_system]['c'], alpha=0.8)
        ax.plot(range(0, self.sim.t_simulate), x_estimate_norm, color=self.plot.plot_parameters[self.plot.plot_system]['c'], ls='dashed', alpha=0.8)

        x_lim = ax.get_xlim()
        if x_lim[0] == 0:
            ax.set_xlim(-10, x_lim[1])

        if ax_in is None:
            ax.set_xlabel('Mode')
            ax.set_ylabel(r'$|\lambda_i(A)|$')
            plt.show()

    def vector_from_dict_key_cost(self, v: dict):
        ret_list = []
        for t in range(0, self.sim.t_simulate):
            ret_list.append(v[t])
        return ret_list

    def vector_from_dict_key_states(self, v: dict):
        ret_list = np.empty((self.number_of_states, self.sim.t_simulate))
        for t in range(0, self.sim.t_simulate):
            ret_list[:, t] = v[t]
        return ret_list


def cost_mapper(S: System, choices):
    # if S.sim.multiprocess_check:
    #     with Pool(processes=os.cpu_count() - 4) as P:
    #         evaluation = list(P.imap(S.evaluate_cost_for_choice, choices))
    # else:
    #     evaluation = list(map(S.evaluate_cost_for_choice, choices))

    evaluation = list(map(S.evaluate_cost_for_choice, choices))
    return evaluation


def greedy_selection(S: System, number_of_changes_limit: int = None, print_check: bool = False, t_start=time.time()):
    work_sys = dc(S)
    cost_improvement = [work_sys.trajectory.cost.predicted[work_sys.sim.t_current]]

    if print_check:
        print('Initial architecture')
        work_sys.architecture_display()

    number_of_changes, number_of_choices, selection_check = 0, 0, True
    while selection_check:
        work_iteration = dc(work_sys)
        choices = work_iteration.available_choices_selection()
        if print_check:
            print('Iteration: {}'.format(number_of_changes))
            work_iteration.architecture_display()

        if len(choices) == 0:
            # Exit if there are no available selections or the only option is no change
            selection_check = False
            if print_check:
                print('Selection exit 1: No available selections')

        else:
            number_of_choices += len(choices)
            # evaluations = list(map(work_iteration.evaluate_cost_for_choice, choices))
            evaluations = cost_mapper(work_iteration, choices)
            smallest_cost = min(d['cost'] for d in evaluations)
            index_of_smallest_cost = evaluations.index(min(d for d in evaluations if d['cost'] == smallest_cost))

            if print_check:
                print('Choice evaluations')
                for i in range(0, len(evaluations)):
                    print('{}: {} | {} | {} | {}'.format(i, evaluations[i]['arch'], evaluations[i]['idx'], evaluations[i]['change'], evaluations[i]['cost']))
                print('Smallest cost: {} @ {}'.format(smallest_cost, index_of_smallest_cost))
                evaluations[index_of_smallest_cost]['system'].architecture_display()

            if evaluations[index_of_smallest_cost]['arch'] == 'skip':
                # Exit if the best option is no change
                selection_check = False
                if print_check:
                    print('Selection exit 2: No valuable selections')

            else:
                work_sys = dc(evaluations[index_of_smallest_cost]['system'])
                cost_improvement.append(smallest_cost)
                number_of_changes += 1

                if number_of_changes_limit is not None and number_of_changes >= number_of_changes_limit:
                    # Exit if maximum number of changes have been completed
                    selection_check = False

                    if print_check:
                        print('Selection exit 3: Maximum selections done')

    work_sys.trajectory.computation_time[work_sys.sim.t_current] = time.time() - t_start
    if print_check:
        print('Number of iterations: {}'.format(number_of_changes))
        work_sys.architecture_display()
        print('Computation time: {}\nCost Improvement: {}'.format(work_sys.trajectory.computation_time[work_sys.sim.t_current], cost_improvement))
    return work_sys


def greedy_rejection(S: System, number_of_changes_limit: int = None, print_check: bool = False, t_start=time.time()):
    work_sys = dc(S)
    cost_improvement = [work_sys.trajectory.cost.predicted[work_sys.sim.t_current]]

    if print_check:
        print('Initial architecture')
        work_sys.architecture_display()

    number_of_changes, number_of_choices, rejection_check = 0, 0, True
    while rejection_check:
        work_iteration = dc(work_sys)
        choices = work_iteration.available_choices_rejection()
        if print_check:
            print('Iteration: {}'.format(number_of_changes))
            work_iteration.architecture_display()

        if len(choices) == 0:
            # Exit if there are no available rejections or the only option is no change
            rejection_check = False
            if print_check:
                print('Rejection exit 1: No available rejections')

        else:
            number_of_choices += len(choices)
            # evaluations = list(map(work_iteration.evaluate_cost_for_choice, choices))
            evaluations = cost_mapper(work_iteration, choices)
            smallest_cost = min(d['cost'] for d in evaluations)
            index_of_smallest_cost = evaluations.index(min(d for d in evaluations if d['cost'] == smallest_cost))

            if print_check:
                print('Choice evaluations')
                for i in range(0, len(evaluations)):
                    print('{}: {} | {} | {} | {}'.format(i, evaluations[i]['arch'], evaluations[i]['idx'], evaluations[i]['change'], evaluations[i]['cost']))
                print('Smallest cost: {} @ {}'.format(smallest_cost, index_of_smallest_cost))
                evaluations[index_of_smallest_cost]['system'].architecture_display()

            if evaluations[index_of_smallest_cost]['arch'] == 'skip':
                # Exit if the best option is no change
                rejection_check = False
                if print_check:
                    print('Rejection exit 2: No valuable rejections')

            else:
                work_sys = dc(evaluations[index_of_smallest_cost]['system'])
                cost_improvement.append(smallest_cost)
                number_of_changes += 1

                if number_of_changes_limit is not None and number_of_changes >= number_of_changes_limit:
                    # Exit if maximum number of changes have been completed
                    rejection_check = False

                    if print_check:
                        print('Rejection exit 3: Maximum rejections done')

    work_sys.trajectory.computation_time[work_sys.sim.t_current] = time.time() - t_start
    if print_check:
        print('Number of iterations: {}'.format(number_of_changes))
        work_sys.architecture_display()
        print('Computation time: {}\nCost Improvement: {}'.format(work_sys.trajectory.computation_time[work_sys.sim.t_current], cost_improvement))
    return work_sys


def greedy_simultaneous(S: System, number_of_changes_limit: int = None, number_of_changes_per_iteration: int = None, print_check_outer: bool = False, print_check_inner: bool = False, t_start=time.time()):
    work_sys = dc(S)
    cost_improvement = [work_sys.trajectory.cost.predicted[work_sys.sim.t_current]]
    swap_limit_mod = 1 if number_of_changes_per_iteration is None else number_of_changes_per_iteration

    if print_check_outer:
        print('Initial architecture')
        work_sys.architecture_display()

    number_of_changes, number_of_choices, simultaneous_check = 0, 0, True
    while simultaneous_check:
        force_swap = dc(work_sys)
        force_swap.architecture_limit_mod(min_mod=swap_limit_mod, max_mod=swap_limit_mod)
        force_swap = greedy_selection(force_swap, number_of_changes_limit=2*swap_limit_mod, print_check=print_check_inner)

        if print_check_outer:
            print('After force selection')
            force_swap.architecture_display()

        force_swap = dc(force_swap)
        force_swap.architecture_limit_mod(min_mod=-swap_limit_mod, max_mod=-swap_limit_mod)
        force_swap = greedy_rejection(force_swap, number_of_changes_limit=2*swap_limit_mod, print_check=print_check_inner)

        cost_improvement.append(force_swap.trajectory.cost.predicted[force_swap.sim.t_current])

        if print_check_outer:
            print('After force rejection')
            force_swap.architecture_display()

        if work_sys.architecture_compare_active_set_to_system(force_swap):
            simultaneous_check = False
            if print_check_outer:
                print('Swap exit 1: No more valuable swaps')

        else:
            number_of_changes += work_sys.architecture_compute_active_set_changes(force_swap)
            work_sys = dc(force_swap)

            if number_of_changes_limit is not None and number_of_changes >= number_of_changes_limit:
                simultaneous_check = False
                if print_check_outer:
                    print('Swap exit 2: Maximum swaps done')

    work_sys.trajectory.computation_time[work_sys.sim.t_current] = time.time() - t_start
    if print_check_outer:
        work_sys.architecture_display()
        print('Computation time: {}\nCost Improvement: {}'.format(work_sys.trajectory.computation_time[work_sys.sim.t_current], cost_improvement))
    return work_sys


def optimize_initial_architecture(S: System, print_check: bool = False):
    if print_check:
        print('Optimizing design-time architecture from:')
        S.architecture_display()

    t_predict_ref = dc(S.sim.t_predict)
    S.sim.t_predict = 2*t_predict_ref
    S.trajectory.cost.metric_control = 2
    S.prediction_gains()
    S.cost_prediction_wrapper()

    S = greedy_simultaneous(S)

    S.sim.t_predict = dc(t_predict_ref)
    S.trajectory.cost.metric_control = 1
    S.prediction_gains()
    S.cost_prediction_wrapper()

    if print_check:
        print('Design-time architecture optimized to:')
        S.architecture_display()

    return S


def simulate_fixed_architecture(S: System, print_check: bool = False, tqdm_check: bool = True):
    S_fix = dc(S)
    S_fix.sim.sim_model = "fixed"
    S_fix.model_namer()

    if print_check:
        print('Simulating Fixed Architecture')

    if tqdm_check:
        with tqdm(total=S_fix.sim.t_simulate, ncols=100, desc='Fixed (P_ID:' + str(os.getpid()) + ')', leave=False) as pbar:
            for _ in range(0, S_fix.sim.t_simulate):
                S_fix.cost_true_wrapper()
                S_fix.one_step_system_update()
                pbar.update()
    else:
        for _ in range(0, S_fix.sim.t_simulate):
            S_fix.cost_true_wrapper()
            S_fix.one_step_system_update()

    if print_check:
        print('Fixed Architecture Simulation: DONE')

    return S_fix


def simulate_self_tuning_architecture(S: System, number_of_changes_limit: int = None, print_check: bool = False, tqdm_check: bool = True):
    S_self_tuning = dc(S)
    S_self_tuning.sim.sim_model = "selftuning"
    S_self_tuning.model_namer()

    if print_check:
        print('Simulating Self-Tuning Architecture')

    if tqdm_check:
        with tqdm(total=S_self_tuning.sim.t_simulate, ncols=100, desc='Self-Tuning (P_ID:' + str(os.getpid()) + ')', leave=False) as pbar:
            for t in range(0, S_self_tuning.sim.t_simulate):
                if t > 0:
                    S_self_tuning = greedy_simultaneous(S_self_tuning, number_of_changes_limit=number_of_changes_limit, print_check_outer=print_check)
                S_self_tuning.cost_true_wrapper()
                S_self_tuning.one_step_system_update()
                pbar.update()
    else:
        for t in range(0, S_self_tuning.sim.t_simulate):
            if t > 0:
                S_self_tuning = greedy_simultaneous(S_self_tuning, number_of_changes_limit=number_of_changes_limit, print_check_outer=print_check)
            S_self_tuning.cost_true_wrapper()
            S_self_tuning.one_step_system_update()

    if print_check:
        print('Self-Tuning Architecture Simulation: DONE')

    return S_self_tuning


def simulate_experiment_fixed_vs_selftuning(exp_no: int = 1, number_of_changes_limit: int = None, print_check: bool = False, tqdm_check: bool = True, statistics_model=0):
    S = initialize_system_from_experiment_number(exp_no)

    S = optimize_initial_architecture(S, print_check=print_check)

    S_fix = simulate_fixed_architecture(S, print_check=print_check, tqdm_check=tqdm_check)
    S_fix.plot_name = 'fixed arch'

    S_tuning = simulate_self_tuning_architecture(S, number_of_changes_limit=S.sim.test_parameter, print_check=print_check, tqdm_check=tqdm_check)
    S_tuning.plot_name = 'selftuning arch'

    if statistics_model == 0:
        system_model_to_memory_sim_model(S, S_fix, S_tuning)
    else:
        system_model_to_memory_statistics(S, S_fix, S_tuning, statistics_model)

    return S, S_fix, S_tuning


def simulate_experiment_fixed_vs_selftuning_pointdistribution_openloop(exp_no: int = 1, print_check: bool = False, tqdm_check: bool = True, statistics_model=0):
    S = initialize_system_from_experiment_number(exp_no)

    S = system_model_from_memory_gen_model(S.model_name)

    S.initialize_trajectory(statistics_model-1)

    S = optimize_initial_architecture(S, print_check=print_check)

    S_fix = simulate_fixed_architecture(S, print_check=print_check, tqdm_check=tqdm_check)
    S_fix.plot_name = 'fixed arch'

    S_tuning = simulate_self_tuning_architecture(S, number_of_changes_limit=S.sim.test_parameter, print_check=print_check, tqdm_check=tqdm_check)
    S_tuning.plot_name = 'selftuning arch'

    if statistics_model == 0:
        system_model_to_memory_sim_model(S, S_fix, S_tuning)
    else:
        system_model_to_memory_statistics(S, S_fix, S_tuning, statistics_model)

    return S, S_fix, S_tuning


def simulate_experiment_selftuning_number_of_changes(exp_no: int = 1, print_check: bool = False, tqdm_check: bool = True, statistics_model=0):
    S = initialize_system_from_experiment_number(exp_no)

    S = optimize_initial_architecture(S, print_check=print_check)

    S_tuning_1change = simulate_self_tuning_architecture(S, number_of_changes_limit=1, print_check=print_check, tqdm_check=tqdm_check)
    S_tuning_1change.plot_name = 'selftuning 1change'

    S_tuning_bestchange = simulate_self_tuning_architecture(S, number_of_changes_limit=S.sim.test_parameter, print_check=print_check, tqdm_check=tqdm_check)
    S_tuning_bestchange.plot_name = 'selftuning bestchange'

    if statistics_model == 0:
        system_model_to_memory_sim_model(S, S_tuning_1change, S_tuning_bestchange)
    else:
        system_model_to_memory_statistics(S, S_tuning_1change, S_tuning_bestchange, statistics_model)

    return S, S_tuning_1change, S_tuning_bestchange


def simulate_experiment_selftuning_prediction_horizon(exp_no: int = 1, print_check: bool = False, tqdm_check: bool = True, statistics_model=0):
    S = initialize_system_from_experiment_number(exp_no)

    S = optimize_initial_architecture(S, print_check=print_check)

    S_tuning_Tp = dc(S)
    S_tuning_Tp = simulate_self_tuning_architecture(S_tuning_Tp, number_of_changes_limit=1, print_check=print_check, tqdm_check=tqdm_check)
    S_tuning_Tp.plot_name = 'selftuning Tp' + str(S_tuning_Tp.sim.t_predict)

    S_tuning_nTp = dc(S)
    S_tuning_nTp.sim.t_predict *= S.sim.test_parameter
    S_tuning_nTp = simulate_self_tuning_architecture(S_tuning_nTp, number_of_changes_limit=1, print_check=print_check, tqdm_check=tqdm_check)
    S_tuning_nTp.plot_name = 'selftuning Tp' + str(S_tuning_nTp.sim.t_predict)

    if statistics_model == 0:
        system_model_to_memory_sim_model(S, S_tuning_Tp, S_tuning_nTp)
    else:
        system_model_to_memory_statistics(S, S_tuning_Tp, S_tuning_nTp, statistics_model)

    return S, S_tuning_Tp, S_tuning_nTp


def simulate_experiment_selftuning_architecture_cost(exp_no: int = 1, print_check: bool = False, tqdm_check: bool = True, statistics_model=0):
    S = initialize_system_from_experiment_number(exp_no)
    # S = optimize_initial_architecture(S, print_check=print_check)

    S_tuning_WO_cost = dc(S)
    S_tuning_WO_cost.B.R2, S_tuning_WO_cost.B.R3, S_tuning_WO_cost.C.R2, S_tuning_WO_cost.C.R3 = 0, 0, 0, 0
    S_tuning_WO_cost = optimize_initial_architecture(S_tuning_WO_cost, print_check=print_check)
    S_tuning_WO_cost = simulate_self_tuning_architecture(S_tuning_WO_cost, number_of_changes_limit=1, print_check=print_check, tqdm_check=tqdm_check)
    S_tuning_WO_cost.plot_name = 'selftuning w/o arch cost'

    S_tuning_W_cost = dc(S)
    S_tuning_W_cost = optimize_initial_architecture(S_tuning_W_cost, print_check=print_check)
    S_tuning_W_cost = simulate_self_tuning_architecture(S_tuning_W_cost, number_of_changes_limit=1, print_check=print_check, tqdm_check=tqdm_check)
    S_tuning_W_cost.plot_name = 'selftuning w/ arch cost'

    if statistics_model == 0:
        system_model_to_memory_sim_model(S, S_tuning_WO_cost, S_tuning_W_cost)
    else:
        system_model_to_memory_statistics(S, S_tuning_WO_cost, S_tuning_W_cost, statistics_model)

    return S, S_tuning_WO_cost, S_tuning_W_cost


# def simulate_statistics_experiment_fixed_vs_selftuning(exp_no: int = 0, start_idx: int = 1, number_of_samples: int = 100):
#
#     idx_range = list(range(start_idx, number_of_samples + start_idx))
#     S = initialize_system_from_experiment_number(exp_no)
#
#     if S.sim.multiprocess_check:
#         arg_list = zip([exp_no] * len(idx_range), [1] * len(idx_range), [False] * len(idx_range), [True] * len(idx_range), idx_range)
#         with multiprocessing.Pool(processes=os.cpu_count() - 4) as P:
#             # list(tqdm.tqdm(p.imap(func, iterable), total=len(iterable)))
#             # list(tqdm(P.starmap(simulate_experiment_fixed_vs_selftuning, arg_list), desc='Simulations', ncols=100, total=len(idx_range)))
#             P.imap(simulate_experiment_fixed_vs_selftuning, arg_list)
#     else:
#         for test_no in tqdm(idx_range, desc='Simulations', ncols=100):
#             _, _, _ = simulate_experiment_fixed_vs_selftuning(exp_no=exp_no, number_of_changes_limit=1, statistics_model=test_no)


def simulate_statistics_experiment_fixed_vs_selftuning(exp_no: int = 0, start_idx: int = 1, number_of_samples: int = 100):

    idx_range = list(range(start_idx, number_of_samples + start_idx))
    S = initialize_system_from_experiment_number(exp_no)

    if S.sim.multiprocess_check:
        with tqdm(total=len(idx_range), ncols=100, desc='Model ID', leave=True) as pbar:
            # with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for _ in executor.map(simulate_experiment_fixed_vs_selftuning, itertools.repeat(exp_no), itertools.repeat(False), itertools.repeat(False), idx_range):
                    pbar.update()
    else:
        for test_no in tqdm(idx_range, desc='Simulations', ncols=100, position=0, leave=True):
            _, _, _ = simulate_experiment_fixed_vs_selftuning(exp_no=exp_no, number_of_changes_limit=1, statistics_model=test_no)


def simulate_statistics_experiment_selftuning_number_of_changes(exp_no: int = 0, start_idx: int = 1, number_of_samples: int = 100):
    idx_range = list(range(start_idx, number_of_samples + start_idx))
    S = initialize_system_from_experiment_number(exp_no)

    if S.sim.multiprocess_check:
        with tqdm(total=len(idx_range), ncols=100, desc='Model ID', leave=True) as pbar:
            # with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for _ in executor.map(simulate_experiment_selftuning_number_of_changes, itertools.repeat(exp_no), itertools.repeat(False), itertools.repeat(False), idx_range):
                    pbar.update()
    else:
        for test_no in tqdm(idx_range, desc='Simulations', ncols=100, position=0, leave=True):
            _, _, _ = simulate_experiment_selftuning_number_of_changes(exp_no=exp_no, tqdm_check=True, statistics_model=test_no)


def simulate_statistics_experiment_selftuning_prediction_horizon(exp_no: int = 0, start_idx: int = 1, number_of_samples: int = 100):
    idx_range = list(range(start_idx, number_of_samples + start_idx))
    S = initialize_system_from_experiment_number(exp_no)

    if S.sim.multiprocess_check:
        with tqdm(total=len(idx_range), ncols=100, desc='Model ID', leave=True) as pbar:
            # with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for _ in executor.map(simulate_experiment_selftuning_prediction_horizon, itertools.repeat(exp_no), itertools.repeat(False), itertools.repeat(False), idx_range):
                    pbar.update()
    else:
        for test_no in tqdm(idx_range, desc='Simulations', ncols=100, position=0, leave=True):
            _, _, _ = simulate_experiment_selftuning_prediction_horizon(exp_no=exp_no, tqdm_check=True, statistics_model=test_no)


def simulate_statistics_experiment_selftuning_architecture_cost(exp_no: int = 0, start_idx: int = 1, number_of_samples: int = 100):
    idx_range = list(range(start_idx, number_of_samples + start_idx))
    S = initialize_system_from_experiment_number(exp_no)

    if S.sim.multiprocess_check:
        with tqdm(total=len(idx_range), ncols=100, desc='Model ID', leave=True) as pbar:
            # with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for _ in executor.map(simulate_experiment_selftuning_architecture_cost, itertools.repeat(exp_no), itertools.repeat(False), itertools.repeat(False), idx_range):
                    pbar.update()
    else:
        for test_no in tqdm(idx_range, desc='Simulations', ncols=100, position=0, leave=True):
            _, _, _ = simulate_experiment_selftuning_architecture_cost(exp_no=exp_no, tqdm_check=True, statistics_model=test_no)


def simulate_statistics_experiment_pointdistribution_openloop(exp_no: int = 0):
    S_temp = initialize_system_from_experiment_number(exp_no)
    system_model_to_memory_gen_model(S_temp)

    idx_range = list(range(1, 1 + S_temp.number_of_states))

    if S_temp.sim.multiprocess_check:
        with tqdm(total=len(idx_range), ncols=100, desc='Model ID', leave=True) as pbar:
            # with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for _ in executor.map(simulate_experiment_fixed_vs_selftuning_pointdistribution_openloop, itertools.repeat(exp_no), itertools.repeat(False), itertools.repeat(False), idx_range):
                    pbar.update()
    else:
        for test_no in tqdm(idx_range, desc='Simulations', ncols=100, position=0, leave=True):
            _, _, _ = simulate_experiment_fixed_vs_selftuning_pointdistribution_openloop(exp_no=exp_no, tqdm_check=True, statistics_model=test_no)


def simulate_experiment(exp_no: int = None, print_check: bool = False):

    if exp_no is None:
        raise Exception('No experiment number provided')
    else:
        print('Experiment number: ', exp_no)

    S = initialize_system_from_experiment_number(exp_no)
    if S.sim.test_model is None or S.sim.test_model == 'fixed_vs_selftuning':
        _, _, _ = simulate_experiment_fixed_vs_selftuning(exp_no=exp_no, number_of_changes_limit=S.sim.test_parameter, print_check=print_check)
    elif S.sim.test_model == 'selftuning_number_of_changes':
        _, _, _ = simulate_experiment_selftuning_number_of_changes(exp_no=exp_no, print_check=print_check)
    elif S.sim.test_model == 'selftuning_prediction_time':
        _, _, _ = simulate_experiment_selftuning_prediction_horizon(exp_no=exp_no, print_check=print_check)
    elif S.sim.test_model == 'selftuning_architecture_cost':
        _, _, _ = simulate_experiment_selftuning_architecture_cost(exp_no=exp_no, print_check=print_check)
    elif S.sim.test_model == 'pointdistribution_openloop':
        _, _, _ = simulate_experiment_fixed_vs_selftuning_pointdistribution_openloop(exp_no=exp_no, print_check=print_check, statistics_model=1)

    elif S.sim.test_model == 'statistics_fixed_vs_selftuning':
        simulate_statistics_experiment_fixed_vs_selftuning(exp_no=exp_no)
    elif S.sim.test_model == 'statistics_selftuning_number_of_changes':
        simulate_statistics_experiment_selftuning_number_of_changes(exp_no=exp_no)
    elif S.sim.test_model == 'statistics_selftuning_prediction_horizon':
        simulate_statistics_experiment_selftuning_prediction_horizon(exp_no=exp_no)
    elif S.sim.test_model == 'statistics_selftuning_architecture_cost':
        simulate_statistics_experiment_selftuning_architecture_cost(exp_no=exp_no)
    elif S.sim.test_model == 'statistics_experiment_pointdistribution_openloop':
        simulate_statistics_experiment_pointdistribution_openloop(exp_no=exp_no)
    else:
        raise Exception('Experiment not defined')


def retrieve_experiment(exp_no: int = 1):
    S = initialize_system_from_experiment_number(exp_no)
    S, S_1, S_2 = system_model_from_memory_sim_model(S.model_name)
    return S, S_1, S_2


def system_model_to_memory_gen_model(S: System):  # Store model generated from experiment parameters
    shelve_filename = datadump_folder_path + 'gen_' + S.model_name
    with shelve.open(shelve_filename, writeback=True) as shelve_data:
        shelve_data['s'] = S
    print('\nShelving gen model: ', shelve_filename)


def system_model_from_memory_gen_model(model, print_check=False):  # Retrieve model generated from experiment parameters
    shelve_filename = datadump_folder_path + 'gen_' + model
    if print_check:
        print('\nReading gen model: ', shelve_filename)
    with shelve.open(shelve_filename, flag='r') as shelve_data:
        S = shelve_data['s']
    if not isinstance(S, System):
        raise Exception('System model error')
    return S


def system_model_to_memory_sim_model(S: System, S_1: System, S_2: System):  # Store simulated models
    shelve_filename = datadump_folder_path + 'sim_' + S.model_name
    print('\nShelving sim model:', shelve_filename)
    with shelve.open(shelve_filename, writeback=True) as shelve_data:
        shelve_data['s'] = S
        shelve_data['s1'] = S_1
        shelve_data['s2'] = S_2


def system_model_from_memory_sim_model(model):  # Retrieve simulated models
    shelve_filename = datadump_folder_path + 'sim_' + model
    print('\nReading sim model: ', shelve_filename)
    with shelve.open(shelve_filename, flag='r') as shelve_data:
        S = shelve_data['s']
        S_1 = shelve_data['s1']
        S_2 = shelve_data['s2']
    if not isinstance(S, System) or not isinstance(S_1, System) or not isinstance(S_2, System):
        raise Exception('Data type mismatch')
    S.plot = PlotParameters()
    S_1.plot = PlotParameters(1)
    S_2.plot = PlotParameters(2)
    return S, S_1, S_2


def system_model_to_memory_statistics(S: System, S_1: System, S_2: System, model_id: int, print_check: bool = False):
    shelve_filename = datadump_folder_path + 'statistics/' + S.model_name
    if not os.path.isdir(shelve_filename):
        os.makedirs(shelve_filename)
    shelve_filename = shelve_filename + '/model_' + str(model_id)
    with shelve.open(shelve_filename, writeback=True) as shelve_data:
        shelve_data['s'] = S
        shelve_data['s1'] = S_1
        shelve_data['s2'] = S_2
    if print_check:
        print('\nShelving model:', shelve_filename)


def data_from_memory_statistics(exp_no: int = None, model_id: int = None, print_check=False):
    if exp_no is None:
        raise Exception('Experiment not provided')

    S = initialize_system_from_experiment_number(exp_no)

    shelve_filename = datadump_folder_path + 'statistics/' + S.model_name + '/model_' + str(model_id)
    with shelve.open(shelve_filename, flag='r') as shelve_data:
        S = shelve_data['s']
        S_1 = shelve_data['s1']
        S_2 = shelve_data['s2']
    if not isinstance(S, System) or not isinstance(S_1, System) or not isinstance(S_2, System):
        raise Exception('Data type mismatch')
    if print_check:
        print('\nModel read done: ', shelve_filename)
    return S, S_1, S_2


def plot_experiment(exp_no: int = None):
    if exp_no is None:
        raise Exception('Check experiment number')

    S = initialize_system_from_experiment_number(exp_no)

    if S.sim.test_model is None or S.sim.test_model == 'fixed_vs_selftuning' or S.sim.test_model == 'selftuning_number_of_changes' or S.sim.test_model == 'selftuning_prediction_time' or S.sim.test_model == 'selftuning_architecture_cost':
        plot_comparison_exp_no(exp_no)
    elif S.sim.test_model == 'statistics_fixed_vs_selftuning' or S.sim.test_model == 'statistics_selftuning_number_of_changes' or S.sim.test_model == 'statistics_selftuning_prediction_horizon' or S.sim.test_model == 'statistics_selftuning_architecture_cost' or S.sim.test_model == 'statistics_experiment_pointdistribution_openloop':
        plot_statistics_exp_no(exp_no)
    else:
        raise Exception('Experiment not defined')


def plot_comparison_exp_no(exp_no: int = 1):

    S, S_1, S_2 = retrieve_experiment(exp_no)

    if S_1.plot is None:
        S_1.plot = PlotParameters()

    if S_2.plot is None:
        S_2.plot = PlotParameters()

    # fig = plt.figure(tight_layout=True)
    fig = plt.figure(constrained_layout=True)
    outer_grid = gs.GridSpec(2, 1, figure=fig, height_ratios=[4, 1])

    time_grid = gs.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer_grid[0, 0])
    eval_grid = gs.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[1, 0])

    ax_cost = fig.add_subplot(time_grid[0, 0])
    ax_B = fig.add_subplot(time_grid[1, 0], sharex=ax_cost)
    ax_C = fig.add_subplot(time_grid[2, 0], sharex=ax_cost)
    ax_state = fig.add_subplot(time_grid[3, 0], sharex=ax_cost)

    ax_eval = fig.add_subplot(eval_grid[0, 0])
    
    S_1.plot_cost(ax_in=ax_cost)
    S_2.plot_cost(ax_in=ax_cost)
    S_1.plot_cost(ax_in=ax_cost, cost_type='predict')
    S_2.plot_cost(ax_in=ax_cost, cost_type='predict')

    leg1 = ax_cost.legend(handles=[mlines.Line2D([], [], color=S_1.plot.plot_parameters[S_1.plot.plot_system]['c'], ls='solid', label='True'),
                                   mlines.Line2D([], [], color=S_1.plot.plot_parameters[S_1.plot.plot_system]['c'], ls='dashed', label='Predicted')],
                          loc='upper left', ncol=2)
    ax_cost.add_artist(leg1)

    leg2 = ax_cost.legend(handles=[mpatches.Patch(color=S_1.plot.plot_parameters[S_1.plot.plot_system]['c'], label=S_1.plot_name),
                                   mpatches.Patch(color=S_2.plot.plot_parameters[S_2.plot.plot_system]['c'], label=S_2.plot_name)],
                          loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)
    # ax_cost.add_artist(leg2)

    ax_cost.tick_params(axis="x", labelbottom=False)
    ax_cost.set_ylabel('Cost\n'+r'$J_t$')
    # ax_cost.ticklabel_format(axis='y', style='sci', scilimits=[-2, 2])
    ax_cost.set_yscale('log')
    ax_cost.grid(visible=True, which='major', axis='x')

    S_1.plot_architecture_history(arch='B', ax_in=ax_B)
    S_2.plot_architecture_history(arch='B', ax_in=ax_B)
    ax_B.set_ylabel('Actuator\nPosition\n'+r'$S_t$')
    ax_B.tick_params(axis="x", labelbottom=False)
    ax_B.grid(visible=True, which='major', axis='x')
    # ax_B.legend()

    S_1.plot_architecture_history(arch='C', ax_in=ax_C)
    S_2.plot_architecture_history(arch='C', ax_in=ax_C)
    ax_C.set_ylabel('Sensor\nPosition\n'+r'$S_t$'+'\'')
    # ax_C.legend()
    # ax_C.set_xlabel('Time')
    ax_C.tick_params(axis="x", labelbottom=False)
    ax_C.grid(visible=True, which='major', axis='x')

    S_1.plot_states_estimates_norm(ax_in=ax_state)
    S_2.plot_states_estimates_norm(ax_in=ax_state)

    # S_1.plot_states(ax_in=ax_state)
    # S_2.plot_states(ax_in=ax_state)
    # bmin, bmax = ax_state.get_ylim()
    # b = max(abs(bmin), abs(bmax))
    # ax_state.set_ylim(-b, b)

    ax_state.legend(handles=[mlines.Line2D([], [], color=S_1.plot.plot_parameters[S_1.plot.plot_system]['c'], ls='solid', label=r'$|x_t|$'),
                             mlines.Line2D([], [], color=S_1.plot.plot_parameters[S_1.plot.plot_system]['c'], ls='dashed', label=r'$|\hat{x}_t|$')],
                    loc='upper left', ncol=2)

    ax_state.set_xlabel('Time')
    ax_state.set_ylabel('States')
    ax_state.grid(visible=True, which='major', axis='x')

    # x_1 = S_1.vector_from_dict_key_states(S_1.trajectory.x)
    # x_2 = S_2.vector_from_dict_key_states(S_2.trajectory.x)
    # print(np.shape(x_1))
    # ax_state.stairs(range(0, S_1.sim.t_simulate), x_1)
    # ax_state.stairs(range(0, S_2.sim.t_simulate), x_2)

    S_1.plot_openloop_eigvals(ax_in=ax_eval)
    ax_eval.hlines(1, xmin=1, xmax=S.number_of_states, colors='black', ls='dotted')
    ax_eval.set_xlabel('Mode')
    ax_eval.set_ylabel('Openloop\nEigenvalues\n' + r'$|\lambda_i(A)|$')

    fig.suptitle('Experiment No: ' + str(exp_no))

    plt.savefig(image_save_folder_path + 'exp' + str(exp_no) + '.pdf', format='pdf')

    plt.show()


def element_wise_min_max(v_ref_min, v_ref_max, v):

    v_ret_min = [min(e) for e in zip(v_ref_min, v)]
    v_ret_max = [max(e) for e in zip(v_ref_max, v)]

    return v_ret_min, v_ret_max


def plot_statistics_exp_no(exp_no: int = None):
    if exp_no is None:
        raise Exception('Experiment not provided')

    S = initialize_system_from_experiment_number(exp_no)

    fig = plt.figure(tight_layout=True)
    outer_grid = gs.GridSpec(2, 2, figure=fig)
    arch_grid = gs.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[1, 1], wspace=0)

    cost_ax = fig.add_subplot(outer_grid[0, :])

    archB_ax = fig.add_subplot(arch_grid[0, 0])
    archC_ax = fig.add_subplot(arch_grid[0, 1], sharey=archB_ax)

    eig_ax = fig.add_subplot(outer_grid[1, 0])

    cstyle = ['tab:blue', 'tab:orange', 'black']
    lstyle = ['dotted', 'dashed']
    mstyle = ['o', '+', 'x']

    cost_min_1 = np.inf * np.ones(S.sim.t_simulate)
    cost_max_1 = np.zeros(S.sim.t_simulate)

    cost_min_2 = np.inf * np.ones(S.sim.t_simulate)
    cost_max_2 = np.zeros(S.sim.t_simulate)

    sample_cost_1 = np.zeros(S.sim.t_simulate)
    sample_cost_2 = np.zeros(S.sim.t_simulate)
    sample_eig = np.zeros(S.number_of_states)
    # sample_arch_count1 = []
    # sample_arch_count2 = []

    arch_change_1 = {'B': [], 'C': []}
    arch_change_2 = {'B': [], 'C': []}

    sim_range = 100 if S.sim.test_model == 'statistics_fixed_vs_selftuning' or S.sim.test_model == 'statistics_selftuning_number_of_changes' or S.sim.test_model == 'statistics_selftuning_prediction_horizon' or S.sim.test_model == 'statistics_selftuning_architecture_cost' else S.number_of_states

    sample_ID = np.random.choice(range(1, sim_range + 1))

    for model_no in tqdm(range(1, sim_range + 1), ncols=100, desc='Model ID'):
        S, S_1, S_2 = data_from_memory_statistics(exp_no, model_no)

        S_1_true_cost = list(itertools.accumulate(S_1.vector_from_dict_key_cost(S_1.trajectory.cost.true)))
        S_2_true_cost = list(itertools.accumulate(S_2.vector_from_dict_key_cost(S_2.trajectory.cost.true)))

        # cost_ax.plot(range(0, S.sim.t_simulate), S_1_true_cost, color='tab:blue', alpha=0.1)
        # cost_ax.plot(range(0, S.sim.t_simulate), S_2_true_cost, color='tab:orange', alpha=0.1)

        cost_min_1, cost_max_1 = element_wise_min_max(cost_min_1, cost_max_1, S_1_true_cost)
        cost_min_2, cost_max_2 = element_wise_min_max(cost_min_2, cost_max_2, S_2_true_cost)

        S_1.architecture_count_number_of_sim_changes()
        arch_change_1['B'].append(S_1.B.change_count)
        arch_change_1['C'].append(S_1.C.change_count)

        S_2.architecture_count_number_of_sim_changes()
        arch_change_2['B'].append(S_2.B.change_count)
        arch_change_2['C'].append(S_2.C.change_count)

        eig_ax.scatter(range(1, S.number_of_states + 1), np.sort(np.abs(S.A.open_loop_eig_vals)), marker=mstyle[2], color=cstyle[0], alpha=float(1/S.number_of_states))
        if sample_ID == model_no:
            sample_cost_1 = S_1_true_cost
            sample_cost_2 = S_2_true_cost
            sample_eig = np.sort(np.abs(S.A.open_loop_eig_vals))
            # sample_arch_count1 = [S_1.B.change_count, S_1.C.change_count]
            # sample_arch_count2 = [S_2.B.change_count, S_2.C.change_count]

    cost_ax.fill_between(range(0, S.sim.t_simulate), cost_min_1, cost_max_1, color=cstyle[0], alpha=0.4)
    cost_ax.fill_between(range(0, S.sim.t_simulate), cost_min_2, cost_max_2, color=cstyle[1], alpha=0.4)
    cost_ax.plot(range(0, S.sim.t_simulate), sample_cost_1, color=cstyle[2], ls=lstyle[0], linewidth=1)
    cost_ax.plot(range(0, S.sim.t_simulate), sample_cost_2, color=cstyle[2], ls=lstyle[1], linewidth=1)
    cost_ax.set_yscale('log')
    cost_ax.set_xlabel('Time')
    cost_ax.set_ylabel('Cost')
    cost_ax.legend(handles=[mpatches.Patch(color=cstyle[0], label=r'$M_1$'),
                            mpatches.Patch(color=cstyle[1], label=r'$M_2$'),
                            mlines.Line2D([], [], color=cstyle[2], ls=lstyle[0], label='Sample ' + r'$M_1$'),
                            mlines.Line2D([], [], color=cstyle[2], ls=lstyle[1], label='Sample ' + r'$M_2$')],
                   loc='lower right', ncols=2)

    # arch_ax.scatter(arch_change_1['B'], arch_change_1['C'], marker=mstyle[0], color=c_lines[0], alpha=0.05)
    # arch_ax.scatter(arch_change_2['B'], arch_change_2['C'], marker=mstyle[0], color=c_lines[1], alpha=0.05)
    # arch_ax.scatter(sample_arch_count1[0], sample_arch_count1[1], marker=mstyle[1], color=c_lines[2], alpha=1)
    # arch_ax.scatter(sample_arch_count2[0], sample_arch_count2[1], marker=mstyle[2], color=c_lines[2], alpha=1)
    # arch_ax.set_xlabel('Actuator \n change count')
    # arch_ax.set_ylabel('Sensor \n change count')
    # arch_ax.legend(handles=[mlines.Line2D([], [], marker=mstyle[0], linewidth=0, color=c_lines[0], label=r'$S_1$'),
    #                         mlines.Line2D([], [], marker=mstyle[0], linewidth=0, color=c_lines[1], label=r'$S_2$'),
    #                         mlines.Line2D([], [], marker=mstyle[1], linewidth=0, color=c_lines[2], label='Sample ' + r'$S_1$'),
    #                         mlines.Line2D([], [], marker=mstyle[2], linewidth=0, color=c_lines[2], label='Sample ' + r'$S_2$')],
    #                loc='upper left')

    eig_ax.scatter(range(1, S.number_of_states + 1), sample_eig, marker=mstyle[2], color=cstyle[2], alpha=0.5)
    eig_ax.hlines(1, xmin=1, xmax=S.number_of_states, colors=cstyle[2], ls=lstyle[1])
    eig_ax.set_xlabel('Mode ' + r'$i$')
    eig_ax.set_ylabel(r'$|\lambda_i(A)|$')
    eig_ax.legend(handles=[
        mlines.Line2D([], [], color=cstyle[2], marker=mstyle[0], linewidth=0, label='Modes'),
        mlines.Line2D([], [], color=cstyle[2], marker=mstyle[2], linewidth=0, label='Sample')],
        loc='upper left')

    archB_ax.boxplot([arch_change_1['B'], arch_change_2['B']], labels=[r'$M_1$', r'$M_2$'])
    archC_ax.boxplot([arch_change_1['C'], arch_change_2['C']], labels=[r'$M_1$', r'$M_2$'])
    archC_ax.yaxis.set_tick_params(labelleft=False, left=False)
    archB_ax.set_ylabel('Number of\nChanges')
    archB_ax.set_title('Actuator ' + r'$S$')
    archC_ax.set_title('Sensor ' + r'$S$' + '\'')

    fig.suptitle('Experiment No: ' + str(exp_no))

    plt.savefig(image_save_folder_path + 'exp' + str(exp_no) + '.pdf', format='pdf')

    plt.show()
