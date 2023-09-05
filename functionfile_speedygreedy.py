import numpy as np
import networkx as netx

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator, FuncFormatter

from time import process_time
from copy import deepcopy as dc
import pandas as pd
import shelve
import itertools
# import multiprocessing
# from multiprocessing import Pool
import concurrent.futures

import os
import socket

from tqdm.autonotebook import tqdm

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
                print('Duplicate experiment at : {}'.format(i))
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


def normalize_columns_of_matrix(A_mat: np.ndarray):
    for i in range(0, np.shape(A_mat)[0]):
        if np.linalg.norm(A_mat[:, i]) != 0:
            A_mat[:, i] /= np.linalg.norm(A_mat[:, i])
    return A_mat


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
            {1: {'node': 'tab:blue', 'B': 'tab:orange', 'C': 'tab:green', 'm': 'x', 'c': 'tab:blue', 'ms': 20, 'ls': 'solid'},
             2: {'node': 'tab:blue', 'B': 'tab:orange', 'C': 'tab:green', 'm': 'o', 'c': 'tab:orange', 'ms': 20, 'ls': 'dashed'}}
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
            self.A_augmented_mat = np.zeros((0, 0))  # Open-loop augmented dynamics matrix for current t

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
            self.active_count = []  # Count size of active architecture at each timestep of the simulation horizon
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
            self.F_augmented = np.zeros((0, 0))  # augmented matrix for noise at current time

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
            self.X_augmented = {}  # augmented state trajectory
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
        print('Model name = {}'.format(self.model_name))
        print('Number of Nodes = {}'.format(self.number_of_nodes))
        print('Number of States = {}'.format(self.number_of_states))
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
                V_mat = normalize_columns_of_matrix(V_mat)

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
        self.A.open_loop_eig_vecs = normalize_columns_of_matrix(self.A.open_loop_eig_vecs)
        self.A.number_of_non_stable_modes = len([e for e in self.A.open_loop_eig_vals if e >= 1])

    def rescale(self):
        if self.A.rho is not None:
            self.A.A_mat = self.A.rho * self.A.adjacency_matrix / np.max(np.abs(np.linalg.eigvals(self.A.adjacency_matrix)))
        else:
            self.A.A_mat = self.A.adjacency_matrix

    def second_order_matrix(self):
        if self.A.second_order:
            if self.A.second_order_network_type == 1:

                self.A.A_mat = np.block([[self.A.A_mat, np.zeros_like(self.A.A_mat)],
                                         [self.A.second_order_scaling_factor * np.identity(self.number_of_nodes), self.A.second_order_scaling_factor * np.identity(self.number_of_nodes)]])
            elif self.A.second_order_network_type == 2:
                self.A.A_mat = np.block([[np.identity(self.number_of_nodes), np.zeros_like(self.A.A_mat)],
                                         [self.A.second_order_scaling_factor * np.identity(self.number_of_nodes), self.A.second_order_scaling_factor * self.A.A_mat]])
            else:
                raise SecondOrderError()

    def rescale_wrapper(self):
        self.rescale()
        self.second_order_matrix()
        self.evaluate_modes()

    def initialize_active_matrix(self, arch=None):
        for a in architecture_iterator(arch):
            if a == 'B':
                self.B.active_matrix = np.zeros((self.number_of_states, len(self.B.active_set)))
            else:  # self.architecture_type == 'C':
                self.C.active_matrix = np.zeros((len(self.C.active_set), self.number_of_states))

    def initialize_available_vectors_as_basis_vectors(self, arch=None):
        for a in architecture_iterator(arch):
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
        for a in architecture_iterator(arch):
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
            self.trajectory.x = {0: self.A.open_loop_eig_vecs[:, x0_idx] * self.trajectory.X0_scaling}

        self.trajectory.x_estimate = {0: np.random.default_rng().multivariate_normal(np.zeros(self.number_of_states), self.trajectory.X0_covariance)}

        self.trajectory.X_augmented = {0: np.concatenate((self.trajectory.x[0], self.trajectory.x_estimate[0]))}

        self.trajectory.control_cost_matrix = {}
        self.trajectory.estimation_matrix = {0: np.identity(self.number_of_states)}

        self.trajectory.error = {0: self.trajectory.x[0] - self.trajectory.x_estimate[0]}
        self.trajectory.error_2norm = {0: np.linalg.norm(self.trajectory.error[0])}

    def architecture_limit_set(self, arch=None, min_set: int = None, max_set: int = None):
        for a in architecture_iterator(arch):
            if a == 'B':
                self.B.min = self.B.number_of_available if self.B.min is None else min_set if min_set is not None else self.B.min
                self.B.max = self.B.number_of_available if self.B.max is None else max_set if max_set is not None else self.B.max
            else:  # a == 'C'
                self.C.min = self.C.number_of_available if self.C.min is None else min_set if min_set is not None else self.C.min
                self.C.max = self.C.number_of_available if self.C.max is None else max_set if max_set is not None else self.C.max

    def architecture_limit_mod(self, arch=None, min_mod: int = None, max_mod: int = None):
        min_mod = 0 if min_mod is None else min_mod
        max_mod = 0 if max_mod is None else max_mod
        for a in architecture_iterator(arch):
            if a == 'B':
                self.B.min = self.B.min + min_mod
                self.B.max = self.B.max + max_mod
            else:  # a == 'C'
                self.C.min = self.C.min + min_mod
                self.C.max = self.C.max + max_mod

    def architecture_update_to_matrix_from_active_set(self, arch=None):
        self.initialize_active_matrix(arch)
        for a in architecture_iterator(arch):
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
        for a in architecture_iterator(arch):
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
        for a in architecture_iterator(arch):
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

    def architecture_limit_check(self):
        if self.B.min <= len(self.B.active_set) <= self.B.max and self.C.min <= len(self.C.active_set) <= self.C.max:
            return True
        else:
            return False

    def architecture_compute_active_set_changes(self, reference_system):
        if not isinstance(reference_system, System):
            raise ClassError
        B_compare = compare_lists(self.B.active_set, reference_system.B.active_set)
        C_compare = compare_lists(self.C.active_set, reference_system.C.active_set)
        number_of_changes = max(len(B_compare['only1']), len(B_compare['only2'])) + max(len(C_compare['only1']), len(C_compare['only2']))
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
        print('B: {}'.format(self.B.active_set))
        print('C: {}'.format(self.C.active_set))

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
        for a in architecture_iterator(arch):
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
        A_augmented_mat = {}
        W_augmented = {}
        F_augmented = {}
        Q_augmented = {}

        W_mat = np.block([[self.disturbance.W, np.zeros((self.number_of_states, len(self.C.active_set)))],
                          [np.zeros((len(self.C.active_set), self.number_of_states)), self.disturbance.V[:, self.C.active_set][self.C.active_set, :]]])

        for t in range(0, self.sim.t_predict):
            BKt = self.B.active_matrix @ self.B.gain[t]
            ALtC = self.A.A_mat @ self.C.gain[t] @ self.C.active_matrix

            A_augmented_mat[t] = np.block([[self.A.A_mat, -BKt],
                                          [ALtC, self.A.A_mat - ALtC - BKt]])

            F_augmented[t] = np.block([[np.identity(self.number_of_states), np.zeros((self.number_of_states, len(self.C.active_set)))],
                                      [np.zeros((self.number_of_states, self.number_of_states)), self.A.A_mat @ self.C.gain[t]]])
            W_augmented[t] = F_augmented[t] @ W_mat @ F_augmented[t].T

            Q_augmented[t] = np.block([[self.B.Q, np.zeros((self.number_of_states, self.number_of_states))],
                                      [np.zeros((self.number_of_states, self.number_of_states)), self.B.gain[t].T @ self.B.R1 @ self.B.gain[t]]])


        Q_augmented[self.sim.t_predict] = np.block([[self.B.Q, np.zeros((self.number_of_states, self.number_of_states))],
                                                   [np.zeros((self.number_of_states, self.number_of_states)), np.zeros((self.number_of_states, self.number_of_states))]])

        self.A.A_augmented_mat = A_augmented_mat[0]
        self.disturbance.F_augmented = F_augmented[0]

        self.trajectory.cost.predicted_matrix = {self.sim.t_predict: Q_augmented[self.sim.t_predict]}
        self.trajectory.cost.control = 0

        for t in range(self.sim.t_predict - 1, -1, -1):
            self.trajectory.cost.control += np.trace(self.trajectory.cost.predicted_matrix[t + 1] @ W_augmented[t])
            self.trajectory.cost.predicted_matrix[t] = A_augmented_mat[t].T @ self.trajectory.cost.predicted_matrix[t + 1] @ A_augmented_mat[t]

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

        # High selection priority of actuators or sensors if < minimum
        if len(self.B.active_set) < self.B.min or len(self.C.active_set) < self.C.min:
            if len(self.B.active_set) < self.B.min:
                choices.extend([{'arch': 'B', 'idx': i, 'change': '+'} for i in compare_lists(self.B.active_set, self.B.available_indices)['only2']])
            if len(self.C.active_set) < self.C.min:
                choices.extend([{'arch': 'C', 'idx': i, 'change': '+'} for i in compare_lists(self.C.active_set, self.C.available_indices)['only2']])
        else:  # Low selection priority of actuators or sensors if >= min and < maximum
            if len(self.B.active_set) < self.B.max:
                choices.extend([{'arch': 'B', 'idx': i, 'change': '+'} for i in compare_lists(self.B.active_set, self.B.available_indices)['only2']])
            if len(self.C.active_set) < self.C.max:
                choices.extend([{'arch': 'C', 'idx': i, 'change': '+'} for i in compare_lists(self.C.active_set, self.C.available_indices)['only2']])

        return choices

    def available_choices_rejection(self):
        if len(self.B.active_set) <= self.B.max and len(self.C.active_set) <= self.C.max:
            # If maximum number of actuators AND sensors are active
            choices = [{'arch': 'skip', 'idx': None, 'change': None}]
        else:
            choices = []

        # High rejection priority of actuators or sensors if >= maximum
        if len(self.B.active_set) > self.B.max or len(self.C.active_set) > self.C.max:
            if len(self.B.active_set) > self.B.max:
                choices.extend([{'arch': 'B', 'idx': i, 'change': '-'} for i in self.B.active_set])
            if len(self.C.active_set) > self.C.max:
                choices.extend([{'arch': 'C', 'idx': i, 'change': '-'} for i in self.C.active_set])
        else:  # Low rejection priority of actuators or sensors if > minimum and <= maximum
            if len(self.B.active_set) > self.B.min:
                choices.extend([{'arch': 'B', 'idx': i, 'change': '-'} for i in self.B.active_set])
            if len(self.C.active_set) > self.C.min:
                choices.extend([{'arch': 'C', 'idx': i, 'change': '-'} for i in self.C.active_set])

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
        # print('1:', np.shape(self.disturbance.F_augmented), np.shape(self.disturbance.w_gen[self.sim.t_current]), np.shape(self.disturbance.v_gen[self.sim.t_current]))
        # print('2:', np.shape(self.A.A_augmented_mat), np.shape(self.trajectory.X_augmented[self.sim.t_current]))
        self.trajectory.X_augmented[self.sim.t_current + 1] = (self.A.A_augmented_mat @ self.trajectory.X_augmented[self.sim.t_current]) + (self.disturbance.F_augmented @ np.concatenate((self.disturbance.w_gen[self.sim.t_current], self.disturbance.v_gen[self.sim.t_current][self.C.active_set])))

        self.trajectory.x[self.sim.t_current + 1] = self.trajectory.X_augmented[self.sim.t_current + 1][0:self.number_of_states]

        self.trajectory.x_estimate[self.sim.t_current + 1] = self.trajectory.X_augmented[self.sim.t_current + 1][self.number_of_states:]

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

    def generate_network_architecture_graph_matrix(self, mA: float = 1, mB: float = 1, mC: float = 1):
        A_mat = np.array((self.A.adjacency_matrix > 0) * mA, dtype=float)
        B_mat = np.array((self.B.active_matrix > 0) * mB, dtpye=float)
        C_mat = np.array((self.C.active_matrix > 0) * mC, dtpye=float)

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

    def generate_architecture_history_points(self, arch=None):
        for a in architecture_iterator(arch):
            if a == 'B':
                for t in self.B.history_active_set:
                    for node in self.B.history_active_set[t]:
                        self.plot.B_history[0].append(t)
                        self.plot.B_history[1].append(node+1)
            else:  # a == 'C'
                for t in self.C.history_active_set:
                    for node in self.C.history_active_set[t]:
                        self.plot.C_history[0].append(t)
                        self.plot.C_history[1].append(node+1)

    def architecture_active_count(self, arch=None):
        for a in architecture_iterator(arch):
            if a == 'B':
                if self.sim.sim_model == 'selftuning':
                    self.B.active_count = [len(self.B.history_active_set[i]) for i in range(0, len(self.B.history_active_set))]
                elif self.sim.sim_model == "fixed":
                    self.B.active_count = [len(self.B.active_set)] * len(self.B.history_active_set)
                else:
                    raise Exception('Invalid test model')

            else:  # if arch == 'C'
                if self.sim.sim_model == 'selftuning':
                    self.C.active_count = [len(self.C.history_active_set[i]) for i in range(0, len(self.C.history_active_set))]
                elif self.sim.sim_model == "fixed":
                    self.C.active_count = [len(self.C.active_set)] * len(self.C.history_active_set)
                else:
                    raise Exception('Invalid test model')

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

        x_val, y_val = [], []
        for a in architecture_iterator(arch):
            if a == 'B':
                self.generate_architecture_history_points(arch=a)
                x_val = self.plot.B_history[0]
                y_val = self.plot.B_history[1]
            else:  # a=='C'
                self.generate_architecture_history_points(arch=a)
                x_val = self.plot.C_history[0]
                y_val = self.plot.C_history[1]
        ax.scatter(x_val, y_val, s=10, alpha=0.7,
                   marker=self.plot.plot_parameters[self.plot.plot_system]['m'],
                   c=self.plot.plot_parameters[self.plot.plot_system]['c'])
        ax.set_ylim([0, self.number_of_states + 2])
        ax.grid(visible=True, which='major', axis='x')

        if ax_in is None:
            plt.show()

    def plot_compute_time(self, ax_in=None):
        if ax_in is None:
            fig = plt.figure()
            grid = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(grid[0, 0])
        else:
            ax = ax_in

        compute_time = self.list_from_dict_key_cost(self.trajectory.computation_time)
        ax.scatter(range(0, len(compute_time)), compute_time, color=self.plot.plot_parameters[self.plot.plot_system]['c'], marker='o', s=5)
        ax.grid(visible=True, which='major', axis='x')

        if ax_in is None:
            plt.show()

    def plot_architecture_count(self, ax_in=None, arch=None):
        if ax_in is None:
            fig = plt.figure()
            grid = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(grid[0, 0])
        else:
            ax = ax_in

        x_val, y_val = [], []
        for a in architecture_iterator(arch):
            self.architecture_active_count(arch=a)
            if a == 'B':
                x_val, y_val = range(0, len(self.B.active_count)), self.B.active_count
            else:  # a == 'C'
                x_val, y_val = range(0, len(self.C.active_count)), self.C.active_count

        ax.plot(x_val, y_val, color=self.plot.plot_parameters[self.plot.plot_system]['c'], alpha=0.7)
        ax.set_ylim(self.B.min - 1, self.B.max + 1)
        ax.set_yticks([self.B.min, self.B.max])
        ax.grid(visible=True, which='major', axis='x')

        if ax_in is None:
            plt.show()

    def plot_cost(self, cost_type=None, ax_in=None, set_details_flag=False):
        if ax_in is None:
            fig = plt.figure()
            grid = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(grid[0, 0])
        else:
            ax = ax_in

        if cost_type is None:
            cost_type = ['true', 'predict']

        cost, ls, legend_handler = [], '', []

        for t in cost_type:
            if t == 'true':
                cost = list(itertools.accumulate(self.list_from_dict_key_cost(self.trajectory.cost.true)))
                ls, labeler = 'solid', 'Cumulate Cost'
            elif t == 'predict':
                cost = self.list_from_dict_key_cost(self.trajectory.cost.predicted)
                ls, labeler = 'dashed', 'Predict'
            else:
                raise Exception('Check argument')

            ax.plot(range(0, self.sim.t_simulate), cost, c=self.plot.plot_parameters[self.plot.plot_system]['c'], linestyle=ls, alpha=0.7)
            legend_handler.append(mlines.Line2D([], [], color='black', ls=ls, label=labeler))

        ax.set_yscale('log')
        ax.grid(visible=True, which='major', axis='x')
        if set_details_flag:
            ax.set_ylabel('Cost\n' + r'$J_t$')
            leg1 = ax.legend(handles=legend_handler, loc='upper left', ncol=1)
            ax.add_artist(leg1)

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
                   marker='x', s=10, c='black', alpha=0.7)
        ax.axhline(y=1, color='tab:gray', ls='dashdot', alpha=0.5)
        ax.set_ylim(np.min(np.abs(self.A.open_loop_eig_vals)), np.max(np.abs(self.A.open_loop_eig_vals)))
        ax.set_ylabel(r'$|\lambda_i(A)|$')
        ax.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False)

        if ax_in is None:
            ax.set_xlabel('Mode')
            plt.show()

    def plot_states(self, ax_in=None, state_marker=None, set_details_flag=False):
        if ax_in is None:
            fig = plt.figure()
            grid = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(grid[0, 0])
        else:
            ax = ax_in

        if state_marker is None:
            state_marker = ['state', 'estimate', 'error']

        legend_handler = []

        for state in state_marker:
            # x, ls = [], ''
            if state == 'state':
                x = self.ndarray_from_dict_key_states(self.trajectory.x)
                ls = 'solid'
                labeler = r'$|x_t|_2$'
            elif state == 'estimate':
                x = self.ndarray_from_dict_key_states(self.trajectory.x_estimate)
                ls = 'dashdot'
                labeler = r'$|\hat{x}_t|_2$'
            elif state == 'error':
                x = self.ndarray_from_dict_key_states(self.trajectory.error)
                ls = 'dashed'
                labeler = r'$|x_t - \hat{x}_t|_2$'
            else:
                raise Exception('Check iterator')

            x_norm = [np.linalg.norm(x[:, i]) for i in range(0, self.sim.t_simulate)]
            ax.plot(range(0, self.sim.t_simulate), x_norm, color=self.plot.plot_parameters[self.plot.plot_system]['c'], ls=ls, alpha=0.8)
            legend_handler.append(mlines.Line2D([], [], color='black', ls=ls, label=labeler))

        if set_details_flag:
            ax.set_ylabel('States')
            ax.set_yscale('log')
            ax.legend(handles=legend_handler, loc='upper left', ncol=1)
            ax.grid(visible=True, which='major', axis='x')

        if ax_in is None:
            if set_details_flag:
                ax.set_xlabel('Time')
            plt.show()

    def list_from_dict_key_cost(self, v: dict):
        ret_list = []
        for t in range(0, self.sim.t_simulate):
            ret_list.append(v[t])
        return ret_list

    def ndarray_from_dict_key_states(self, v: dict):
        ret_list = np.empty((self.number_of_states, self.sim.t_simulate))
        for t in range(0, self.sim.t_simulate):
            ret_list[:, t] = v[t]
        return ret_list


def cost_mapper(S: System, choices):
    evaluation = list(map(S.evaluate_cost_for_choice, choices))
    return evaluation


def greedy_selection(S: System, number_of_changes_limit: int = None, print_check: bool = False):
    exit_condition = 0
    work_sys = dc(S)
    arch_ref = dc(S)
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
            exit_condition = 1
            if print_check:
                print('Selection exit 1: No available selections')

        elif len(choices) == 1 and choices[0]['arch'] == 'skip':
            # Exit if the only option is no change
            selection_check = False
            exit_condition = 2
            if print_check:
                print('Selection exit 2: No valuable selections')

        else:
            number_of_choices += len(choices)
            # evaluations = list(map(work_iteration.evaluate_cost_for_choice, choices))
            evaluations = cost_mapper(work_iteration, choices)
            # smallest_cost = min(d['cost'] for d in evaluations)
            # index_of_smallest_cost = evaluations.index(min(d for d in evaluations if d['cost'] == smallest_cost))

            index_of_smallest_cost = min(range(len(evaluations)), key=lambda change: evaluations[change].get('cost', float('inf')))
            smallest_cost = evaluations[index_of_smallest_cost]['cost']

            if print_check:
                print('Choice evaluations')
                for i in range(0, len(evaluations)):
                    print('{}: {} | {} | {} | {}'.format(i, evaluations[i]['arch'], evaluations[i]['idx'], evaluations[i]['change'], evaluations[i]['cost']))
                print('Smallest cost: {} @ {}'.format(smallest_cost, index_of_smallest_cost))
                evaluations[index_of_smallest_cost]['system'].architecture_display()

            if evaluations[index_of_smallest_cost]['arch'] == 'skip':
                # Exit if the best option is no change
                selection_check = False
                exit_condition = 5
                if print_check:
                    print('Selection exit 5: No valuable selections')

            else:
                work_sys = dc(evaluations[index_of_smallest_cost]['system'])
                cost_improvement.append(smallest_cost)
                number_of_changes = arch_ref.architecture_compute_active_set_changes(work_sys)
                # number_of_changes += 1

                if number_of_changes_limit is not None and number_of_changes >= number_of_changes_limit:
                    # Exit if maximum number of changes have been completed
                    selection_check = False
                    exit_condition = 3
                    if print_check:
                        print('Selection exit 3: Maximum selections done')

                elif number_of_changes == 0:
                    # Trigger 2 should always activate before this
                    selection_check = False
                    exit_condition = 4
                    if print_check:
                        print('Selection exit 4: No selections done')

    # work_sys.trajectory.computation_time[work_sys.sim.t_current] = time.time() - t_start
    if print_check:
        print('Number of iterations: {}'.format(number_of_changes))
        work_sys.architecture_display()
        # print('Computation time: {}\nCost Improvement: {}'.format(work_sys.trajectory.computation_time[work_sys.sim.t_current], cost_improvement))

    if not work_sys.architecture_limit_check():
        print('\nB: {} | C: {} | max: {} | min: {} | Exit condition: {}'.format(len(work_sys.B.active_set), len(work_sys.C.active_set), work_sys.B.max, work_sys.B.min, exit_condition))
        raise Exception('Architecture limits failed')
    return work_sys


def greedy_rejection(S: System, number_of_changes_limit: int = None, print_check: bool = False):
    exit_condition = 0
    work_sys = dc(S)
    arch_ref = dc(S)
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
            exit_condition = 1
            if print_check:
                print('Rejection exit 1: No available rejections')

        elif len(choices) == 1 and choices[0]['arch'] == 'skip':
            # Exit if the only option is no change
            rejection_check = False
            exit_condition = 2
            if print_check:
                print('Rejection exit 2: No valuable rejections')

        else:
            number_of_choices += len(choices)
            evaluations = cost_mapper(work_iteration, choices)

            index_of_smallest_cost = min(range(len(evaluations)), key=lambda change: evaluations[change].get('cost', float('inf')))
            smallest_cost = evaluations[index_of_smallest_cost]['cost']

            if print_check:
                print('Choice evaluations')
                for i in range(0, len(evaluations)):
                    print('{}: {} | {} | {} | {}'.format(i, evaluations[i]['arch'], evaluations[i]['idx'], evaluations[i]['change'], evaluations[i]['cost']))
                print('Smallest cost: {} @ {}'.format(smallest_cost, index_of_smallest_cost))
                evaluations[index_of_smallest_cost]['system'].architecture_display()

            if evaluations[index_of_smallest_cost]['arch'] == 'skip':
                # Exit if the best option is no change
                rejection_check = False
                exit_condition = 5
                if print_check:
                    print('Rejection exit 5: No valuable rejections')

            else:
                work_sys = dc(evaluations[index_of_smallest_cost]['system'])
                cost_improvement.append(smallest_cost)
                number_of_changes = arch_ref.architecture_compute_active_set_changes(work_sys)

                if number_of_changes_limit is not None and number_of_changes >= number_of_changes_limit:
                    # Exit if maximum number of changes have been completed
                    rejection_check = False
                    exit_condition = 3
                    if print_check:
                        print('Rejection exit 3: Maximum rejections done')

                elif number_of_changes == 0:
                    # Trigger 2 should always activate before this
                    rejection_check = False
                    exit_condition = 4
                    if print_check:
                        print('Rejection exit 4: No valuable rejections')

    # work_sys.trajectory.computation_time[work_sys.sim.t_current] = time.time() - t_start
    if print_check:
        print('Number of iterations: {}'.format(number_of_changes))
        work_sys.architecture_display()
        # print('Computation time: {}\nCost Improvement: {}'.format(work_sys.trajectory.computation_time[work_sys.sim.t_current], cost_improvement))

    if not work_sys.architecture_limit_check():
        print('\nB: {} | C: {} | max: {} | min: {} | Exit condition: {}'.format(len(work_sys.B.active_set), len(work_sys.C.active_set), work_sys.B.max, work_sys.B.min, exit_condition))
        raise Exception('Architecture limits failed')
    return work_sys


def greedy_simultaneous(S: System, number_of_changes_limit: int = None, number_of_changes_per_iteration: int = None, print_check_outer: bool = False, print_check_inner: bool = False):
    work_sys = dc(S)
    ref_arch = dc(S)
    cost_improvement = [work_sys.trajectory.cost.predicted[work_sys.sim.t_current]]
    swap_limit_mod = 1 if number_of_changes_per_iteration is None else number_of_changes_per_iteration
    safety_counter = 0
    exit_condition = 0

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

        force_swap.architecture_limit_mod(min_mod=-swap_limit_mod, max_mod=-swap_limit_mod)
        force_swap = greedy_rejection(force_swap, number_of_changes_limit=2*swap_limit_mod, print_check=print_check_inner)

        cost_improvement.append(force_swap.trajectory.cost.predicted[force_swap.sim.t_current])

        if print_check_outer:
            print('After force rejection')
            force_swap.architecture_display()

        if work_sys.architecture_compare_active_set_to_system(force_swap):      # Comparison of previous to current iteration of architecture update
            simultaneous_check = False
            exit_condition = 1
            if print_check_outer:
                print('Swap exit 1: No more valuable forced swaps')

        else:
            number_of_changes = ref_arch.architecture_compute_active_set_changes(force_swap)    # Comparison of initial reference to current architecture
            work_sys = dc(force_swap)
            safety_counter += 1

            if number_of_changes_limit is not None and number_of_changes >= 2*number_of_changes_limit:
                simultaneous_check = False
                exit_condition = 2
                if print_check_outer:
                    print('Swap exit 2: Maximum swaps done')
            elif number_of_changes == 0:
                simultaneous_check = False
                exit_condition = 3
                if print_check_outer:
                    print('Swap exit 3: No valuable swaps or cyclic swaps')

            if safety_counter > work_sys.number_of_states:
                raise Exception('Triggered safety constraint - too many changes - check system')

    # work_sys.trajectory.computation_time[work_sys.sim.t_current] = time.time() - t_start
    if print_check_outer:
        work_sys.architecture_display()
        # print('Computation time: {}\nCost Improvement: {}'.format(work_sys.trajectory.computation_time[work_sys.sim.t_current], cost_improvement))

    if not work_sys.architecture_limit_check():
        print('\nB: {} | C: {} | max: {} | min: {} | Exit condition: {}'.format(len(work_sys.B.active_set), len(work_sys.C.active_set), work_sys.B.max, work_sys.B.min, exit_condition))
        raise Exception('Architecture limits failed')
    return work_sys


def optimize_initial_architecture(S: System, print_check: bool = False):
    if print_check:
        print('Optimizing design-time architecture from:')
        S.architecture_display()

    t_predict_ref = dc(S.sim.t_predict)
    S.sim.t_predict = 2*t_predict_ref
    S.trajectory.cost.metric_control = 2
    BR3, CR3 = dc(S.B.R3), dc(S.C.R3)
    S.B.R3 *= 0
    S.C.R3 *= 0
    S.prediction_gains()
    S.cost_prediction_wrapper()

    S = greedy_simultaneous(S, print_check_inner=print_check, print_check_outer=print_check)

    S.sim.t_predict = dc(t_predict_ref)
    S.trajectory.cost.metric_control = 1
    S.B.R3 = dc(BR3)
    S.C.R3 = dc(CR3)
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
                t_start = process_time()
                S_fix.cost_true_wrapper()
                S_fix.trajectory.computation_time[S_fix.sim.t_current] = process_time() - t_start
                S_fix.one_step_system_update()
                pbar.update()
    else:
        for _ in range(0, S_fix.sim.t_simulate):
            t_start = process_time()
            S_fix.cost_true_wrapper()
            S_fix.trajectory.computation_time[S_fix.sim.t_current] = process_time() - t_start
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
                t_start = process_time()
                if t > 0:
                    S_self_tuning = greedy_simultaneous(S_self_tuning, number_of_changes_limit=number_of_changes_limit, print_check_outer=print_check)
                S_self_tuning.cost_true_wrapper()
                S_self_tuning.trajectory.computation_time[S_self_tuning.sim.t_current] = process_time() - t_start
                S_self_tuning.one_step_system_update()
                pbar.update()
    else:
        for t in range(0, S_self_tuning.sim.t_simulate):
            t_start = process_time()
            if t > 0:
                S_self_tuning = greedy_simultaneous(S_self_tuning, number_of_changes_limit=number_of_changes_limit, print_check_outer=print_check)
            S_self_tuning.cost_true_wrapper()
            S_self_tuning.trajectory.computation_time[S_self_tuning.sim.t_current] = process_time() - t_start
            S_self_tuning.one_step_system_update()

    if print_check:
        print('Self-Tuning Architecture Simulation: DONE')

    return S_self_tuning


def simulate_experiment_fixed_vs_selftuning(exp_no: int = 1, print_check: bool = False, tqdm_check: bool = True, statistics_model=0):
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

    S.sim.test_parameter = None if S.sim.test_parameter == 0 else S.sim.test_parameter

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

    S.sim.test_parameter = None if S.sim.test_parameter == 0 else S.sim.test_parameter

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

    prediction_scaling = 2 if S.sim.test_parameter is None else S.sim.test_parameter

    S_tuning_baseTp = dc(S)
    S_tuning_baseTp = simulate_self_tuning_architecture(S_tuning_baseTp, number_of_changes_limit=1, print_check=print_check, tqdm_check=tqdm_check)
    S_tuning_baseTp.plot_name = 'selftuning Tp' + str(S_tuning_baseTp.sim.t_predict)

    S_tuning_nTp = dc(S)
    S_tuning_nTp.sim.t_predict *= prediction_scaling
    S_tuning_nTp = simulate_self_tuning_architecture(S_tuning_nTp, number_of_changes_limit=1, print_check=print_check, tqdm_check=tqdm_check)
    S_tuning_nTp.plot_name = 'selftuning Tp' + str(S_tuning_nTp.sim.t_predict)

    if statistics_model == 0:
        system_model_to_memory_sim_model(S, S_tuning_baseTp, S_tuning_nTp)
    else:
        system_model_to_memory_statistics(S, S_tuning_baseTp, S_tuning_nTp, statistics_model)

    return S, S_tuning_baseTp, S_tuning_nTp


def simulate_experiment_selftuning_architecture_cost(exp_no: int = 1, print_check: bool = False, tqdm_check: bool = True, statistics_model=0):
    S = initialize_system_from_experiment_number(exp_no)

    S_tuning_base_cost = dc(S)
    S_tuning_base_cost = optimize_initial_architecture(S_tuning_base_cost, print_check=print_check)
    S_tuning_base_cost = simulate_self_tuning_architecture(S_tuning_base_cost, number_of_changes_limit=1, print_check=print_check, tqdm_check=tqdm_check)
    S_tuning_base_cost.plot_name = 'selftuning base arch cost'

    S_tuning_scale_cost = dc(S)
    cost_scale = 0 if S.sim.test_parameter is None else S.sim.test_parameter
    S_tuning_scale_cost.B.R2 = S.B.R2 * cost_scale
    S_tuning_scale_cost.B.R3 = S.B.R3 * cost_scale
    S_tuning_scale_cost.C.R2 = S.C.R2 * cost_scale
    S_tuning_scale_cost.C.R3 = S.C.R3 * cost_scale
    S_tuning_scale_cost = optimize_initial_architecture(S_tuning_scale_cost, print_check=print_check)
    S_tuning_scale_cost = simulate_self_tuning_architecture(S_tuning_scale_cost, number_of_changes_limit=1, print_check=print_check, tqdm_check=tqdm_check)
    if cost_scale == 0:
        S_tuning_scale_cost.plot_name = 'selftuning free arch'
    else:
        S_tuning_scale_cost.plot_name = 'selftuning ' + str(cost_scale) + 'scale arch cost'

    if statistics_model == 0:
        system_model_to_memory_sim_model(S, S_tuning_base_cost, S_tuning_scale_cost)
    else:
        system_model_to_memory_statistics(S, S_tuning_base_cost, S_tuning_scale_cost, statistics_model)

    return S, S_tuning_base_cost, S_tuning_scale_cost


def simulate_experiment_selftuning_architecture_cost_no_lim(exp_no: int = 1, print_check: bool = False, tqdm_check: bool = True, statistics_model=0):
    S = initialize_system_from_experiment_number(exp_no)

    S_tuning_base_cost = dc(S)
    S_tuning_base_cost = optimize_initial_architecture(S_tuning_base_cost, print_check=print_check)
    S_tuning_base_cost = simulate_self_tuning_architecture(S_tuning_base_cost, number_of_changes_limit=None, print_check=print_check, tqdm_check=tqdm_check)
    S_tuning_base_cost.plot_name = 'selftuning base arch cost'

    S_tuning_scale_cost = dc(S)
    cost_scale = 0 if S.sim.test_parameter is None else S.sim.test_parameter
    S_tuning_scale_cost.B.R2 = S.B.R2 * cost_scale
    S_tuning_scale_cost.B.R3 = S.B.R3 * cost_scale
    S_tuning_scale_cost.C.R2 = S.C.R2 * cost_scale
    S_tuning_scale_cost.C.R3 = S.C.R3 * cost_scale
    S_tuning_scale_cost = optimize_initial_architecture(S_tuning_scale_cost, print_check=print_check)
    S_tuning_scale_cost = simulate_self_tuning_architecture(S_tuning_scale_cost, number_of_changes_limit=None, print_check=print_check, tqdm_check=tqdm_check)
    if cost_scale == 0:
        S_tuning_scale_cost.plot_name = 'selftuning free arch'
    else:
        S_tuning_scale_cost.plot_name = 'selftuning ' + str(cost_scale) + 'scale arch cost'

    if statistics_model == 0:
        system_model_to_memory_sim_model(S, S_tuning_base_cost, S_tuning_scale_cost)
    else:
        system_model_to_memory_statistics(S, S_tuning_base_cost, S_tuning_scale_cost, statistics_model)

    return S, S_tuning_base_cost, S_tuning_scale_cost


def simulate_statistics_experiment(exp_no: int, print_check: bool = False, start_idx: int = 1, number_of_samples: int = 100):
    experiment_function_mapper = {
                                  'statistics_fixed_vs_selftuning'                : simulate_experiment_fixed_vs_selftuning,
                                  'statistics_selftuning_number_of_changes'       : simulate_experiment_selftuning_number_of_changes,
                                  'statistics_selftuning_prediction_horizon'      : simulate_experiment_selftuning_prediction_horizon,
                                  'statistics_selftuning_architecture_cost'       : simulate_experiment_selftuning_architecture_cost,
                                  'statistics_selftuning_architecture_cost_no_lim': simulate_experiment_selftuning_architecture_cost_no_lim,
                                  'statistics_pointdistribution_openloop'         : simulate_experiment_fixed_vs_selftuning_pointdistribution_openloop
                                  }

    S = initialize_system_from_experiment_number(exp_no)
    if S.sim.test_model == 'statistics_pointdistribution_openloop':
        system_model_to_memory_gen_model(S_temp)
        idx_range = list(range(1, 1 + S_temp.number_of_states))
    else:
        idx_range = list(range(start_idx, number_of_samples + start_idx))

    if S.sim.multiprocess_check:
        with tqdm(total=len(idx_range), ncols=100, desc='Model ID', leave=True) as pbar:
            # with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for _ in executor.map(experiment_function_mapper[S.sim.test_model], itertools.repeat(exp_no), itertools.repeat(print_check), itertools.repeat(False), idx_range):
                    pbar.update()
    else:
        for test_no in tqdm(idx_range, desc='Simulations', ncols=100, position=0, leave=True):
            _, _, _ = experiment_function_mapper[S.sim.test_model](exp_no=exp_no, statistics_model=test_no, print_check=print_check, tqdm_check=True)


def simulate_experiment(exp_no: int = None, print_check: bool = False):

    if exp_no is None:
        raise Exception('No experiment number provided')
    else:
        print('Experiment number: ', exp_no)

    experiment_function_mapper = {
                                    'fixed_vs_selftuning'                           : simulate_experiment_fixed_vs_selftuning,
                                    'selftuning_number_of_changes'                  : simulate_experiment_selftuning_number_of_changes,
                                    'selftuning_prediction_horizon'                 : simulate_experiment_selftuning_prediction_horizon,
                                    'selftuning_architecture_cost'                  : simulate_experiment_selftuning_architecture_cost,
                                    'pointdistribution_openloop'                    : simulate_experiment_fixed_vs_selftuning_pointdistribution_openloop,
                                    'selftuning_architecture_cost_no_lim'           : simulate_experiment_selftuning_architecture_cost_no_lim,
                                    'statistics_fixed_vs_selftuning'                : simulate_statistics_experiment,
                                    'statistics_selftuning_number_of_changes'       : simulate_statistics_experiment,
                                    'statistics_selftuning_prediction_horizon'      : simulate_statistics_experiment,
                                    'statistics_selftuning_architecture_cost'       : simulate_statistics_experiment,
                                    'statistics_selftuning_architecture_cost_no_lim': simulate_statistics_experiment,
                                    'statistics_pointdistribution_openloop'         : simulate_statistics_experiment
                                  }

    S = initialize_system_from_experiment_number(exp_no)

    if S.sim.test_model not in experiment_function_mapper:
        raise Exception('Experiment not defined')

    experiment_function_mapper[S.sim.test_model](exp_no=exp_no)


def retrieve_experiment(exp_no: int = 1):
    S = initialize_system_from_experiment_number(exp_no)
    S, S_1, S_2 = system_model_from_memory_sim_model(S.model_name)
    return S, S_1, S_2


def system_model_to_memory_gen_model(S: System):  # Store model generated from experiment parameters
    shelve_filename = datadump_folder_path + 'gen_' + S.model_name
    with shelve.open(shelve_filename, writeback=True) as shelve_data:
        shelve_data['s'] = S
    print('\nShelving gen model: {}'.format(shelve_filename))


def system_model_from_memory_gen_model(model, print_check=False):  # Retrieve model generated from experiment parameters
    shelve_filename = datadump_folder_path + 'gen_' + model
    if print_check:
        print('\nReading gen model: {}'.format(shelve_filename))
    with shelve.open(shelve_filename, flag='r') as shelve_data:
        S = shelve_data['s']
    if not isinstance(S, System):
        raise Exception('System model error')
    return S


def system_model_to_memory_sim_model(S: System, S_1: System, S_2: System):  # Store simulated models
    shelve_filename = datadump_folder_path + 'sim_' + S.model_name
    print('\nShelving sim model: {}'.format(shelve_filename))
    with shelve.open(shelve_filename, writeback=True) as shelve_data:
        shelve_data['s'] = S
        shelve_data['s1'] = S_1
        shelve_data['s2'] = S_2


def system_model_from_memory_sim_model(model):  # Retrieve simulated models
    shelve_filename = datadump_folder_path + 'sim_' + model
    print('\nReading sim model: {}'.format(shelve_filename))
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
        print('\nShelving model: {}'.format(shelve_filename))


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
        print('\nModel read done: {}'.format(shelve_filename))
    return S, S_1, S_2


def plot_experiment(exp_no: int = None):
    if exp_no is None:
        raise Exception('Check experiment number')

    S = initialize_system_from_experiment_number(exp_no)
    print('\nPlotting Experiment No: {}'.format(exp_no))

    if S.sim.test_model in ['fixed_vs_selftuning', 'selftuning_number_of_changes', 'selftuning_prediction_horizon', 'selftuning_architecture_cost', 'selftuning_architecture_cost_no_lim']:
        plot_comparison_exp_no(exp_no)
    elif S.sim.test_model in ['statistics_fixed_vs_selftuning', 'statistics_selftuning_number_of_changes', 'statistics_selftuning_prediction_horizon', 'statistics_selftuning_architecture_cost', 'statistics_pointdistribution_openloop', 'statistics_selftuning_architecture_cost_no_lim']:
        plot_statistics_exp_no(exp_no)
    else:
        raise Exception('Experiment not defined')


def plot_comparison_exp_no(exp_no: int = 1):
    S, S_1, S_2 = retrieve_experiment(exp_no)

    if S_1.plot is None:
        S_1.plot = PlotParameters()

    if S_2.plot is None:
        S_2.plot = PlotParameters()

    fig = plt.figure(figsize=(6, 8), tight_layout=True)
    outer_grid = gs.GridSpec(2, 2, figure=fig, height_ratios=[1, 7], width_ratios=[1, 1])

    ax_exp_legend = fig.add_subplot(outer_grid[0, 0])
    ax_eval = fig.add_subplot(outer_grid[0, 1])

    time_grid = gs.GridSpecFromSubplotSpec(7, 1, subplot_spec=outer_grid[1, :], hspace=0.2, height_ratios=[1, 1, 1, 0.7, 1, 0.7, 0.7])

    ax_cost = fig.add_subplot(time_grid[0, 0])
    ax_state = fig.add_subplot(time_grid[1, 0], sharex=ax_cost)
    ax_B_scatter = fig.add_subplot(time_grid[2, 0], sharex=ax_cost)
    ax_B_count = fig.add_subplot(time_grid[3, 0], sharex=ax_cost)
    ax_C_scatter = fig.add_subplot(time_grid[4, 0], sharex=ax_cost)
    ax_C_count = fig.add_subplot(time_grid[5, 0], sharex=ax_cost)
    ax_compute_time = fig.add_subplot(time_grid[6, 0], sharex=ax_cost)

    S_1.plot_openloop_eigvals(ax_in=ax_eval)

    S_1.plot_cost(ax_in=ax_cost)
    S_2.plot_cost(ax_in=ax_cost, set_details_flag=True)
    ax_cost.tick_params(axis="x", labelbottom=False)

    ax_exp_legend.legend(handles=[mpatches.Patch(color=S_1.plot.plot_parameters[S_1.plot.plot_system]['c'], label=S_1.plot_name),
                                  mpatches.Patch(color=S_2.plot.plot_parameters[S_2.plot.plot_system]['c'], label=S_2.plot_name)],
                         loc='center', ncol=1, title='Experiment No:' + str(exp_no))
    ax_exp_legend.axis('off')

    S_1.plot_states(ax_in=ax_state)
    S_2.plot_states(ax_in=ax_state, set_details_flag=True)
    ax_state.tick_params(axis="x", labelbottom=False)

    S_1.plot_architecture_history(arch='B', ax_in=ax_B_scatter)
    S_2.plot_architecture_history(arch='B', ax_in=ax_B_scatter)
    ax_B_scatter.set_ylabel('Actuator\nPosition\n' + r'$S_t$')
    ax_B_scatter.tick_params(axis="x", labelbottom=False)

    S_1.plot_architecture_history(arch='C', ax_in=ax_C_scatter)
    S_2.plot_architecture_history(arch='C', ax_in=ax_C_scatter)
    ax_C_scatter.set_ylabel('Sensor\nPosition\n' + r'$S_t$' + '\'')
    ax_C_scatter.tick_params(axis="x", labelbottom=False)

    for lim_val in [S.B.min, S.B.max]:
        ax_B_count.axhline(y=lim_val, color='tab:gray', ls='dashdot', alpha=0.5)
        ax_C_count.axhline(y=lim_val, color='tab:gray', ls='dashdot', alpha=0.5)

    S_1.plot_architecture_count(ax_in=ax_B_count, arch='B')
    S_2.plot_architecture_count(ax_in=ax_B_count, arch='B')
    ax_B_count.set_ylabel('Actuator\nCount\n'+r'$|S_t|$')
    ax_B_count.tick_params(axis="x", labelbottom=False)

    S_1.plot_architecture_count(ax_in=ax_C_count, arch='C')
    S_2.plot_architecture_count(ax_in=ax_C_count, arch='C')
    ax_C_count.set_ylabel('Sensor\nCount\n'+r'$|S$' + '\'' + '$_t|$')
    ax_C_count.tick_params(axis="x", labelbottom=False)

    S_1.plot_compute_time(ax_in=ax_compute_time)
    S_2.plot_compute_time(ax_in=ax_compute_time)
    # ax_compute_time.set_yscale('log')
    ax_compute_time.tick_params(axis='y', labelrotation=30)
    ax_compute_time.set_ylabel('Compute\nTime')
    ax_compute_time.set_xlabel(r'Time $t$')

    plt.show()

    save_path = image_save_folder_path + 'exp' + str(exp_no) + '.pdf'
    fig.savefig(save_path, dpi=fig.dpi)
    print('Image saved: {}'.format(save_path))


def element_wise_min_max(v_ref_min, v_ref_max, v):

    v_ret_min = [min(e) for e in zip(v_ref_min, v)]
    v_ret_max = [max(e) for e in zip(v_ref_max, v)]

    return v_ret_min, v_ret_max


def statistics_data_parser(S: System, cost_min, cost_max, compute_time, arch_change, arch_count):

    cost_min, cost_max = element_wise_min_max(cost_min, cost_max, list(itertools.accumulate(S_1.list_from_dict_key_cost(S_1.trajectory.cost.true))))
    compute_time.append(np.average(list(itertools.accumulate(S.list_from_dict_key_cost(S.trajectory.computation_time)))))
    S.architecture_count_number_of_sim_changes()
    S.architecture_active_count()

    arch_change['B'].append(float(S.B.change_count / S.sim.t_simulate))
    arch_change['C'].append(float(S.C.change_count / S.sim.t_simulate))

    arch_count['B'].append(np.average(S.B.active_count))
    arch_count['C'].append(np.average(S.C.active_count))

    return cost_min, cost_max, compute_time, arch_change, arch_count


def plot_statistics_exp_no(exp_no: int = None):

    if exp_no is None:
        raise Exception('Experiment not provided')

    S = initialize_system_from_experiment_number(exp_no)
    sim_range = 100 if S.sim.test_model in ['statistics_fixed_vs_selftuning', 'statistics_selftuning_number_of_changes', 'statistics_selftuning_prediction_horizon', 'statistics_selftuning_architecture_cost'] else S.number_of_states

    fig = plt.figure(tight_layout=True)
    grid_outer = gs.GridSpec(3, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1.5, 2, 1])

    ax_exp_legend = fig.add_subplot(grid_outer[0, 0])
    ax_eigmodes = fig.add_subplot(grid_outer[0, 1])
    ax_cost = fig.add_subplot(grid_outer[1, :])

    grid_architecture = gs.GridSpecFromSubplotSpec(1, 5, subplot_spec=grid_outer[2, :], hspace=0)
    ax_architecture_B_count = fig.add_subplot(grid_architecture[0, 0])
    ax_architecture_C_count = fig.add_subplot(grid_architecture[0, 1], sharey=ax_architecture_B_count)
    ax_architecture_B_change = fig.add_subplot(grid_architecture[0, 2], sharey=ax_architecture_B_count)
    ax_architecture_C_change = fig.add_subplot(grid_architecture[0, 3], sharey=ax_architecture_B_count)
    ax_architecture_compute_time = fig.add_subplot(grid_architecture[0, 4], sharey=ax_architecture_B_count)

    cstyle = ['tab:blue', 'tab:orange', 'black']
    lstyle = ['dashdot', 'dashed']
    mstyle = ['o', '+', 'x']

    cost_min_1, cost_min_2 = np.inf * np.ones(S.sim.t_simulate), np.inf * np.ones(S.sim.t_simulate)
    cost_max_1, cost_max_2 = np.zeros(S.sim.t_simulate), np.zeros(S.sim.t_simulate)
    sample_cost_1, sample_cost_2 = np.zeros(S.sim.t_simulate), np.zeros(S.sim.t_simulate)
    compute_time_1, compute_time_2 = [], []
    sample_eig = np.zeros(S.number_of_states)
    arch_change_1 = {'B': [], 'C': []}
    arch_change_2 = {'B': [], 'C': []}
    arch_count_1 = {'B': [], 'C': []}
    arch_count_2 = {'B': [], 'C': []}

    m1_name, m2_name = '', ''

    sample_ID = np.random.choice(range(1, sim_range + 1))

    for model_no in tqdm(range(1, sim_range + 1), ncols=100, desc='Model ID'):
        S, S_1, S_2 = data_from_memory_statistics(exp_no, model_no)

        cost_min_1, cost_max_1, compute_time_1, arch_change_1, arch_count_1 = statistics_data_parser(S_1, cost_min_1, cost_max_1, compute_time_1, arch_change_1, arch_count_1)
        cost_min_2, cost_max_2, compute_time_2, arch_change_2, arch_count_2 = statistics_data_parser(S_2, cost_min_2, cost_max_2, compute_time_2, arch_change_2, arch_count_2)

        ax_eigmodes.scatter(range(1, S.number_of_states + 1), np.sort(np.abs(S.A.open_loop_eig_vals)), marker=mstyle[0], s=10, color=cstyle[0], alpha=float(1 / S.number_of_states))

        if sample_ID == model_no:
            sample_cost_1 = list(itertools.accumulate(S_1.list_from_dict_key_cost(S_1.trajectory.cost.true)))
            sample_cost_2 = list(itertools.accumulate(S_2.list_from_dict_key_cost(S_2.trajectory.cost.true)))
            sample_eig = np.sort(np.abs(S.A.open_loop_eig_vals))
            m1_name = S_1.plot_name
            m2_name = S_2.plot_name

    ax_exp_legend.legend(handles=[mpatches.Patch(color=cstyle[0], label=r'$M_1$:' + m1_name),
                                  mpatches.Patch(color=cstyle[1], label=r'$M_2$:' + m2_name)],
                         title='Experiment No:' + str(exp_no))
    ax_exp_legend.axis('off')

    ax_cost.fill_between(range(0, S.sim.t_simulate), cost_min_1, cost_max_1, color=cstyle[0], alpha=0.4, linewidth=0)
    ax_cost.fill_between(range(0, S.sim.t_simulate), cost_min_2, cost_max_2, color=cstyle[1], alpha=0.4, linewidth=0)
    ax_cost.plot(range(0, S.sim.t_simulate), sample_cost_1, color=cstyle[2], ls=lstyle[0], linewidth=1)
    ax_cost.plot(range(0, S.sim.t_simulate), sample_cost_2, color=cstyle[2], ls=lstyle[1], linewidth=1)
    ax_cost.legend(handles=[mlines.Line2D([], [], color=cstyle[2], ls=lstyle[0], label='Sample ' + r'$M_1$'),
                            mlines.Line2D([], [], color=cstyle[2], ls=lstyle[1], label='Sample ' + r'$M_2$')],
                   loc='upper left', ncols=2)
    ax_cost.set_yscale('log')
    ax_cost.set_xlabel(r'Time $t$')
    ax_cost.set_ylabel(r'Cost $J_t$')

    ax_eigmodes.scatter(range(1, S.number_of_states + 1), sample_eig, marker=mstyle[2], color=cstyle[2], s=10)
    ax_eigmodes.hlines(1, xmin=1, xmax=S.number_of_states, colors=cstyle[2], ls=lstyle[1])
    # ax_eigmodes.set_xlabel('Mode ' + r'$i$')
    ax_eigmodes.set_ylabel(r'$|\lambda_i(A)|$')
    ax_eigmodes.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False)
    ax_eigmodes.legend(handles=[mlines.Line2D([], [], color=cstyle[2], marker=mstyle[2], linewidth=0, label='Sample'),
                                mlines.Line2D([], [], color=cstyle[0], marker=mstyle[0], linewidth=0, label='Modes')],
                       loc='upper left')

    a1 = ax_architecture_B_change.boxplot([arch_change_2['B'], arch_change_1['B']], labels=[r'$M_2$', r'$M_1$'], vert=False, widths=0.5)
    a2 = ax_architecture_C_change.boxplot([arch_change_2['C'], arch_change_1['C']], labels=[r'$M_2$', r'$M_1$'], vert=False, widths=0.5)
    a3 = ax_architecture_B_count.boxplot([arch_count_2['B'], arch_count_1['B']], labels=[r'$M_2$', r'$M_1$'], vert=False, widths=0.5)
    a4 = ax_architecture_C_count.boxplot([arch_count_2['C'], arch_count_1['C']], labels=[r'$M_2$', r'$M_1$'], vert=False, widths=0.5)
    a5 = ax_architecture_compute_time.boxplot([compute_time_2, compute_time_1], labels=[r'$M_2$', r'$M_1$'], vert=False, widths=0.5)

    for bplot in (a1, a2, a3, a4, a5):
        for patch, color in zip(bplot['medians'], [cstyle[1], cstyle[0]]):
            patch.set_color(color)

    # ax_architecture_C_count.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax_architecture_C_count.locator_params(axis='x', integer=True)
    ax_architecture_B_count.set_xlabel('Avg ' + r'$|S_t|$' + '\nSize')
    ax_architecture_C_count.set_xlabel('Avg ' + r'$|S$' + '\'' + r'$|$' + '\nSize')
    ax_architecture_B_change.set_xlabel('Avg ' + r'$|S_t|$' + '\nChanges')
    ax_architecture_C_change.set_xlabel('Avg ' + r'$|S$' + '\'' + r'$|$' + '\nChanges')
    ax_architecture_compute_time.set_xlabel('Avg Compute \n Time (s)')
    ax_architecture_compute_time.set_xscale('log')
    ax_architecture_compute_time.tick_params(axis='x', labelrotation=30)

    # ax_architecture_B_count.tick_params(axis='x', labelbottom=False, bottom=False)
    # ax_architecture_B_change.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
    ax_architecture_B_change.tick_params(axis='y', labelleft=False, left=False)
    ax_architecture_C_count.tick_params(axis='y', labelleft=False, left=False)
    ax_architecture_C_change.tick_params(axis='y', labelleft=False, left=False)
    ax_architecture_compute_time.tick_params(axis='y', labelleft=False, left=False)

    plt.show()

    save_path = image_save_folder_path + 'exp' + str(exp_no) + '.pdf'
    fig.savefig(save_path, dpi=fig.dpi)
    print('Image saved: {}'.format(save_path))
