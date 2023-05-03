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

image_save_folder_path = '../Images/'


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


class System:

    class Dynamics:
        def __init__(self, initialize_check=False):
            self.number_of_nodes = 20
            self.number_of_states = 20
            self.network_model = 'rand'
            self.network_model_parameter = None
            self.rho = 1
            self.self_loop = True

            self.second_order = False

            self.second_order_factor = 1
            self.second_order_type = 1

            self.open_loop_eig_vals = np.zeros(self.number_of_nodes)
            self.adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes))
            self.number_of_non_stable_modes = 0
            self.A = np.zeros((self.number_of_nodes, self.number_of_nodes))

            if initialize_check:
                self.initialize_dynamics()

        def initialize_dynamics(self):
            self.adjacency_matrix_initialize()
            self.rescale_wrapper()
            if self.second_order:
                self.number_of_states = self.number_of_nodes * 2
                self.second_order_matrix()

        def adjacency_matrix_initialize(self):
            if self.network_model not in ['rand', 'ER', 'BA', 'path', 'cycle', 'eval_squeeze', 'eval_bound']:
                raise Exception('Network model not defined')

            connected_graph_check = False
            G = netx.Graph()

            while not connected_graph_check:

                if self.network_model == 'ER':
                    if self.network_model_parameter is None:
                        self.network_model_parameter = 0.3
                    elif self.network_model_parameter < 0 or self.network_model_parameter > 1:
                        raise Exception('Check network model parameter')
                    G = netx.generators.random_graphs.erdos_renyi_graph(self.number_of_nodes, self.network_model_parameter)

                elif self.network_model == 'BA':
                    if self.network_model_parameter is None:
                        self.network_model_parameter = 2
                    elif self.network_model_parameter <= 0 or self.network_model_parameter >= self.number_of_nodes:
                        raise Exception('Check network model parameter')
                    G = netx.generators.random_graphs.barabasi_albert_graph(self.number_of_nodes, self.network_model_parameter)

                elif self.network_model == 'rand':
                    self.network_model_parameter = None
                    A = np.random.rand(self.number_of_nodes, self.number_of_nodes)
                    G = netx.from_numpy_array(A)

                elif self.network_model == 'path':
                    G = netx.generators.classic.path_graph(self.number_of_nodes)

                elif self.network_model == 'cycle':
                    G = netx.generators.classic.cycle_graph(self.number_of_nodes)

                elif self.network_model in ['eval_squeeze', 'eval_bound']:
                    if self.network_model_parameter is None:
                        self.network_model_parameter = 0.1
                    elif self.network_model_parameter < 0 or self.network_model_parameter > 1:
                        raise Exception('Check network model parameter')

                    A = np.random.rand(self.number_of_nodes, self.number_of_nodes)
                    A = A.T + A
                    _, V_mat = np.linalg.eig(A)

                    if self.network_model == 'eval_squeeze':
                        e = 2 * self.network_model_parameter * (0.5 - np.random.default_rng().random(self.number_of_nodes))
                    elif self.network_model == 'eval_bound':
                        e = 1 - self.network_model_parameter * np.random.default_rng().random(self.number_of_nodes)
                    else:
                        raise Exception('Check Network model')

                    self.self_loop = None
                    self.rho = None

                    e = np.array([i*-1 if coin_toss() else i for i in e])
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
                if self.second_order_type == 1:
                    self.A = np.block([
                        [self.A, np.zeros_like(self.A)],
                        [self.second_order_factor * np.identity(self.number_of_nodes), self.second_order_factor * np.identity(self.number_of_nodes)]
                    ])
                elif self.second_order_type == 2:
                    self.A = np.block([
                        [np.identity(self.number_of_nodes), np.zeros_like(self.A)],
                        [self.second_order_factor * self.A, self.second_order_factor * np.identity(self.number_of_nodes)]
                    ])
                else:
                    raise SecondOrderError()

        def rescale_wrapper(self):
            self.rescale()
            self.second_order_matrix()
            self.evaluate_modes()

    class Architecture:
        def __init__(self, architecture_type='B'):

            if architecture_type in ['B', 'C']:
                self.architecture_type = architecture_type
            else:
                raise ArchitectureError

            self.second_order_architecture_type = None

            self.available_vectors = {}

            self.number_of_available = 0

            self.min = 1
            self.max = 0

            self.Q = 0
            self.R1_reference = 1
            self.R1 = 0
            self.R2 = 0
            self.R3 = 0

            self.available_indices = []

            self.active_set = []

            self.active_matrix = 0
            self.initialize_active_matrix()

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

        def architecture_duplicate_active_inner(self, reference_architecture):
            # Needs pair outer function for each architecture type
            if not isinstance(reference_architecture, System.Architecture):
                raise ClassError

            self.active_set = dc(reference_architecture.active_set)
            self.architecture_update_wrapper_from_active_set()

        def initialize_random_architecture_active_set(self, initialize_random):
            self.active_set = np.random.default_rng().choice(self.available_indices, size=initialize_random, replace=False)

    class Disturbance:
        def __init__(self, number_of_nodes, scaling_W=1, scaling_V=1, t_simulate=100, noise_model=None):
            self.number_of_nodes = number_of_nodes
            self.W = np.identity(self.number_of_nodes) * scaling_W
            self.V = np.identity(self.number_of_nodes) * scaling_V
            self.w_gen = []
            self.v_gen = []
            self.generate_noise_matrices(t_simulate)
            self.noise_model = noise_model
            if self.noise_model is not None:
                self.noise_modification(t_simulate)

        def generate_noise_matrices(self, t_simulate):
            self.w_gen = np.random.default_rng().multivariate_normal(np.zeros(self.number_of_nodes), self.W, t_simulate)
            self.v_gen = np.random.default_rng().multivariate_normal(np.zeros(self.number_of_nodes), self.V, t_simulate)

        def noise_modification(self, t_simulate, disturbance_nodes=None, disturbance_scaling=50, disturbance_steps=None):
            if disturbance_nodes is None:
                disturbance_nodes = np.random.default_rng().choice(self.number_of_nodes, self.number_of_nodes//2, replace=False)
            if disturbance_steps is None:
                disturbance_steps = 20

            if self.noise_model not in ['combined', 'process', 'measurement', None]:
                raise Exception('Invalid noise model')

            if self.noise_model in ['combined', 'process']:
                self.w_gen[disturbance_nodes, :][:, [k for k in range(0, t_simulate) if k % disturbance_steps == 0]] *= disturbance_scaling

            if self.noise_model in ['combined', 'measurement']:
                self.v_gen[:, [k for k in range(0, t_simulate) if k % disturbance_steps == 0]] *= disturbance_scaling

    class Simulation:
        def __init__(self):
            self.t_simulate = 100
            self.t_predict = 10
            self.model = None

    def __init__(self, parameter_set=None):

        if parameter_set is None:
            self.dynamics = self.Dynamics()
            self.B = self.Architecture('B')
            self.C = self.Architecture('C')
            self.simulation_parameters = self.Simulation()

        self.disturbance = self.Disturbance(self.dynamics.number_of_nodes, self.simulation_parameters.t_simulate)

    def initialize_dynamics(self, number_of_nodes=20, available_set_vectors=None, second_order_architecture_type=None, initialize_random=None):

        if self.available_vectors is None:
            self.initialize_available_vectors_as_basis_vectors()

        self.number_of_available = len(self.available_vectors)
        self.available_indices = [k for k in self.available_vectors]

        self.R1_reference = np.identity(self.number_of_available)
        self.R1 = np.zeros_like(self.R1_reference)

        if initialize_random is None:
            initialize_random = int(self.number_of_available // 10)

        if initialize_random != 0:
            self.initialize_random_architecture(initialize_random)


























def coin_toss():
    return np.random.default_rng().random() > 0


if __name__ == "__main__":
    print('Function File check done')
