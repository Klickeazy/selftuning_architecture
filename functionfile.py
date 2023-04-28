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

image_save_folder_path = 'Images/'


class System:

    class Dynamics:
        def __init__(self, number_of_nodes=20, network_model='rand', network_model_parameter=None, rho=1, self_loop=True, second_order=False):
            self.number_of_nodes = number_of_nodes
            self.network_model = network_model
            self.network_model_parameter = network_model_parameter
            self.rho = rho
            self.self_loop = self_loop
            self.second_order = second_order

            self.second_order_factor = 1
            self.second_order_type = 1

            self.open_loop_eig_vals = np.zeros(self.number_of_nodes)
            self.adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes))
            self.number_of_non_stable_modes = 0
            self.A = np.zeros((self.number_of_nodes, self.number_of_nodes))

            self.adjacency_matrix_initialize()

            if self.second_order:
                self.number_of_nodes *= 2

            self.rescale_wrapper()

        def adjacency_matrix_initialize(self):
            if self.network_model not in ['rand', 'ER', 'BA', 'path', 'cycle', 'eval_squeeze', 'eval_bound']:
                raise Exception('Network model not defined')

            connected_graph_check = False

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

                    e = np.array([i*-1 if coin_toss() else i for i in e])
                    A = V_mat @ np.diag(e) @ np.linalg.inv(V_mat)
                    G = netx.from_numpy_array(A)

                    self.self_loop = None
                    self.rho = None

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
                    raise Exception('Check second order type')

        def rescale_wrapper(self):
            self.rescale()
            self.second_order_matrix()
            self.evaluate_modes()

    class Architecture:
        def __init__(self, number_of_nodes=20, available_set_vectors=None, second_order=False):
            self.min = 1
            self.max = number_of_nodes
            self.Q = np.identity(number_of_nodes)
            self.R1 = np.identity(number_of_nodes)
            self.R2 = 0
            self.R3 = 0
            self.active_set = []
            self.active_matrix = np.zeros((number_of_nodes, number_of_nodes))
            self.indicator_vector = np.zeros(number_of_nodes)
            self.available_vectors = available_set_vectors
            self.available = range(0, len(self.available_set_vectors))
            self.history = []
            self.change_count = 0
            self.gain = {}

            if self.available_vectors is None:
                self.initialize_architecture_set_as_basis_vectors(number_of_nodes)

        def initialize_architecture_set_as_basis_vectors(self, number_of_nodes, second_order):







    def __init__(self):

        self.dynamics = self.Dynamics
        self.B = {}

def coin_toss():
    return np.random.default_rng().random() > 0