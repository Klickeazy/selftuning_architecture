import networkx as nx
import numpy as np
import networkx as netx
import random

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
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

import time
from copy import deepcopy as dc


class System:
    def __init__(self, number_of_nodes=None, graph_model=None, rho=1, architecture=None, mpl=None):
        if number_of_nodes is None:
            number_of_nodes = 10
        self.dynamics = {'number': number_of_nodes}
        self.dynamics['A'] = np.zeros((self.dynamics['number'], self.dynamics['number']))
        if graph_model is None:
            graph_model = {'type': 'rand', 'self_loop': True}
        self.graph_initialize(graph_model)
        self.dynamics['A'] *= rho

        self.architecture = {'B': {}, 'C': {}}
        keys = ['B', 'C']
        if architecture is not None:
            if 'B' in architecture:
                self.architecture['B'] = architecture['B']
                keys = [i for i in keys if i != 'B']
            if 'C' in architecture:
                self.architecture['C'] = architecture['C']
                keys = [i for i in keys if i != 'C']
        self.initialize_architecture(keys)

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
            self.architecture[i]['active'] = random.sample(self.architecture[i]['available'], k=(self.architecture[i]['min']+self.architecture[i]['max'])//2)
        self.architecture_active_to_matrix(architecture_type)

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

    def initialize_architecture(self, keys):
        for i in keys:
            self.architecture[i] = {'min': 0, 'max': self.dynamics['number'],
                                    'cost': {'Q': np.identity(self.dynamics['number']),
                                             'R1': np.identity(self.dynamics['number']),
                                             'R2': np.identity(self.dynamics['number']),
                                             'R3': np.identity(self.dynamics['number'])},
                                    'active': [],
                                    'matrix': np.zeros((self.dynamics['number'], self.dynamics['number'])),
                                    'available': range(0, self.dynamics['number']),
                                    'set': []}
        self.initialize_architecture_basis_vectors(keys)
        self.random_architecture(keys)

    def initialize_architecture_basis_vectors(self, keys):
        basis = []
        for i in range(0, self.dynamics['number']):
            basis.append(np.zeros(self.dynamics['number']))
            basis[-1][i] = 1
        if 'B' in keys:
            self.architecture['B']['set'] = basis
        if 'C' in keys:
            self.architecture['C']['set'] = [k.T for k in basis]

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
            node_pos = nx.circular_layout(G_base)
        node_pos = nx.spring_layout(G, pos=node_pos, fixed=[str(i+1) for i in range(0, self.dynamics['number'])])
        return {'G': G, 'pos': node_pos, 'node_color': nc}


def animate_architecture(S, architecture_history):
    # fig = plt.figure(figsize=(6, 4))
    # gs = GridSpec(1, 1, figure=fig)
    # ax1 = fig.add_subplot(gs[0, 0])
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    ax1.axis('off')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    Sys_dummy = dc(S)
    Sys_dummy.architecture['B']['active'] = Sys_dummy.architecture['B']['available']
    Sys_dummy.architecture['C']['active'] = Sys_dummy.architecture['C']['available']
    Sys_dummy.architecture_active_to_matrix()
    pos_gen = Sys_dummy.display_graph_gen()['pos']
    # print(pos_gen)

    node_pos = {i: pos_gen[i] for i in pos_gen if i.isnumeric()}

    def update(t):
        ax1.clear()
        sys_t = dc(S)
        for i in architecture_history[t]:
            sys_t.architecture[i]['active'] = dc(architecture_history[t][i]['active'])
        sys_t.architecture_active_to_matrix()
        sys_t_plot = sys_t.display_graph_gen(node_pos)
        netx.draw_networkx(sys_t_plot['G'], ax=ax1, pos=sys_t_plot['pos'], node_color=sys_t_plot['node_color'])
        ax1.set_title('t='+str(t))
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        # print(ax1.get_xlim())
        # print(ax1.get_ylim())

    ani = matplotlib.animation.FuncAnimation(fig, update, frames=np.arange(0,len(architecture_history),1), interval=1000, repeat=False)
    # plt.show()
    ani.save("Test.mp4")
