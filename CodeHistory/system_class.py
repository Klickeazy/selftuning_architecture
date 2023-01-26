import numpy as np
import networkx as netx
import time
from copy import deepcopy as dc


class System:
    def __init__(self, number_of_nodes=10, graph_model=None, rho=1, number_of_actuators=None, number_of_sensors=None,
                 initial_conditions=None, disturbances=None, mpl_matrices=None, costs=None, architecture_limits=None):
        self.number = {'nodes': dc(number_of_nodes)}
        self.A = np.zeros((self.number['nodes'], self.number['nodes']))
        if graph_model is None:
            graph_model = {'type': 'rand', 'self_loop': True}
        self.graph_initialize(graph_model)
        self.A *= rho

        self.architecture = {'B': {}, 'C': {}}

        self.architecture['B']['available'] = np.linspace(0, self.number['nodes'] - 1, num=self.number['nodes'] - 1,
                                                          dtype=int)
        if number_of_actuators is None:
            number_of_actuators = 0
        self.number['actuators'] = number_of_actuators
        self.architecture['B']['active'] = np.random.choice(self.architecture['B']['available'],
                                                            size=self.number['actuators'], replace=False)
        self.architecture['B']['matrix'] = np.zeros((self.number['nodes'], self.number['nodes']))
        self.architecture['B']['history'] = [self.architecture['B']['active']]
        self.architecture['B']['min'] = 0
        self.architecture['B']['max'] = self.number['nodes']

        self.architecture['C']['available'] = np.linspace(0, self.number['nodes'] - 1, num=self.number['nodes'] - 1,
                                                          dtype=int)
        if number_of_sensors is None:
            number_of_sensors = 0
        self.number['sensors'] = number_of_sensors
        self.architecture['C']['active'] = np.random.choice(self.architecture['C']['available'],
                                                            size=self.number['sensors'], replace=False)
        self.architecture['C']['matrix'] = np.zeros((self.number['nodes'], self.number['nodes']))
        self.architecture['C']['history'] = [self.architecture['C']['active']]
        self.architecture['C']['min'] = 0
        self.architecture['C']['max'] = self.number['nodes']

        self.matrix_from_active()

        if architecture_limits is not None:
            for i in architecture_limits:
                for j in i:
                    self.architecture[i][j] = architecture_limits[i][j]

        if initial_conditions is None:
            self.metric_model = 'eigP'
            self.x0 = np.zeros((self.number['nodes'], 1))
            self.X0 = np.zeros((self.number['nodes'], self.number['nodes']))
        elif 'x0' in initial_conditions:
            self.metric_model = 'x0'
            self.x0 = dc(initial_conditions['x0'])
            self.X0 = np.zeros((self.number['nodes'], self.number['nodes']))
        elif 'X0' in initial_conditions:
            self.metric_model = 'X0'
            self.x0 = np.zeros((self.number['nodes'], 1))
            self.X0 = dc(initial_conditions['X0'])
        else:
            raise Exception('Check metric parameters')

        self.number['alphai'] = 0
        self.number['betaj'] = 0
        self.number['gammak'] = 0
        self.disturbances = {'W': np.zeros((self.number['nodes'], self.number['nodes'])),
                             'V': np.zeros((self.number['nodes'], self.number['nodes'])),
                             'alphai': np.zeros(self.number['alphai']), 'betaj': np.zeros(self.number['betaj']),
                             'gammak': np.zeros(self.number['gammak'])}
        if disturbances is not None:
            for i in disturbances:
                self.disturbances[i] = dc(disturbances[i])
                if i in self.number:
                    self.number[i] = len(self.disturbances[i])
        self.Ai = np.zeros((self.number['nodes'], self.number['nodes'], self.number['alphai']))
        self.Bj = np.zeros((self.number['nodes'], self.number['nodes'], self.number['betaj']))
        self.Ck = np.zeros((self.number['nodes'], self.number['nodes'], self.number['gammak']))
        if mpl_matrices is not None:
            if 'Ai' in mpl_matrices:
                self.Ai = mpl_matrices['Ai']
            if 'Bj' in mpl_matrices:
                self.Bj = mpl_matrices['Bj']
            if 'Ck' in mpl_matrices:
                self.Ck = mpl_matrices['Ck']

        self.cost = {'B': {'Q': np.identity(self.number['nodes']), 'R1': np.identity(self.number['nodes']),
                           'R2': np.identity(self.number['nodes']), 'R3': np.identity(self.number['nodes'])},
                     'C': {'Q': np.identity(self.number['nodes']), 'R1': np.identity(self.number['nodes']),
                           'R2': np.identity(self.number['nodes']), 'R3': np.identity(self.number['nodes'])}}
        if costs is not None:
            for architecture_type in costs:
                for i in architecture_type:
                    if i in self.cost[architecture_type]:
                        self.cost[architecture_type][i] = costs[architecture_type][i]

    def architecture_limits(self, architecture_type='B', algorithm='select'):
        limit = {'min': False, 'max': False}
        if algorithm == 'select':
            limit['min'] = len(self.architecture[architecture_type]['active']) >= self.architecture[architecture_type]['min']
            limit['max'] = len(self.architecture[architecture_type]['active']) <= self.architecture[architecture_type]['max']
        elif algorithm == 'reject':
            limit['min'] = len(self.architecture[architecture_type]['active']) >= self.architecture[architecture_type]['min']+1
            limit['max'] = len(self.architecture[architecture_type]['active']) <= self.architecture[architecture_type]['max']+1
        else:
            raise Exception('Limit error')
        return limit

    def graph_initialize(self, graph_model):
        connected_network_check = False
        G = netx.Graph()
        while not connected_network_check:
            if graph_model['type'] == 'ER':
                if 'p' not in graph_model:
                    graph_model['p'] = 0.4
                G = netx.generators.random_graphs.erdos_renyi_graph(self.number['nodes'], graph_model['p'])
            elif graph_model['type'] == 'BA':
                if 'p' not in graph_model:
                    graph_model['p'] = self.number['nodes']//2
                G = netx.generators.random_graphs.barabasi_albert_graph(self.number['nodes'], graph_model['p'])
            elif graph_model['type'] == 'rand':
                A = np.random.rand(self.number['nodes'], self.number['nodes'])
                G = netx.from_numpy_matrix(A)
            elif graph_model['type'] == 'path':
                G = netx.generators.classic.path_graph(self.number['nodes'])
            else:
                raise Exception('Check graph model')
            if netx.algorithms.components.is_connected(G):
                connected_network_check = False

        Adj = netx.to_numpy_array(G)
        if graph_model['self_loop']:
            Adj += np.identity(self.number['nodes'])
        e = np.max(np.abs(np.linalg.eigvals(Adj)))
        self.A = Adj/e