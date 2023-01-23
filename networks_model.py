import numpy as np
import networkx as netx
import time
from copy import deepcopy as dc


class System:
    def __init__(self, number_of_nodes=10, graph_model=None, rho=1, number_of_actuators=None, number_of_sensors=None, initial_conditions=None, disturbances=None, mpl_matrices=None, costs=None, architecture_limits=None):
        self.number = {'nodes': dc(number_of_nodes)}
        self.A = np.zeros((self.number['nodes'], self.number['nodes']))
        if graph_model is None:
            graph_model = {'type': 'rand', 'self_loop': True}
        self.graph_initialize(graph_model)
        self.A *= rho

        self.architecture = {'B': {}, 'C': {}}
        
        self.architecture['B']['available'] = np.linspace(0, self.number['nodes']-1, num=self.number['nodes']-1, dtype=int)
        if number_of_actuators is None:
            number_of_actuators = 0
        self.number['actuators'] = number_of_actuators
        self.architecture['B']['active'] = np.random.choice(self.architecture['B']['available'], size=self.number['actuators'], replace=False)
        self.architecture['B']['matrix'] = np.zeros((self.number['nodes'], self.number['nodes']))
        self.architecture['B']['history'] = [self.architecture['B']['active']]
        self.architecture['B']['min'] = 0
        self.architecture['B']['max'] = self.number['nodes']

        self.architecture['C']['available'] = np.linspace(0, self.number['nodes']-1, num=self.number['nodes']-1, dtype=int)
        if number_of_sensors is None:
            number_of_sensors = 0
        self.number['sensors'] = number_of_sensors
        self.architecture['C']['active'] = np.random.choice(self.architecture['C']['available'], size=self.number['sensors'], replace=False)
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
        self.disturbances = {'W': np.zeros((self.number['nodes'], self.number['nodes'])), 'V': np.zeros((self.number['nodes'], self.number['nodes'])), 'alphai': np.zeros(self.number['alphai']), 'betaj': np.zeros(self.number['betaj']), 'gammak': np.zeros(self.number['gammak'])}
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

        self.cost = {'B': {'Q': np.identity(self.number['nodes']), 'R1': np.identity(self.number['nodes']), 'R2': np.identity(self.number['nodes']), 'R3': np.identity(self.number['nodes'])}, 'C': {'Q': np.identity(self.number['nodes']), 'R1': np.identity(self.number['nodes']), 'R2': np.identity(self.number['nodes']), 'R3': np.identity(self.number['nodes'])}}
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

    def evaluate_cost_matrix(self, P):
        if self.metric_model == 'eigP':
            return np.max(np.linalg.eigvals(P))
        elif self.metric_model == 'x0':
            return self.x0.T @ P @ self.x0
        elif self.metric_model == 'X0':
            return np.matrix.trace(P @ self.X0)
        else:
            raise Exception('Check Metric')

    def matrix_from_active(self, architecture=None):
        if architecture is None:
            architecture = ['B', 'C']
        for architecture_type in architecture:
            self.architecture[architecture_type]['matrix'] = np.zeros((self.number['nodes'], self.number['nodes']))
            for i in range(0, len(self.architecture[architecture_type]['active'])):
                self.architecture[architecture_type]['matrix'][self.architecture[architecture_type]['active'], i] = 1

    def add_architecture(self, node, architecture_type='B', status_check=False):
        if node in self.architecture[architecture_type]['available'] and node not in self.architecture[architecture_type]['active']:
            self.architecture[architecture_type]['active'] = np.append(self.architecture[architecture_type]['active'], node)
            self.matrix_from_active([architecture_type])
        elif status_check:
            print('Element not found')

    def remove_architecture(self, node, architecture_type='B', status_check=False):
        if node in self.architecture[architecture_type]['active']:
            self.architecture[architecture_type]['active'] = np.delete(self.architecture[architecture_type]['active'], np.where(self.architecture[architecture_type]['active'] == node))
            self.matrix_from_active([architecture_type])
        elif status_check:
            print('Element not found')


def total_control_cost(S, T_horizon=30):
    if not isinstance(S, System):
        raise Exception('Incorrect data type')
    feedback_costs = control_feedback(S, T_horizon=T_horizon)
    cost = 0
    cost += S.evaluate_cost_matrix(feedback_costs['P'][-1])
    cost += architecture_cost(S, architecture_type='B')
    return cost


def control_feedback(S, T_horizon=30, converge_accuracy=10**(-3)):
    if not isinstance(S, System):
        raise Exception('Incorrect data type')
    P = dc(S.cost['B']['Q'])
    P_history = [P]
    K_history = []
    convergence_check = False
    T_run = 0
    for t in range(0, T_horizon):
        P, K = iteration_control_cost(S, P)
        P_history.append(P)
        K_history.append(K)
        convergence_check = matrix_convergence_check(P_history[-1], P_history[-2], accuracy=converge_accuracy)
        if convergence_check:
            T_run = t
            break
    return {'P': P_history, 'K': K_history, 't_run': T_run, 'convergence_check': convergence_check}


def iteration_control_cost(S, P):
    K_mat = iteration_control_feedback(P)
    P_mat = np.zeros_like(S.A)
    P_mat += S.cost['B']['Q'] + S.A.T @ P @ S.A
    if S.number['alphai'] > 0:
        for i in range(0, S.number['alphai']):
            P_mat += S.disturbances['alphai'][i]*(S.Ai[:, :, i].T @ P @ S.Ai[:, :, i])
    P_mat += S.A.T@P@S.architecture['B']['matrix']@K_mat
    return P_mat, K_mat


def iteration_control_feedback(S, P):
    if not isinstance(S, System):
        raise Exception('Incorrect data type')
    K_mat = S.cost['B']['R1'] + S.architecture['B']['matrix'].T @ P @ S.architecture['B']['matrix']
    if S.number['betaj'] > 0:
        for j in range(0, S.number['betaj']):
            K_mat += S.disturbances['betaj'][j] * (S.Bj[:, :, j].T @ P @ S.Bj[:, :, j])
    K_mat = -K_mat @ S.architecture['B']['matrix'].T @ P @ S.A
    return K_mat


def architecture_cost(S, architecture_type='B'):
    cost_architecture = 0
    cost_architecture += S.architecture[architecture_type]['active'].T @ S.cost[architecture_type]['R2'] @ S.architecture[architecture_type]['active']
    cost_architecture += (S.architecture[architecture_type]['active'] - S.architecture[architecture_type]['history'][-2]).T @ S.cost[architecture_type]['R3'] @ (S.architecture[architecture_type]['active'] - S.architecture[architecture_type]['history'][-2])
    return cost_architecture


def compare_architecture(S1, S2=None, architecture_type='B'):
    if not isinstance(S1, System):
        raise Exception('Incorrect data type')
    compare = {}
    if S2 is None:
        compare['choices'] = np.array([])
        for i in S1.architecture[architecture_type]['available']:
            if i not in S1.architecture[architecture_type]['active']:
                compare['choices'] = np.append(compare['choices'], i)
    else:
        if not isinstance(S2, System):
            raise Exception('Incorrect data type')
        compare['added'] = np.array([])
        compare['common'] = np.array([])
        compare['removed'] = np.array([])
        for i in np.array(range(0, S1.number['nodes'])):
            if i in S1.architecture[architecture_type]['active'] and i in S2.architecture[architecture_type]['active']:
                compare['common'] = np.append(compare['common'], i)
            elif i in S1.architecture[architecture_type]['active']:
                compare['added'] = np.append(compare['added'], i)
            elif i in S2.architecture[architecture_type]['active']:
                compare['removed'] = np.append(compare['removed'], i)
    return compare


def cost_calculation(S, architecture_type='B'):
    if architecture_type == 'B':
        return total_control_cost(S)
    # elif architecture_type == 'C':
    #     return total_estimation_cost(S)
    else:
        raise Exception('Check architecture type')


def item_index_from_policy(values, policy):
    if policy == "max":
        return values.index(max(values))
    elif policy == "min":
        return values.index(min(values))
    else:
        raise Exception('Check policy')


def greedy_selection(S, architecture_type='B', number_of_changes=None, policy="max", t_start=time.time(), no_select=False, status_check=False):
    if not isinstance(S, System):
        raise Exception('Incorrect data type')
    choice_history = []
    work_history = []
    value_history = []
    count_of_changes = 0
    work_iteration = dc(S)
    limit = work_iteration.architecture_limits(architecture_type=architecture_type, algorithm='select')
    while limit['max']:
        work_history.append(dc(work_iteration))
        choice_iteration = compare_architecture(work_iteration, architecture_type=architecture_type)['choices']
        choice_history.append(choice_iteration)
        if len(choice_iteration) == 0:
            if status_check:
                print('No selections possible')
            break
        iteration_cases = []
        values = []
        if no_select and limit['min']:
            iteration_cases.append(dc(work_iteration))
            values.append(cost_calculation(iteration_cases[-1], architecture_type))
        for i in range(0, len(choice_iteration)):
            iteration_cases.append(dc(work_iteration))
            iteration_cases[-1].add_architecture(choice_iteration[i], architecture_type)
            values.append(cost_calculation(iteration_cases[-1], architecture_type))
        value_history.append(values)
        target_idx = item_index_from_policy(values, policy)
        work_iteration = dc(iteration_cases[target_idx])
        limit = work_iteration.architecture_limits(architecture_type=architecture_type, algorithm='select')
        if len(compare_architecture(work_iteration, work_history[-1])['added']) == 0:
            if status_check:
                print('No valuable selections')
            break
        count_of_changes += 1
        if number_of_changes is not None and count_of_changes == number_of_changes:
            if status_check:
                print('Maximum number of changes done')
            break
    work_history.append(work_iteration)
    return {'work_set': work_iteration, 'work_history': work_history, 'choice_history': choice_history, 'value_history': value_history, 'time': time.time()-t_start}


def greedy_rejection(S, architecture_type='B', number_of_changes=None, policy="max", t_start=time.time(), no_reject=False, status_check=False):
    if not isinstance(S, System):
        raise Exception('Incorrect data type')
    choice_history = []
    work_history = []
    value_history = []
    count_of_changes = 0
    work_iteration = dc(S)
    limit = work_iteration.architecture_limits(architecture_type=architecture_type, algorithm='reject')
    while limit['min']:
        work_history.append(dc(work_iteration))
        choice_iteration = work_iteration.architecture[architecture_type]['active']
        choice_history.append(choice_iteration)
        if len(choice_iteration) == 0:
            if status_check:
                print('No rejections possible')
            break
        iteration_cases = []
        values = []
        if no_reject and limit['max']:
            iteration_cases.append(dc(work_iteration))
            values.append(cost_calculation(iteration_cases[-1], architecture_type))
        for i in range(0, len(choice_iteration)):
            iteration_cases.append(dc(work_iteration))
            iteration_cases[-1].remove_architecture(choice_iteration[i], architecture_type)
            values.append(cost_calculation(iteration_cases[-1], architecture_type))
        value_history.append(values)
        target_idx = item_index_from_policy(values, policy)
        work_iteration = dc(iteration_cases[target_idx])
        limit = work_iteration.architecture_limits(architecture_type=architecture_type, algorithm='reject')
        if len(compare_architecture(work_iteration, work_history[-1])['removed']) == 0:
            if status_check:
                print('No valuable rejections')
            break
        count_of_changes += 1
        if number_of_changes is not None and count_of_changes == number_of_changes:
            if status_check:
                print('Maximum number of changes done')
            break
    work_history.append(work_iteration)
    return {'work_set': work_iteration, 'work_history': work_history, 'choice_history': choice_history, 'value_history': value_history, 'time': time.time()-t_start}


def matrix_convergence_check(A, B, accuracy=10**(-3), check_type=None):
    np_norm_methods = ['inf', 'fro', 2, None]
    if check_type is None:
        return np.allclose(A, B, a_tol=accuracy, r_tol=accuracy)
    elif check_type in np_norm_methods:
        return np.norm(A-B, ord=check_type) < accuracy
    else:
        raise Exception('Check Matrix Convergence')


if __name__ == '__main__':
    print('Main code complete')
