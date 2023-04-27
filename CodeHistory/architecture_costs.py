import numpy as np
import networkx as netx
import time
from networks_model import System
from copy import deepcopy as dc


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


def greedy_selection(S, architecture_type='B', number_of_changes=None, policy="max", t_start=time.time(), no_select=False, print_check=False):
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
            if print_check:
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
            if print_check:
                print('No valuable selections')
            break
        count_of_changes += 1
        if number_of_changes is not None and count_of_changes == number_of_changes:
            if print_check:
                print('Maximum number of changes done')
            break
    work_history.append(work_iteration)
    return {'work_set': work_iteration, 'work_history': work_history, 'choice_history': choice_history, 'value_history': value_history, 'time': time.time()-t_start}


def greedy_rejection(S, architecture_type='B', number_of_changes=None, policy="max", t_start=time.time(), no_reject=False, print_check=False):
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
            if print_check:
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
            if print_check:
                print('No valuable rejections')
            break
        count_of_changes += 1
        if number_of_changes is not None and count_of_changes == number_of_changes:
            if print_check:
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