import time
import numpy as np
from copy import deepcopy as dc
from networks_model import System
from architecture_costs import total_control_cost


def initialize_greedy(total_set, work_set, fixed_set, failure_set):
    if not isinstance(total_set, System) or not isinstance(work_set, System) or not (isinstance(fixed_set, System) or fixed_set is None) or not (isinstance(failure_set, System) or failure_set is None):
        raise Exception('Incorrect data type')

    work_iteration = dc(work_set)
    available_set = dc(total_set)
    if failure_set is not None:
        work_iteration = work_iteration.compare_lists(failure_set)['unique']
        available_set = available_set.compare_lists(failure_set)['unique']
    return work_iteration, available_set


def max_limit_System(work_set, architecture_type, algorithm):
    return work_set.architecture_limits()


def min_limit_System(work_set, limit, algorithm):
    return len(work_set.items) >= limit


def item_index_from_policy(values, policy):
    if policy == "max":
        return values.index(max(values))
    elif policy == "min":
        return values.index(min(values))
    else:
        raise Exception('Check policy')
    
    
def greedy_selection(total_set, work_set, sys, architecture_type='B', number_of_changes=None, fixed_set=None, failure_set=None, max_greedy_limit=max_limit_System, min_greedy_limit=min_limit_System, cost_metric=total_control_cost, policy="min", t_start=time.time(), no_select=False, status_check=False):
    work_iteration, available_set = initialize_greedy(total_set, work_set, fixed_set, failure_set)
    choice_history = []
    work_history = []
    value_history = []
    count_of_changes = 0
    while max_greedy_limit(work_iteration, architecture_type, algorithm='selection'):
        work_history.append(dc(work_iteration))
        choice_iteration = available_set.compare_lists(work_iteration)['unique']
        if fixed_set is not None:
            choice_iteration = fixed_set.compare_lists(choice_iteration)['absent']
        choice_history.append(choice_iteration)
        if len(choice_iteration.items) == 0:
            if status_check:
                print('No selections possible')
            break
        iteration_cases = []
        values = []
        if no_select and min_greedy_limit(work_iteration, architecture_type, algorithm='selection'):
            iteration_cases.append(dc(work_iteration))
            values.append(cost_metric(iteration_cases[-1]))
        for i in range(0, len(choice_iteration.items)):
            iteration_cases.append(dc(work_iteration))
            iteration_cases[-1].add_architecture(choice_iteration.items[i])
            values.append(cost_metric(iteration_cases[-1]))
        value_history.append(values)
        target_idx = item_index_from_policy(values, policy)
        work_iteration = dc(iteration_cases[target_idx])
        if len(work_iteration.compare_lists(work_history[-1])['unique'].items) == 0:
            if status_check:
                print('No valuable selections')
            break
        count_of_changes += 1
        if number_of_changes is not None and count_of_changes == number_of_changes:
            if status_check:
                print('Maximum number of changes done')
            break
    work_history.append(work_iteration)
    work_iteration.list_id = "Greedy Selection"
    return {'work_set': work_iteration, 'work_history': work_history, 'choice_history': choice_history, 'value_history': value_history, 'time': time.time()-t_start}

