import numpy as np
from copy import deepcopy as dc


def experiment_model(exp_no=0):

    network_model_choices = ['ER', 'BA', 'rand', 'rand_eval', 'path', 'cycle']
    test_model_choices = [None, 'combined', 'process', 'sensor']
    second_order_choices = {'check': [False, True],
                            'A_type': [1, 2], 'B_type': [1, 2], 'C_type': [1, 2]}

    experiments = [
        {},
        {'number_of_nodes': 30},  # 1
        {'number_of_nodes': 50},  # 2
        {'second_order': {'check': second_order_choices['check'][1]}},  # 3
        {'test_model': test_model_choices[1]},  # 4
        {'network_model': network_model_choices[3]}  # 5
    ]

    return experiments[exp_no]


def default_experiment():
    parameters = {'number_of_nodes': 20,
                  'network_model': 'rand',
                  'rho': 3,
                  'p': 0.05,
                  'rand_arch': 3,
                  'test_model': None,
                  'second_order': {'check': False, 'A_type': 1, 'B_type': 1, 'C_type': 2},
                  'disturbance': {'check': False},
                  'W': 1,
                  'V': 1,
                  'sim_model': None,
                  'T_sim': 100,
                  'Tp': 10}
    parameters = build_disturbance_model(parameters)
    return parameters


def build_disturbance_model(parameters, step=None, number=None, magnitude=None):
    if parameters['disturbance']['check']:
        parameters['step'] = step if step is not None else 15
        parameters['number'] = number if number is not None else int(np.floor(parameters['number_of_nodes'] / 2))
        parameters['magnitude'] = magnitude if magnitude is not None else 10
    return parameters


def build_experiment_parameters(exp_no=None):
    experiment_parameters = default_experiment() if exp_no is None else iterative_update(default_experiment(), experiment_model(exp_no))
    experiment_parameters = build_disturbance_model(experiment_parameters)
    return experiment_parameters


def iterative_update(base_params, update_params):
    new_params = dc(base_params)
    for key in new_params:
        if key in update_params:
            if type(update_params[key]) is dict and type(new_params[key]) is dict:
                new_params[key] = iterative_update(new_params[key], update_params[key])
            elif new_params[key] is dict:
                new_params[key] = update_params
            else:
                new_params[key] = update_params[key]
    return new_params
