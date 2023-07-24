from CodeHistory import greedy_architecture_combined as gac
import numpy as np

if __name__ == "__main__":

    print('Generating Model')

    n = 50
    Tp = 10
    n_arch = 3
    n_arch_B = n_arch
    n_arch_C = n_arch

    network_model = 'eval_squeeze'
    # network_model = 'eval_bound'
    # network_model = 'rand'
    # rho = 3
    rho = None
    p = 0.1
    # second_order = True
    second_order = False

    test_model = 'combined'  #  'process', 'sensor', 'combined', None
    # sim_model = None
    disturbance_step = 15
    disturbance_number = int(np.floor(n / 2))
    disturbance_magnitude = 100
    disturbance = {'step': disturbance_step, 'number': disturbance_number, 'magnitude': disturbance_magnitude}

    # Model Gen
    S = gac.System(graph_model={'number_of_nodes': n, 'type': network_model, 'p': p, 'rho': rho, 'second_order': second_order}, architecture={'rand': n_arch}, additive={'type': test_model, 'disturbance': disturbance, 'W': 1, 'V': 1}, simulation_parameters={'T_sim': 100, 'T_predict': Tp, 'model': None})
    # print(S.model_name)

    # for k in ['B', 'C']:
    #     for l in ['min', 'max']:
    #         print('k: ', k, '|l: ', l, ' : ', S.architecture[k][l])

    # Architecture selection parameters
    S.architecture_limit_modifier(min_mod=n_arch-1, max_mod=-n+n_arch)

    # print(S.additive['W'])
    # print(S.additive['V'])

    # Architecture selection costs
    S.architecture_cost_update({'R2': 0, 'R3': 0})
    # S.architecture['B']['cost']['R2'] = 100
    # S.architecture['C']['cost']['R2'] = 0
    # S.architecture['B']['cost']['R3'] = 100
    # S.architecture['C']['cost']['R3'] = 0

    S.model_rename()

    # print('Unstable modes: ', S.dynamics['n_unstable'])
    # print('Eig Vals: ', S.dynamics['ol_eig'])

    print('Optimizing fixed architecture')
    S_init = gac.greedy_architecture_initialization(S, True)
    S.active_architecture_duplicate(S_init)

    # print(S.dynamics['A'])

    # for k in ['B', 'C']:
    #     for l in ['min', 'max']:
    #         print('k: ', k, '|l: ', l, ' : ', S.architecture[k][l])

    print('Model: ', S.model_name)

    gac.data_shelving_gen_model(S)
