import greedy_architecture_combined as gac
import numpy as np

if __name__ == "__main__":

    n_samples = 100

    # print('Model Generation Parameters')

    n = 50
    rho = 5
    Tp = 10
    n_arch = 5
    n_arch_B = n_arch
    n_arch_C = n_arch

    # test_model = 'combined'
    test_model = None

    disturbance_step = 10
    disturbance_number = int(np.floor(n/2))
    disturbance_magnitude = 20
    disturbance = {'step': disturbance_step, 'number': disturbance_number, 'magnitude': disturbance_magnitude}

    s = 1
    fail_count = 0
    while s <= n_samples:
        try:
            print('Sample s: ', str(s))
            # Model Gen
            S = gac.System(graph_model={'number_of_nodes': n, 'rho': rho}, architecture={'rand': n_arch}, additive={'type': test_model, 'disturbance': disturbance}, simulation_parameters={'T_sim': 100, 'T_predict': Tp})

            # Architecture selection parameters
            S.architecture_limit_modifier(min_mod=n_arch-1, max_mod=-n+n_arch)
            # Architecture selection costs
            S.architecture_cost_update({'R2': 0, 'R3': 0})

            S.model_rename()

            S_fixed = gac.simulate_fixed_architecture(S, print_check=False)
            S_tuning = gac.simulate_selftuning_architecture(S, print_check=False)
            if s == 1:
                gac.statistics_shelving_initialize(S.model_name)
            gac.data_shelving_statistics(S, S_fixed, S_tuning, s)
            s += 1
        except Exception as e:
            print(e)
            print('Fail at s: ', str(s))
            fail_count += 1

    print('Fail count: ', str(fail_count))
