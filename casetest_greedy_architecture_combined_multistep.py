import greedy_architecture_combined as gac
from copy import deepcopy as dc

if __name__ == "__main__":

    n = 50
    rho = 6
    Tp = 10
    n_arch = 5

    # test_model = 'combined'
    test_model = None

    # second_order = False
    second_order = True

    model = gac.model_namer(n, rho, Tp, n_arch, test_model, second_order)
    S = gac.data_reading_gen_model(model)

    print('Retrieved model: ', S.model_name)

    # # Model update:
    # S.simulation_parameters['T_predict'] = 30
    # S.rescale_dynamics(7)
    # S.model_rename()

    print('Optimizing design-time architecture')
    S = dc(gac.greedy_simultaneous(S)['work_set'])

    print('Number of unstable modes: ', S.dynamics['n_unstable'])
    print('Simulating Model: ', S.model_name)

    S_fixed = gac.simulate_fixed_architecture(S)
    if test_model == 'unlimited_arch_change':
        S_tuning = gac.simulate_selftuning_architecture(S, iterations_per_step=n_arch, changes_per_iteration=n_arch)
    else:
        S_tuning = gac.simulate_selftuning_architecture(S)

    gac.data_shelving_sim_model(S, S_fixed, S_tuning)
    print('\n Done')
