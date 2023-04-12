import greedy_architecture_combined as gac
from copy import deepcopy as dc

if __name__ == "__main__":

    n = 30
    rho = 3
    Tp = 10
    n_arch = 3

    # test_model = 'combined'
    test_model = None

    # second_order = True
    second_order = False

    model = gac.model_namer(n, rho, Tp, n_arch, test_model, second_order)
    S = gac.data_reading_gen_model(model)
    S.display_active_architecture()

    print('Retrieved model: ', S.model_name)

    # # Model update:
    # S.simulation_parameters['T_predict'] = 30
    S.rescale_dynamics(6)
    # Architecture selection costs
    # S.architecture_cost_update({'R2': 0, 'R3': 0})
    # S.architecture['B']['cost']['R2'] = 10000
    # S.architecture['C']['cost']['R2'] = 0
    # S.architecture['B']['cost']['R3'] = 10000
    # S.architecture['C']['cost']['R3'] = 0
    S.model_rename()

    print('Simulating Model: ', S.model_name)
    print('Number of unstable modes: ', S.dynamics['n_unstable'])
    # print('Optimizing design-time architecture')
    # S = dc(gac.greedy_simultaneous(S)['work_set'])
    # S.display_active_architecture()

    S_fixed = gac.simulate_fixed_architecture(S)
    if test_model == 'unlimited_arch_change':
        S_tuning = gac.simulate_selftuning_architecture(S, iterations_per_step=n_arch, changes_per_iteration=n_arch)
    else:
        S_tuning = gac.simulate_selftuning_architecture(S)

    gac.data_shelving_sim_model(S, S_fixed, S_tuning)
    print('\n Done')
