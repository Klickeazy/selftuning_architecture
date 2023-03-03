import greedy_architecture_combined as gac

if __name__ == "__main__":

    n = 30
    rho = 4
    Tp = 10
    n_arch = 6

    test_model = 'combined'
    # test_model = None

    model = gac.model_name(n, rho, Tp, n_arch, test_model)
    S = gac.data_reading_gen_model(model)

    print('Retrieved model: ', S.model_name)

    # # Model update:
    # S.simulation_parameters['T_predict'] = 30
    # S.rescale_dynamics(4)

    # test_model = 'unlimited_arch_change'
    # test_model = None
    S.model_rename(test_model)
    print('Simulating Model: ', S.model_name)

    S_fixed = gac.simulate_fixed_architecture(S)
    if test_model == 'unlimited_arch_change':
        S_tuning = gac.simulate_selftuning_architecture(S, iterations_per_step=n_arch, changes_per_iteration=n_arch)
    else:
        S_tuning = gac.simulate_selftuning_architecture(S)

    gac.data_shelving_sim_model(S, S_fixed, S_tuning)
    print('\n Done')
