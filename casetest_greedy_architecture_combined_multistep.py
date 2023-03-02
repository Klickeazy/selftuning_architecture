import greedy_architecture_combined as gac

if __name__ == "__main__":

    n = 10
    rho = 1.05
    Tp = 10
    n_arch = 2

    S = gac.data_reading_gen_model(n, rho, Tp, n_arch)

    # # Model update:
    # S.simulation_parameters['T_predict'] = 30
    # S.rescale_dynamics(1.5)

    S.model_rename()
    print('Retrieved Model: ', S.model_name)

    S_fixed = gac.simulate_fixed_architecture(S)
    S_tuning = gac.simulate_selftuning_architecture(S)

    gac.data_shelving_sim_model(S, S_fixed, S_tuning)
    print('\n Done')
