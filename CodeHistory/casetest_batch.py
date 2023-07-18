from CodeHistory import greedy_architecture_combined as gac
import numpy as np

if __name__ == "__main__":

    print('Model Generation Parameters')

    n = 30
    rho = 0.5
    Tp = 10
    n_arch = 5

    test_model = 'process_noise'

    disturbance_step = 10
    disturbance_number = 0
    disturbance_magnitude = 20

    # Model Gen
    S = gac.System(graph_model={'number_of_nodes': n, 'rho': rho, 'type': 'ER'}, architecture={'rand': n_arch}, simulation_parameters={'T_sim': 100, 'T_predict': Tp})
    S.model_rename(test_model)

    # Architecture selection parameters
    S.architecture['B']['max'] = n_arch
    S.architecture['C']['max'] = n_arch
    S.architecture['B']['min'] = n_arch
    S.architecture['C']['min'] = n_arch

    # Architecture selection costs
    S.architecture['B']['cost']['R2'] = 0
    S.architecture['C']['cost']['R2'] = 0
    S.architecture['B']['cost']['R3'] = 0
    S.architecture['C']['cost']['R3'] = 0

    # Noise reshaping
    if test_model in ['process', 'sensor', 'combined']:
        for i in range(0, S.simulation_parameters['T_sim'], disturbance_step):
            if test_model in ['process', 'combined']:
                S.noise['noise_sim'][i][np.random.choice(S.dynamics['number_of_nodes'], disturbance_number,
                                                         replace=False)] = disturbance_magnitude
            if test_model in ['sensor', 'combined']:
                S.noise['noise_sim'][i][S.dynamics['number_of_nodes']:] = disturbance_magnitude

    S.model_rename(test_model)

    print('Model: ', S.model_name)

    gac.data_shelving_gen_model(S)

    S_fixed = gac.simulate_fixed_architecture(S)
    # S_tuning = gac.simulate_selftuning_architecture(S, iterations_per_step=n_arch, changes_per_iteration=n_arch)
    S_tuning = gac.simulate_selftuning_architecture(S)

    gac.data_shelving_sim_model(S, S_fixed, S_tuning)

    # gac.slider_plot(S, S_fixed, S_tuning)
