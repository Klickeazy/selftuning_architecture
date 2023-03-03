import greedy_architecture_combined as gac
import numpy as np

if __name__ == "__main__":

    print('Model Generation Parameters')

    n = 20
    rho = 2
    Tp = 10
    n_arch = 5

    disturbance_step = 10
    disturbance_number = 2*n
    disturbance_magnitude = 20

    # Model Gen
    S = gac.System(graph_model={'number_of_nodes': n, 'rho': rho, 'type': 'ER'}, architecture={'rand': n_arch}, simulation_parameters={'T_sim': 100, 'T_predict': Tp})

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
    for i in range(0, S.simulation_parameters['T_sim'], disturbance_step):
        S.noise['noise_sim'][i][np.random.choice(2 * S.dynamics['number_of_nodes'], disturbance_number, replace=False)] = disturbance_magnitude

    S.model_rename()
    print('Model: ', S.model_name)

    gac.data_shelving_gen_model(S)
