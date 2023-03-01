import greedy_architecture_combined as gac
import numpy as np
import shelve

datadump_folderpath = 'C:/Users/kxg161630/Box/KarthikGanapathy_Research/SpeedyGreedyAlgorithm/DataDump/'

if __name__ == "__main__":

    print('Model Generation')

    n = 50
    rho = 2
    Tp = 10
    n_arch = 5

    disturbance_step = 10
    disturbance_number = 20
    disturbance_magnitude = 5

    S = gac.System(graph_model={'number_of_nodes': n, 'rho': rho}, architecture={'rand': n_arch}, simulation_parameters={'T_sim': 100, 'T_predict': Tp})

    S.architecture['B']['max'] = n_arch
    S.architecture['C']['max'] = n_arch
    S.architecture['B']['min'] = n_arch
    S.architecture['C']['min'] = n_arch

    S.architecture['B']['cost']['R2'] = 0
    S.architecture['C']['cost']['R2'] = 0
    S.architecture['B']['cost']['R3'] = 0
    S.architecture['C']['cost']['R3'] = 0

    for i in range(0, S.simulation_parameters['T_sim'], disturbance_step):
        S.noise['noise_sim'][i][np.random.choice(2 * disturbance_number, 10, replace=False)] = disturbance_magnitude

    S.model_rename()

    print('Model name: ', S.model_name)

    shelve_filename = datadump_folderpath + 'gen_' + S.model_name
    shelve_data = shelve.open(shelve_filename)
    shelve_data['System'] = S
    shelve_data.close()

    print('Model shelve done: ', shelve_filename)
