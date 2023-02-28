import greedy_architecture_combined as gac
import os
import numpy as np
from copy import deepcopy as dc
import shelve

datadump_folderpath = 'C:/Users/kxg161630/Box/KarthikGanapathy_Research/SpeedyGreedyAlgorithm/DataDump/'

if __name__ == "__main__":

    print('Reading generated model...')
    n = 50
    rho = 2
    Tp = 10

    model_name = 'gen_model_n'+str(n)+'_rho'+str(rho)+'_Tp'+str(Tp)
    shelve_file = datadump_folderpath + model_name
    shelve_data = shelve.open(shelve_file)
    S = shelve_data['System']
    shelve_data.close()
    if not isinstance(S, gac.System):
        raise Exception('System model error')

    # Optimal T_sim update:
    # S.simulation_parameters['T_predict'] = 30
    # S.model_rename()

    print('Retrieved Model: ', S.model_name)
    T_sim = dc(S.simulation_parameters['T_sim']) + 1

    # Simulating regular large disturbances at random nodes
    for i in range(0, T_sim, 10):
        S.noise['noise_sim'][i][np.random.choice(2*n, 5)] = 20

    # print('Model: ', S.model_name)

    print('\n Fixed architecture simulation')
    S_fixed = dc(S)
    S_fixed.model_rename(S.model_name + "_fixed")
    for t in range(0, T_sim):
        print("\r t:" + str(t), end="")
        S_fixed.cost_wrapper_enhanced_true()
        S_fixed.system_one_step_update_enhanced(t)

    print('\n Self-Tuning architecture simulation')
    S_tuning = dc(S)
    S_tuning.model_rename(S.model_name + "_selftuning")
    for t in range(0, T_sim):
        print("\r t:" + str(t), end="")
        S_tuning.cost_wrapper_enhanced_true()
        S_tuning.system_one_step_update_enhanced(t)
        S_tuning = dc(gac.greedy_simultaneous(S_tuning, iterations=1, changes_per_iteration=1)['work_set'])

    print('\n\n Trajectory data shelving')

    shelve_file = datadump_folderpath + S.model_name
    print('Deleting old data...')
    for f_type in ['.bak', '.dat', '.dir']:
        if os.path.exists(shelve_file + f_type):
            os.remove(shelve_file + f_type)

    print('Shelving new data...')
    shelve_data = shelve.open(shelve_file)
    for k in ['System', 'Fixed', 'SelfTuning']:
        if k in shelve_data:
            del shelve_data[k]
    shelve_data['System'] = S
    shelve_data['Fixed'] = S_fixed
    shelve_data['SelfTuning'] = S_tuning
    shelve_data.close()

    print('Shelving done:', shelve_file)

    print('\n Done')
