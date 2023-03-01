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
    n_arch = 5

    model_name = 'gen_model_n'+str(n)+'_rho'+str(rho)+'_Tp'+str(Tp)+'_arch'+str(n_arch)
    shelve_file = datadump_folderpath + model_name
    shelve_data = shelve.open(shelve_file)
    S = shelve_data['System']
    shelve_data.close()
    if not isinstance(S, gac.System):
        raise Exception('System model error')

    # # Update T_sim update:
    # S.simulation_parameters['T_predict'] = 30
    S.rescale_dynamics(1.5)
    S.model_rename()

    print('Retrieved Model: ', S.model_name)

    S_fixed = gac.simulate_fixed_architecture(S)
    S_tuning = gac.simulate_selftuning_architecture(S)

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
