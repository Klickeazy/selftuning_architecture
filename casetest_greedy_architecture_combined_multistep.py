import greedy_architecture_combined as gac
import os
import numpy as np
from copy import deepcopy as dc
import shelve

if __name__ == "__main__":
    n = 30
    rho = 1.05
    Tp = 30

    S = gac.System(graph_model={'number_of_nodes': n, 'rho': rho}, architecture={'rand': 3}, simulation_parameters={'T_sim': 50, 'T_predict': Tp})
    S.architecture['B']['max'] = n
    S.architecture['C']['max'] = n
    T_sim = dc(S.simulation_parameters['T_sim'])+1

    print('Model: ', S.model_name)

    print('\n Fixed architecture')
    S_fixed = dc(S)
    S_fixed.model_name = S_fixed.model_name + "_fixed"
    for t in range(0, T_sim):
        print("\r t:" + str(t), end="")
        S_fixed.cost_wrapper_enhanced_true()
        S_fixed.system_one_step_update_enhanced()

    print('\n Self-Tuning architecture')
    S_tuning = dc(S)
    S_tuning.model_name = S_tuning.model_name + "_selftuning"
    for t in range(0, T_sim):
        print("\r t:" + str(t), end="")
        S_tuning.cost_wrapper_enhanced_true()
        S_tuning.system_one_step_update_enhanced()
        S_tuning = dc(gac.greedy_simultaneous(S_tuning, iterations=1, changes_per_iteration=1)['work_set'])

    print('\n\n Data shelving')
    shelve_file = 'DataDumps/comparison_fixed_vs_selftuning_' + S.model_name
    print(shelve_file)
    for f_type in ['.bak', '.dat', '.dir']:
        if os.path.exists(shelve_file + f_type):
            os.remove(shelve_file + f_type)

    shelve_data = shelve.open(shelve_file)
    for k in ['System', 'Fixed', 'SelfTuning']:
        if k in shelve_data:
            del shelve_data[k]
    shelve_data['System'] = S
    shelve_data['Fixed'] = S_fixed
    shelve_data['SelfTuning'] = S_tuning
    shelve_data.close()

    print('\n Done')
