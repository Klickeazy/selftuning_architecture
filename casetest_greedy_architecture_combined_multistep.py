import greedy_architecture_combined as gac
import numpy as np
from copy import deepcopy as dc
import shelve

if __name__ == "__main__":
    n = 50
    rho = 1.05
    S = gac.System(graph_model={'type': 'ER', 'number_of_nodes': n, 'rho': rho}, architecture={'rand': 10})
    S.simulation_parameters = {'T_sim': 200, 'T_predict': 30}
    S.architecture['B']['max'] = 20
    S.architecture['C']['max'] = 20
    T_sim = dc(S.simulation_parameters['T_sim'])

    print('Model: ', S.model_name)

    print('\n Fixed architecture')
    S_fixed = dc(S)
    S_fixed.model_name = "fixed"+S_fixed.model_name
    for t in range(0, T_sim):
        print("\r t:" + str(t), end="")
        S_fixed.cost_wrapper_enhanced_true()
        S_fixed.system_one_step_update_enhanced()

    print('\n Self-Tuning architecture')
    S_tuning = dc(S)
    S_tuning.model_name = "selftuning" + S_tuning.model_name
    for t in range(0, T_sim):
        print("\r t:" + str(t), end="")
        S_tuning.cost_wrapper_enhanced_true()
        S_tuning.system_one_step_update_enhanced()
        S_tuning = dc(gac.greedy_simultaneous(S_tuning, iterations=1, changes_per_iteration=1)['work_set'])

    print('\n\n Data shelving')
    shelve_data = shelve.open('DataDumps/comparison_fixed_vs_selftuning_n'+str(n)+'_rho'+str(rho))
    for k in ['System', 'Fixed', 'SelfTuning']:
        if k in shelve_data:
            del shelve_data[k]
    shelve_data['System'] = S
    shelve_data['Fixed'] = S_fixed
    shelve_data['SelfTuning'] = S_tuning
    shelve_data.close()

    print('\n Done')
