import greedy_architecture_combined as gac
import numpy as np
from copy import deepcopy as dc

if __name__ == "__main__":
    # S = gac.System(graph_model={'type': 'ER', 'rho': 0.8}, architecture={'rand': True})
    S = gac.System(graph_model={'number_of_nodes': 30, 'rho': 1.05}, architecture={'rand': 2})
    T_sim = 20

    print('\n Fixed architecture')
    S_fixed = dc(S)
    S_fixed.model_name = "fixed"+S_fixed.model_name
    for t in range(0, T_sim):
        print("\r t:" + str(t), end="")
        S_fixed.cost_wrapper_enhanced_true()
        S_fixed.system_one_step_update_enhanced()
    # S_fixed.display_system()

    print('\n Optimal architecture')
    S_tuning = dc(S)
    S_tuning.model_name = "selftuning" + S_tuning.model_name
    for t in range(0, T_sim):
        print("\r t:" + str(t), end="")
        S_tuning.cost_wrapper_enhanced_true()
        S_tuning.system_one_step_update_enhanced()
        S_tuning = dc(gac.greedy_simultaneous(S_tuning, iterations=1, changes_per_iteration=1)['work_set'])
    # S_tuning.display_system()

    print('\n')
    S_tuning.plot_architecture_history()
    S_fixed.plot_trajectory_history()
    S_tuning.plot_trajectory_history()
    gac.cost_plots([S_fixed.trajectory['cost']['true'], S_tuning.trajectory['cost']['true']], S_tuning.model_name)
