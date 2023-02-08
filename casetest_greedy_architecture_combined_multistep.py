import greedy_architecture_combined as gac
import numpy as np
from copy import deepcopy as dc

if __name__ == "__main__":
    # S = gac.System(graph_model={'type': 'ER', 'rho': 0.8}, architecture={'rand': True})
    S = gac.System(graph_model={'number_of_nodes': 30, 'rho': 1.05}, architecture={'rand': 2})
    T_sim = 50

    print('\n Fixed architecture')
    S_fixed = dc(S)
    for t in range(0, T_sim):
        print("\r t:" + str(t), end="")
        S_fixed.cost_wrapper_enhanced_true()
        S_fixed.system_one_step_update_enhanced()
    # S_fixed.display_system()

    print('\n Optimal architecture')
    S_opt = dc(S)
    for t in range(0, T_sim):
        print("\r t:" + str(t), end="")
        S_opt.cost_wrapper_enhanced_true()
        S_opt.system_one_step_update_enhanced()
        S_opt = dc(gac.greedy_simultaneous(S_opt, iterations=1, changes_per_iteration=1)['work_set'])
    # S_opt.display_system()

    print('\n')
    S_opt.architecture_history_plot()
    gac.cost_plots([S_fixed.trajectory['cost']['true'], S_opt.trajectory['cost']['true']], "_rho"+str(S_opt.dynamics['rho']))
