import matplotlib.pyplot as plt
import numpy as np

import greedy_architecture_combined as gac

if __name__ == "__main__":
    S = gac.System(graph_model={'number_of_nodes': 5, 'rho': 1.1}, architecture={'rand': 2})
    S.display_active_architecture()
    # for a in ['B', 'C']:
    #     print(a)
    #     print('Active:', S.architecture[a]['active'])
    #     print('Indicator:', S.architecture[a]['indicator'])
    #     print('Matrix:', S.architecture[a]['matrix'])
    # for i in range(0, 1000):
    # print(i)
    # S = gac.System(graph_model={'number_of_nodes': 5, 'rho': 1.1}, architecture={'rand': 1})
    # S.optimal_estimation_feedback_wrapper()
    # S.optimal_control_feedback_wrapper()
    # S.enhanced_system_matrix()
    # S.enhanced_scipylyapunov_wrapper()
    # if np.min(np.linalg.eigvals(S.trajectory['P'])) <= 0 or np.min(np.linalg.eigvals(S.trajectory['E'])) <= 0 or np.min(np.linalg.eigvals(S.trajectory['P_enhanced'])) <= 0:
    #     print('n_unstable:', S.dynamics['n_unstable'])
    #     S.display_active_architecture()
    #     print(np.linalg.eigvals(S.trajectory['P']))
    #     print(np.linalg.eigvals(S.trajectory['E']))
    #     print(np.linalg.eigvals(S.trajectory['P_enhanced']))
    #     break
    # S.cost_wrapper_enhanced_prediction()
    # print(S.trajectory['P'])
    # print(S.architecture['B']['gain'])
    # print(S.architecture['B']['matrix'])
    # print(S.architecture['B']['matrix']@S.architecture['B']['gain'])
    # plt.imshow(S.dynamics['enhanced'])
    # plt.colorbar()
    # plt.show()
    # print(S.dynamics['enhanced'])

    # print(S.architecture['B']['cost'])
    # print(S.architecture['C']['cost'])

    # S.cost_wrapper('estimate')
    #
    # # for i in S.trajectory:
    # #     print(i, S.trajectory[i])
    #
    greedy = gac.greedy_architecture_selection(S, no_select=False, status_check=True)
    # greedy = gac.greedy_architecture_rejection(S, no_reject=False, status_check=True)

    # greedy = gac.greedy_simultaneous(S, iterations=None, changes_per_iteration=1)#, status_check=True)
    # gac.simultaneous_cost_plot(greedy['value_history'])

    # for i in greedy['value_history']:
    #     print(i)
    greedy['work_set'].display_active_architecture()
