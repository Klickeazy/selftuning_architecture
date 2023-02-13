import greedy_architecture_combined as gac
import numpy as np
from copy import deepcopy as dc


for i in range(0, 1000):
    print('i:', i)
    S = gac.System(graph_model={'number_of_nodes': 10, 'rho': 5}, architecture={'rand': 1})
    # print(np.shape(S.architecture['C']['matrix']))
    # print(np.shape(S.architecture['C']['cost']['R1']))

    S.optimal_estimation_feedback_wrapper()
    S.optimal_control_feedback_wrapper()
    S.enhanced_system_matrix()

    if np.max(np.abs(np.linalg.eigvals(S.dynamics['enhanced']))) >= 1:
        print('A:', np.sort(np.linalg.eigvals(S.dynamics['A'])))
        S.display_active_architecture()
        print('Unstable:', np.sort(np.linalg.eigvals(S.dynamics['enhanced'])))
        print('Control:', np.sort(np.linalg.eigvals(S.dynamics['A']-(S.architecture['B']['matrix']@S.architecture['B']['gain']))))
        # print('Estimation A-LCA:', np.sort(np.linalg.eigvals(S.dynamics['A'] - (S.architecture['C']['gain'] @ S.architecture['C']['matrix']@S.dynamics['A']))))
        print('Estimation (I-LC)A:', np.sort(np.linalg.eigvals((np.identity(S.dynamics['number_of_nodes']) - (S.architecture['C']['gain'] @ S.architecture['C']['matrix']))@S.dynamics['A'])))
        raise Exception('Check exception')

print('Done')
