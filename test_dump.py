import greedy_architecture_combined as gac
import shelve
import numpy as np
from copy import deepcopy as dc
import control

# S = gac.System(graph_model={'number_of_nodes': 5, 'rho': 1.05}, architecture={'rand': 2})
# S.feedback_computations()
# S.enhanced_system_matrix()
# S.enhanced_lyapunov_control_cost()
# print(S.trajectory['cost']['predicted'])
# S.cost_wrapper_enhanced_prediction()
# print(S.trajectory['cost']['predicted'])

# for k in S.dynamics['enhanced']:
#     print(k)
# print(len(S.dynamics['enhanced_stage_cost']))
# print(len(S.noise['enhanced_noise_matrix']))

# S.gramian_wrapper()
# S.optimal_control_feedback_wrapper()
# S.optimal_estimation_feedback_wrapper()
# print(np.shape(S.architecture['B']['gain']))
# print(np.shape(S.architecture['C']['gain']))
# print('Ctrb_gram:', np.sort(np.linalg.eigvals(S.architecture['B']['gram'])))
# print('Obsv_gram:', np.sort(np.linalg.eigvals(S.architecture['C']['gram'])))
# S_sys = control.StateSpace(S.dynamics['A'], S.architecture['B']['matrix'], S.architecture['C']['matrix'], np.zeros((len(S.architecture['C']['active']), len(S.architecture['B']['active']))), dt=True)
# print(control.isdtime(S_sys))
# print(control.gram(S_sys, 'o'))
# print(control.gram(S_sys, 'c'))


# S.optimal_estimation_feedback_wrapper()
# S.optimal_control_feedback_wrapper()
# S.enhanced_system_matrix()
# print(np.shape(S.architecture['B']['gain']))
# print(np.shape(S.dynamics['A'] @ S.architecture['C']['gain']))
# print(np.linalg.matrix_rank(control.ctrb(S.dynamics['A'], S.architecture['B']['matrix'])))
# print(np.linalg.matrix_rank(control.obsv(S.dynamics['A'], S.architecture['C']['matrix'])))

# # Iterate over generations
# n = 10
# count = 0
# for i in range(0, 1000):
#     print('i:', i)
#     S = gac.System(graph_model={'number_of_nodes': n, 'rho': 0.8}, architecture={'rand': 1})
#     # if n != np.linalg.matrix_rank(control.ctrb(S.dynamics['A'], S.architecture['B']['matrix'])) or n != np.linalg.matrix_rank(control.obsv(S.dynamics['A'], S.architecture['C']['matrix'])):
#     #     raise Exception('Not Ctrb/Obsv')
#     # print(np.shape(S.architecture['C']['matrix']))
#     # print(np.shape(S.architecture['C']['cost']['R1']))
#
#     S.optimal_estimation_feedback_wrapper()
#     S.optimal_control_feedback_wrapper()
#     S.enhanced_system_matrix()
#
#     S_sys = control.StateSpace(S.dynamics['A'], S.architecture['B']['matrix'], S.architecture['C']['matrix'], np.zeros((len(S.architecture['C']['active']), len(S.architecture['B']['active']))), dt=True)
#     try:
#         control.gram(S_sys, 'cf')
#         control.gram(S_sys, 'of')
#     except ValueError:
#         S.display_system()
#         raise ValueError
#
#     if np.max(np.abs(np.linalg.eigvals(S.dynamics['enhanced']))) >= 1:
#         count += 1
#         # print('A:', np.sort(np.linalg.eigvals(S.dynamics['A'])))
#         S.display_active_architecture()
#         S_sys = control.StateSpace(S.dynamics['A'], S.architecture['B']['matrix'], S.architecture['C']['matrix'], np.zeros((len(S.architecture['C']['active']), len(S.architecture['B']['active']))), dt=True)
#         print('n_unstable A:', np.sum(np.abs(np.linalg.eigvals(S.dynamics['A'])) >= 1))
#         print('Ctrb check:', S.dynamics['number_of_nodes'] == np.linalg.matrix_rank(control.ctrb(S.dynamics['A'], S.architecture['B']['matrix'])))
#         print('Ctrb gram:', control.gram(S_sys, 'c'))
#         print('Obsv check:', S.dynamics['number_of_nodes'] == np.linalg.matrix_rank(control.obsv(S.dynamics['A'], S.architecture['C']['matrix'])))
#         print('Obsv gram:', control.gram(S_sys, 'o'))
#         print('Enhanced A:', np.sort(np.abs(np.linalg.eigvals(S.dynamics['enhanced']))))
#         # print(S.architecture['B']['matrix'])
#         # print(S.architecture['C']['matrix'])
#         # raise Exception('Failed closed loop')
#     #     print('Unstable:', np.sort(np.linalg.eigvals(S.dynamics['enhanced'])))
#     #     print('Control:', np.sort(np.linalg.eigvals(S.dynamics['A']-(S.architecture['B']['matrix']@S.architecture['B']['gain']))))
#     #     # print('Estimation A-LCA:', np.sort(np.linalg.eigvals(S.dynamics['A'] - (S.architecture['C']['gain'] @ S.architecture['C']['matrix']@S.dynamics['A']))))
#     #     print('Estimation (I-LC)A:', np.sort(np.linalg.eigvals((np.identity(S.dynamics['number_of_nodes']) - (S.architecture['C']['gain'] @ S.architecture['C']['matrix']))@S.dynamics['A'])))
#     #     raise Exception('Check exception')
# print('Count:', count)

# n = 10
# rho = 5
# n_count = []
# for i in range(0, 1000):
#     print('i:', i)
#     S = gac.System(graph_model={'number_of_nodes': n, 'rho': rho}, architecture={'rand': 1})
#     S.gramian_wrapper()
#     # print(S.dynamics['A'])
#     n_count.append(S.dynamics['n_unstable'])
#     # print('n_unstable:', S.dynamics['n_unstable'])
#     S.feedback_computations()
# print(np.histogram(n_count, bins=np.arange(start=0, stop=n+1, step=1, dtype=int)))
# print(n_edges)

print('\n Data reading')
try:
    shelve_data = shelve.open('DataDumps/comparison_fixed_vs_selftuning_n50_rho1.1')
except (FileNotFoundError, IOError):
    raise Exception('test file not found')

S_fixed = shelve_data['Fixed']
S_tuning = shelve_data['SelfTuning']
shelve_data.close()

if not isinstance(S_tuning, gac.System) or not isinstance(S_fixed, gac.System):
    raise Exception('Data type mismatch')

print('\n Plotting')
S_tuning.plot_architecture_history()
S_fixed.plot_trajectory_history()
S_tuning.plot_trajectory_history()
gac.cost_plots([S_fixed.trajectory['cost']['true'], S_tuning.trajectory['cost']['true']], S_tuning.model_name)

print('Done')
