import greedy_architecture_combined as gac
import numpy as np
import scipy as scp
import control as ctrl


def export_to_matlab(S_class):
    sys = {'A': S_class.dynamics['A'], 'B': S_class.architecture['B']['matrix'], 'C': S_class.architecture['C']['matrix'], 'Q_B': S_class.architecture['B']['cost']['Q'], 'R_B': S_class.architecture['B']['cost']['R1_active'], 'Q_C': S_class.architecture['C']['cost']['Q'], 'R_C': S_class.architecture['C']['cost']['R1_active']}
    scp.io.savemat('DataDumps/sys_to_mat.mat', sys)
    print('Saved mat to datadump')


n_nodes = 50
rho = 2
n_test = 1000
n_rand_act = 10
n_bar = []

for i in range(0, n_test):
    print("i:", i)
    S = gac.System(graph_model={'number_of_nodes': n_nodes, 'rho': rho}, architecture={'rand': n_rand_act})
    n_bar.append(S.dynamics['n_unstable'])
    ctrb_mat = ctrl.ctrb(S.dynamics['A'], S.architecture['B']['matrix'])
    if np.linalg.matrix_rank(ctrb_mat) != n_nodes:
        print("B:\n", S.architecture['B']['matrix'])
        print("ctrb mat:\n", ctrb_mat)
        export_to_matlab(S)
        raise Exception('Ctrb rank fail')
    obsv_mat = ctrl.obsv(S.dynamics['A'], S.architecture['C']['matrix'])
    if np.linalg.matrix_rank(obsv_mat) != n_nodes:
        print("C:\n", S.architecture['C']['matrix'])
        print("obsv mat:\n", obsv_mat)
        export_to_matlab(S)
        raise Exception('Obsv rank fail')

    # ctrb_dare = ctrl.dare(S.dynamics['A'], S.architecture['B']['matrix'], S.architecture['B']['cost']['Q'], S.architecture['B']['cost']['R1_active'])
    # if np.max(np.abs(ctrb_dare[1])) >= 1:
    #     print(ctrb_dare[1])
    #     raise Exception('Closed loop ctrl dynamics eigvals fail')
    # if np.min(np.linalg.eigvals(ctrb_dare[0])) < 0:
    #     print(ctrb_dare[0])
    #     raise Exception('Closed loop ctrl cost eigvals fail')
    #
    # obsv_dare = ctrl.dare(S.dynamics['A'].T, S.architecture['C']['matrix'].T, S.architecture['C']['cost']['Q'], S.architecture['C']['cost']['R1_active'])
    # if np.max(np.abs(obsv_dare[1])) >= 1:
    #     print(obsv_dare[1])
    #     raise Exception('Closed loop obsv dynamics eigvals fail')
    # if np.min(np.linalg.eigvals(obsv_dare[0])) < 0:
    #     print(obsv_dare[0])
    #     raise Exception('Closed loop obsv cost eigvals fail')

    ctrb_dare = scp.linalg.solve_discrete_are(S.dynamics['A'], S.architecture['B']['matrix'], S.architecture['B']['cost']['Q'], S.architecture['B']['cost']['R1_active'])
    if np.min(np.linalg.eigvals(ctrb_dare)) < 0:
        print(ctrb_dare)
        export_to_matlab(S)
        raise Exception('Closed loop ctrl cost eigvals fail')

    obsv_dare = scp.linalg.solve_discrete_are(S.dynamics['A'].T, S.architecture['C']['matrix'].T, S.architecture['C']['cost']['Q'], S.architecture['C']['cost']['R1_active'])
    if np.min(np.linalg.eigvals(obsv_dare)) < 0:
        print(obsv_dare)
        export_to_matlab(S)
        raise Exception('Closed loop obsv cost eigvals fail')

print('\nUnstable mode count:')
n_hist = np.histogram(n_bar, bins=np.arange(start=0, stop=n_nodes+1, step=1, dtype=int))
for i in range(0, len(n_hist[0])):
    if n_hist[0][i] != 0:
        print(n_hist[1][i], ' : ', n_hist[0][i])
