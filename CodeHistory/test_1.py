"""
Test independent gains for actuators and sensors for architecture change
"""

import numpy as np
import functionfile_speedygreedy as ff
from copy import deepcopy as dc

if __name__ == "__main__":

    S = ff.initialize_system_from_experiment_number(2)
    print('B:', S.B.active_set)
    print('C:', S.C.active_set)
    S.prediction_gains()
    K1_0 = dc(S.B.recursion_matrix[0])
    K1_10 = dc(S.B.recursion_matrix[10])
    L1_0 = dc(S.C.recursion_matrix[0])
    L1_10 = dc(S.C.recursion_matrix[10])
    # print('K:', '\n', S.B.recursion_matrix[0], '\n', S.B.recursion_matrix[10])
    # print('L:', '\n', S.C.recursion_matrix[0], '\n', S.C.recursion_matrix[10])

    S.initialize_random_architecture_active_set(3, arch='C')
    S.architecture_update_to_matrix_from_active_set()
    print('B:', S.B.active_set)
    print('C:', S.C.active_set)
    S.prediction_gains()
    K2_0 = dc(S.B.recursion_matrix[0])
    K2_10 = dc(S.B.recursion_matrix[10])
    L2_0 = dc(S.C.recursion_matrix[0])
    L2_10 = dc(S.C.recursion_matrix[10])
    # print('K:', '\n', S.B.recursion_matrix[0], '\n', S.B.recursion_matrix[10])
    # print('L:', '\n', S.C.recursion_matrix[0], '\n', S.C.recursion_matrix[10])

    print('K_1:', np.allclose(K1_0, K2_0))
    print('L_1:', np.allclose(L1_0, L2_0))

    print('K_10:', np.allclose(K1_10, K2_10))
    print('L_10:', np.allclose(L1_10, L2_10))


    print('Code run done')
