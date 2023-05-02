import numpy as np
import functionfile_speedygreedy as ff


if __name__ == "__main__":

    exp = ff.Experiment()

    # exp.initialize_table()

    S = ff.initialize_system_from_experiment_number(2)

    print(S.A.open_loop_eig_vals)

    S.prediction_control_gain()
    S.prediction_estimation_gain()

    print(len(S.B.recursion_matrix))
    print(len(S.C.recursion_matrix))


    print('Code done')
