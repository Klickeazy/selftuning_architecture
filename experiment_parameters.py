import numpy as np
import time
import functionfile_speedygreedy as ff


if __name__ == "__main__":
    print('Code run start')

    exp = ff.Experiment()

    # exp.initialize_table()

    S = ff.initialize_system_from_experiment_number(6)
    print(S.model_name)
    print('OL modes:', S.A.open_loop_eig_vals)

    # print(S.B.min)
    # print(S.B.max)
    # print(S.C.min)
    # print(S.C.max)
    #
    # S.architecture_limit_mod()
    #
    # print(S.B.min)
    # print(S.B.max)
    # print(S.C.min)
    # print(S.C.max)

    # S = ff.greedy_selection(S, print_check=True, multiprocess_check=True)
    # S = ff.greedy_rejection(S, print_check=True, multiprocess_check=True)

    S.architecture_display_active_set()
    S = ff.greedy_simultaneous(S, print_check=True, swap_only=False)
    print(S.trajectory.computation_time)
    S.architecture_display_active_set()

    S.cost_display_stage_components()

    print('Code run done')
