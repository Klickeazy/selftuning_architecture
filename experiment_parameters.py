import numpy as np
import time
import functionfile_speedygreedy as ff


if __name__ == "__main__":
    print('Code run start')

    exp = ff.Experiment()

    # exp.initialize_table()

    S = ff.initialize_system_from_experiment_number(4)

    S = ff.greedy_selection(S, print_check=True)
    print(S.trajectory.cost.predicted)
    print(S.trajectory.cost.running)
    print(S.trajectory.cost.switching)
    print(S.trajectory.cost.control)

    # S = ff.greedy_rejection(S, print_check=True)
    # print(S.trajectory.computation_time)

    # ff.greedy_simultaneous(S, number_of_changes_limit=1, number_of_changes_per_iteration=1, print_check=True)
    # ff.greedy_simultaneous(S, number_of_changes_limit=1, number_of_changes_per_iteration=1, print_check=True)

    print('Code run done')
