import numpy as np
import functionfile_speedygreedy as ff


if __name__ == "__main__":

    exp = ff.Experiment()
    # exp.initialize_table()

    # S = ff.initialize_system_from_experiment_number(8)

    ff.test_all_experiments()

    # exp.read_parameters_from_table(2)

    # S = ff.System()
    # S.initialize_system_from_experiment_parameters(exp.parameter_values, exp.parameter_keys[1:])

    # exp.parameter_values = [2, 20, 'rand', False, 0, 2, 0, 1, 1, None, 0, 0, 0, None, 3, 3, 0, 10, 1]
    # exp.write_parameters_to_table()


    print('Code done')
