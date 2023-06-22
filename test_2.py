"""
Test cost and gain computations for time step update
"""


import numpy as np
import functionfile_speedygreedy as ff
from copy import deepcopy as dc

if __name__ == "__main__":
    S = ff.initialize_system_from_experiment_number(2)

    S.cost_prediction_wrapper()

    print('Predicted cost:', S.trajectory.cost.predicted)
    print('Switching cost:', S.trajectory.cost.switching)
    print('Running cost:', S.trajectory.cost.running)

    S.cost_true_wrapper()

    print('Predicted cost:', S.trajectory.cost.true)
    print('Switching cost:', S.trajectory.cost.switching)
    print('Running cost:', S.trajectory.cost.running)
