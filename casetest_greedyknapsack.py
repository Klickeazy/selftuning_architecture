import numpy as np
import greedyalgorithm_knapsackproblem as gkp
from copy import deepcopy as dc


if __name__ == '__main__':

    n = 10
    initial_set = gkp.initialize_rand_items(n, 'Initial Set')
    initial_set.display_list()

    # limit = 3
    # initial_greedy_selection = gkp.greedy_algorithm(initial_set, limit, 'selection')
    # initial_greedy_selection.listid = "Initial Greedy Selection"
    # initial_greedy_selection.display_list()
    #
    # initial_greedy_rejection = gkp.greedy_algorithm(initial_set, limit, 'rejection')
    # initial_greedy_rejection.listid = "Initial Greedy Rejection"
    # initial_greedy_rejection.display_list()

    simultaneous = gkp.greedy_simultaneous_optimal(initial_set, change=3)
    simultaneous.display_list()
