import numpy as np
# import greedyalgorithm_knapsackproblem as gkp
import greedyalgorithm_knapsackproblem2 as gkp
from copy import deepcopy as dc


if __name__ == '__main__':

    # i1 = gkp.Item()
    # i2 = gkp.Item()
    # i3 = dc(i1)
    #
    # i1.display_item()
    # i2.display_item()
    # i3.display_item()
    #
    # print(i1 == i2)
    # print(i1.compare_item(i3))
    #
    # i1.display_item()
    # i3.display_item()

    n = 10
    initial_set = gkp.ItemList(name='Total Set at t=0', n_random_items=n, low=-10, high=10)
    initial_set.display_list()

    limit = np.array([3, 5])
    # print('\n\n')
    # greedy_initial = gkp.greedy_selection(initial_set, gkp.ItemList(), limit, no_select=True)
    # greedy_initial['work_set'].display_list()
    # print(greedy_initial['time'])
    #
    # print('\n\n')
    # greedy_initial = gkp.greedy_selection(initial_set, gkp.ItemList(), limit)
    # greedy_initial['work_set'].display_list()
    # print(greedy_initial['time'])
    #
    # print('\n\n')
    # greedy_initial = gkp.greedy_rejection(initial_set, initial_set, limit, no_reject=True)
    # greedy_initial['work_set'].display_list()
    # print(greedy_initial['time'])
    #
    # print('\n\n')
    # greedy_initial = gkp.greedy_rejection(initial_set, initial_set, limit)
    # greedy_initial['work_set'].display_list()
    # print(greedy_initial['time'])

    print('\n\n')
    work_set = gkp.greedy_selection(initial_set, gkp.ItemList(), limit)['work_set']
    work_set.list_id = "Work set at t=0"
    work_set.display_list()

    T_sim = 10

    values = [work_set.list_value]

    for t in range(1, T_sim):
        print('\n\n')
        t_set = gkp.ItemList(name="Total set at t="+str(t), n_random_items=3)
        t_set.add_item_list(work_set)
        t_set.display_list()
        work_set = gkp.greedy_simultaneous(t_set, work_set, limit)['work_set']
        work_set.list_id = "Work set at t="+str(t)
        work_set.display_list()
        values.append(work_set.list_value)

    print('Cost change: ', values)
    print('Code Complete')
