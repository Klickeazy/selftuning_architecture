import numpy as np
import greedyalgorithm_knapsackproblem as gkp
from copy import deepcopy as dc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.widgets import Button, Slider


if __name__ == '__main__':

    n = 10
    initial_set = gkp.ItemList(name='Total Set at t=0', n_random_items=n, low=-10, high=10)
    initial_set.display_list()

    limit = np.array([3, 5])

    print('\n\n')
    work_set = gkp.greedy_selection(initial_set, gkp.ItemList(), limit)['work_set']
    work_set.list_id = "Work set at t=0"
    work_set.display_list()

    T_sim = 10
    number_of_iterations = 2
    number_of_changes_per_iteration = 2

    values = [work_set.list_value]
    work_history = [work_set]
    val_history = []
    t_range = range(1, T_sim)
    for t in t_range:
        print('\n\n')
        t_set = gkp.ItemList(name="Total set at t="+str(t), n_random_items=3)
        t_set.add_item_list(work_set)
        t_set.display_list()
        simultaneous_greedy_iteration = gkp.greedy_simultaneous(t_set, work_set, limit, iterations=number_of_iterations, changes_per_iteration=number_of_changes_per_iteration)
        work_set = simultaneous_greedy_iteration['work_set']
        work_set.list_id = "Work set at t="+str(t)
        work_set.display_list()
        val_history.append(simultaneous_greedy_iteration['value_history'])
        values.append(work_set.list_value)
        work_history.append(work_set)

    print('Cost change: ', values)

    # def plt_at_t():

    print('\n\n Sorted lists')
    for i in work_history:
        i.items = sorted(i.items, key=lambda h: (h.item_value, h.item_id))
        i.display_list()

    # fig = plt.figure(tight_layout=True)
    # gs = gs.GridSpec(2, 1)
    #
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax1.plot(t_range, values)
    # ax2 = fig.add_subplot(gs[1, 0])

    # print(val_history)
    # print(len(val_history))
    # print('\n\n')
    # for ni in range(0, number_of_iterations):
    #     for nc in range(0, number_of_changes_per_iteration):
    #         for t in t_range:
    #             print(val_history[ni][nc][t])
        # ax2.scatter(t*np.ones(len(val_history[t])), val_history[t])

    plt.show()



    print('Code Complete')
