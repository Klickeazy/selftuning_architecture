import matplotlib.pyplot as plt
import numpy as np

import greedy_architecture_combined as gac

if __name__ == "__main__":
    S = gac.System(architecture={'rand': True})
    S.display_active_architecture()

    # print(S.architecture['B']['cost'])
    # print(S.architecture['C']['cost'])

    # S.cost_wrapper('estimate')
    #
    # # for i in S.trajectory:
    # #     print(i, S.trajectory[i])
    #
    greedy = gac.greedy_architecture_selection(S, no_select=True)
    # greedy = gac.greedy_architecture_rejection(S, no_reject=True)

    # greedy = gac.greedy_simultaneous(S, iterations=None, changes_per_iteration=1)
    # gac.simultaneous_cost_plot(greedy['value_history'])

    # for i in greedy['value_history']:
    #     print(i)
    greedy['work_set'].display_active_architecture()
