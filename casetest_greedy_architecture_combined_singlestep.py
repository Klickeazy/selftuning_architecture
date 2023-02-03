import greedy_architecture_combined as gac

if __name__ == "__main__":
    S = gac.System(graph_model={'type': 'ER'}, architecture={'rand': True})

    print(S.architecture['B']['cost'])
    print(S.architecture['C']['cost'])

    # S.cost_wrapper('estimate')
    #
    # # for i in S.trajectory:
    # #     print(i, S.trajectory[i])
    #
    # greedy = gac.greedy_architecture_selection(S)
    # # greedy = gac.greedy_architecture_rejection(S)
    # # greedy = gac.greedy_simultaneous(S, status_check=True)
    #
    # for i in greedy['value_history']:
    #     print(i)
    # #     # for j in i:
    # #         # print(j)
    # # print(greedy['time'])
    # greedy['work_set'].display_system()
    # # print(greedy['value_history'])
    # # print(greedy['work_set'].architecture['B']['history'])
    # # print(greedy['work_set'].architecture['C']['history'])