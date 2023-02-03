import greedy_architecture_combined as gac

if __name__ == "__main__":
    S = gac.System(architecture={'rand': True})
    # # for i in S.trajectory:
    # #     print(i, S.trajectory[i])
    #
    # greedy_selection = gac.greedy_architecture_rejection(S)
    greedy_simultaneous = gac.greedy_simultaneous(S)
    print(greedy_simultaneous['work_set'])
    print(greedy_simultaneous['value_history'])

