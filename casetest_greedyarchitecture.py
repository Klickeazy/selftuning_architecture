import numpy as np
from copy import deepcopy as dc
import greedy_architecture as ga

if __name__ == '__main__':

    # Sys = ga.System(graph_model={'type': 'path', 'self_loop': True})
    # Sys = ga.System(graph_model={'type': 'ER'})
    # print(Sys.architecture['B']['active'])
    # print(Sys.architecture['B']['matrix'])
    # print(Sys.architecture['B']['set'])
    # Sys.display_system()

    Sys = ga.System(graph_model={'type': 'ER'})
    architecture_record = [{'B': {'active': Sys.architecture['B']['active']},
                            'C': {'active': Sys.architecture['C']['active']}}]
    for t in range(0, 10):
        Sys.random_architecture()
        architecture_record.append({'B': {'active': dc(Sys.architecture['B']['active'])}, 'C': {'active': dc(Sys.architecture['C']['active'])}})

    # print(architecture_record[0])

    ga.animate_architecture(Sys, architecture_record)


    # architecture_choices = []
    # n = 10
    # for i in range(0, n):
    #     architecture_choices.append(np.zeros((n, 1)))
    #     architecture_choices[-1][i, 0] = 1
    # print(architecture_choices)


    print('Main code complete')