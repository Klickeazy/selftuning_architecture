import greedy_architecture_combined as gac
import experiment as exp
import numpy as np
from copy import deepcopy as dc

if __name__ == "__main__":

    print('Generating Model')

    exp_no = 0

    S = gac.create_gen_model(exp.build_experiment_parameters(exp_no))

    # Architecture selection parameters
    S.architecture_limit_modifier(min_mod=S.architecture['B']['max']-1, max_mod=-S.dynamics['number_of_nodes']+S.architecture['B']['max'])


    # Architecture selection costs
    S.architecture_cost_update(R2=0, R3=0)
    # S.architecture['B']['cost']['R2'] = 100
    # S.architecture['C']['cost']['R2'] = 0
    # S.architecture['B']['cost']['R3'] = 100
    # S.architecture['C']['cost']['R3'] = 0

    S.model_rename()

    print('Optimizing fixed architecture')
    print('Initial')
    S.display_active_architecture()
    S_init = gac.greedy_architecture_initialization(S)
    S.active_architecture_duplicate(S_init)
    print('Optimized')
    S.display_active_architecture()

    # print(S.dynamics['A'])

    # for k in ['B', 'C']:
    #     for l in ['min', 'max']:
    #         print('k: ', k, '|l: ', l, ' : ', S.architecture[k][l])

    print('Model: ', S.model_name)

    gac.data_shelving_gen_model(S)
