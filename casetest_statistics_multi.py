import os
from time import sleep

# import scipy.stats
from tqdm import tqdm
import greedy_architecture_combined as gac
from multiprocessing import Pool
import numpy as np


def sys_gen():
    n = 50
    rho = 5
    Tp = 10
    n_arch = 5
    n_arch_B = n_arch
    n_arch_C = n_arch
    # test_model = 'combined'
    test_model = None
    disturbance_step = 10
    disturbance_number = int(np.floor(n / 2))
    disturbance_magnitude = 20
    disturbance = {'step': disturbance_step, 'number': disturbance_number, 'magnitude': disturbance_magnitude}

    S = gac.System(graph_model={'number_of_nodes': n, 'rho': rho}, architecture={'rand': n_arch}, additive={'type': test_model, 'disturbance': disturbance}, simulation_parameters={'T_sim': 100, 'T_predict': Tp})
    # Architecture selection parameters
    S.architecture_limit_modifier(min_mod=n_arch - 1, max_mod=-n + n_arch)
    # Architecture selection costs
    S.architecture_cost_update({'R2': 0, 'R3': 0})
    S.model_rename()
    return S


def sys_test(i):
    sleep(1)
    # print('Sample s: ', str(i))
    fail_check = True
    while fail_check:
        try:
            S = sys_gen()
            S_fixed = gac.simulate_fixed_architecture(S, print_check=False)
            S_tuning = gac.simulate_selftuning_architecture(S, print_check=False)
            gac.data_shelving_statistics(S, S_fixed, S_tuning, i)
            fail_check = False
        except Exception as e:
            print(e)
            print('Fail at s: ', str(i))


if __name__ == "__main__":

    n_samples = 100
    print('CPUs available: ', os.cpu_count())
    active_pool = os.cpu_count() - 4
    print('CPUs for process: ', active_pool)
    m_pool = Pool(active_pool)
    S_init = sys_gen()
    gac.statistics_shelving_initialize(S_init.model_name)
    for _ in tqdm(m_pool.imap_unordered(sys_test, range(1, n_samples+1)), total=len(range(1, n_samples+1)), ncols=100):
        pass
    print('Code Run Done: ', S_init.model_name)
