import greedy_architecture_combined as gac

if __name__ == "__main__":

    n = 50
    Tp = 10
    n_arch = 3
    n_arch_B = n_arch
    n_arch_C = n_arch
    network_model = 'eval_squeeze'
    # network_model = 'eval_bound'
    # network_model = 'rand'
    # rho = 3
    rho = None
    p = 0.1
    # second_order = True
    second_order = False
    test_model = 'combined'  # 'process', 'sensor', 'combined', None
    # test_model = None
    # sim_model = 'arch_replace'
    sim_model = None

    # n = 20
    # Tp = 10
    # n_arch = 2
    # n_arch_B = n_arch
    # n_arch_C = n_arch
    # # network_model = 'eval_squeeze'
    # network_model = 'eval_bound'
    # # network_model = 'rand'
    # # rho = 3
    # rho = None
    # p = 0.05
    # # second_order = True
    # second_order = False
    # test_model = 'combined'  # 'process', 'sensor', 'combined', None
    # # test_model = None
    # # sim_model = 'arch_replace'
    # sim_model = None

    model = gac.model_namer(n, rho, Tp, n_arch, test_model, second_order, network_model, sim_model)

    # gac.combined_plot(S, S_fixed, S_tuning)
    gac.time_axis_plot(model)
