import greedy_architecture_combined as gac

if __name__ == "__main__":

    n = 20
    network_model = 'rand_eval'
    # network_model = 'rand'
    rho = None
    # rho = 3
    Tp = 10
    n_arch = 2

    # test_model = 'combined'
    test_model = None

    second_order = True
    # second_order = False

    # sim_model = 'arch_replace'
    sim_model = None

    model = gac.model_namer(n, rho, Tp, n_arch, test_model, second_order, network_model, sim_model)

    # gac.combined_plot(S, S_fixed, S_tuning)
    gac.time_axis_plot(model)
