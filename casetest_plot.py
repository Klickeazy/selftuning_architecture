import greedy_architecture_combined as gac

if __name__ == "__main__":

    n = 20
    # rho = 3
    rho = None
    Tp = 10
    n_arch = 3

    # test_model = 'combined'
    test_model = None

    second_order = True
    # second_order = False

    sim_model = 'unlimited_arch_change'
    # sim_model = None

    network_model = 'rand_eval'
    # network_model = 'rand'

    model = gac.model_namer(n, rho, Tp, n_arch, test_model, second_order, network_model, sim_model)
    print(model)

    # gac.combined_plot(S, S_fixed, S_tuning)
    gac.time_axis_plot(model)
