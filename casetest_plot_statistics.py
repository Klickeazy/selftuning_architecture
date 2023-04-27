import greedy_architecture_combined as gac

if __name__ == "__main__":

    n_samples = 100

    n = 30
    Tp = 10
    n_arch = 2
    # n_arch_B = n_arch
    # n_arch_C = n_arch

    # rho = 5
    rho = None

    test_model = 'combined'
    # test_model = None

    # second_order = True
    second_order = False

    network_model = 'eval_squeeze'

    # sim_model = 'arch_replace'
    sim_model = None

    model_name = gac.model_namer(n, rho, Tp, n_arch, test_model, second_order, network_model, sim_model)
    print('Model :', model_name)
    gac.statistics_plot(model_name, n_samples)
