import greedy_architecture_combined as gac

if __name__ == "__main__":

    n_samples = 100

    n = 20
    Tp = 10
    n_arch = 3
    # n_arch_B = n_arch
    # n_arch_C = n_arch

    # rho = 5
    rho = None

    test_model = 'combined'
    # test_model = None

    # second_order = True
    second_order = False

    model_name = gac.model_namer(n, rho, Tp, n_arch, test_model, second_order)
    print(model_name)
    gac.statistics_plot(model_name, n_samples)
