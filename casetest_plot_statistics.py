import greedy_architecture_combined as gac

if __name__ == "__main__":

    n_samples = 100

    n = 30
    rho = 5
    Tp = 10
    n_arch = 5
    n_arch_B = n_arch
    n_arch_C = n_arch
    # test_model = 'combined'
    test_model = None
    # second_order = False
    second_order = True
    model_name = gac.model_namer(n, rho, Tp, n_arch, test_model, second_order)
    print(model_name)
    gac.statistics_plot(model_name)
