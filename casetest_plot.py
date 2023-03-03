import greedy_architecture_combined as gac

if __name__ == "__main__":

    n = 30
    rho = 4
    Tp = 10
    n_arch = 6

    test_model = 'combined'
    # test_model = None

    model = gac.model_name(n, rho, Tp, n_arch, test_model)
    print(model)

    # gac.combined_plot(S, S_fixed, S_tuning)
    gac.slider_plot(model)
