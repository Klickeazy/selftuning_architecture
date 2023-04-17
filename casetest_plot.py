import greedy_architecture_combined as gac

if __name__ == "__main__":

    n = 50
    rho = None
    Tp = 10
    n_arch = 5

    # test_model = 'combined'
    test_model = None

    # second_order = True
    second_order = False

    model = gac.model_namer(n, rho, Tp, n_arch, test_model, second_order)
    print(model)

    # gac.combined_plot(S, S_fixed, S_tuning)
    gac.time_axis_plot(model)
