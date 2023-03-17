import greedy_architecture_combined as gac

if __name__ == "__main__":
    n = 10
    rho = 1.05
    Tp = 10
    n_arch = 2

    S, S_fixed, S_tuning = gac.data_reading_sim_model(n, rho, Tp, n_arch)
    gac.time_axis_plot(S, S_fixed, S_tuning)