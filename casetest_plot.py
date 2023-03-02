import greedy_architecture_combined as gac

if __name__ == "__main__":

    n = 50
    rho = 2
    Tp = 30
    n_arch = 5

    S, S_fixed, S_tuning = gac.data_reading_sim_model(n, rho, Tp, n_arch)

    gac.combined_plot(S, S_fixed, S_tuning)
