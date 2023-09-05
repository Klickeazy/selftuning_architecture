import functionfile_speedygreedy as ff

if __name__ == "__main__":
    print('Code run start')

    # exp_master_list = [22, 41, 19, 20, 34, 31, 47, 49, 50, 51]

    exp_list = [22, 41, 19, 20, 34, 31, 47, 49, 50, 51]

    for exp_no in exp_list:

        # ff.simulate_experiment(exp_no)

        ff.plot_experiment(exp_no)

    print('Code run done')
