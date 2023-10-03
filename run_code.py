import functionfile_speedygreedy as ff

if __name__ == "__main__":
    print('Code run start')

    # exp_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    exp_list = [11]

    # run_flag = True
    run_flag = False

    plot_flag = True
    # plot_flag = False

    for exp_no in exp_list:

        ff.run_experiment(exp_no, run_check=run_flag, plot_check=plot_flag)

    print('Code run done')
