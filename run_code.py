import functionfile_speedygreedy as ff

if __name__ == "__main__":
    print('Code run start')

    # exp_master_list = [22, 41, 19, 20, 34, 31, 47, 49, 50, 51, 59]
    # exp_single_run = [52, 53, 54, 55, 56]

    # rerun = [20, 34, 31, 50]

    exp_list = [34]

    run_flag = True
    # run_flag = False

    plot_flag = True
    # plot_flag = False

    for exp_no in exp_list:

        ff.run_experiment(exp_no, run_check=run_flag, plot_check=plot_flag)

    print('Code run done')
