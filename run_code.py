import functionfile_speedygreedy as ff

if __name__ == "__main__":
    print('Code run start')

    # exp_master_list = [34, 31, 47, 49, 50, 51, 59]


    exp_list = [4]

    # run_flag = True
    run_flag = False

    plot_flag = True
    # plot_flag = False

    for exp_no in exp_list:

        ff.run_experiment(exp_no, run_check=run_flag, plot_check=plot_flag)

    print('Code run done')
