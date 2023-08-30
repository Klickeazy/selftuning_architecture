import functionfile_speedygreedy as ff

if __name__ == "__main__":
    print('Code run start')

    # exp_list = [22, 41, 19, 20, 34, 31, 45, 46]

    exp_list = [19, 41]

    for exp_no in exp_list:

        ff.simulate_experiment(exp_no)

        ff.plot_experiment(exp_no)

    print('Code run done')
