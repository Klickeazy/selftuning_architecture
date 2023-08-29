import functionfile_speedygreedy as ff

S = ff.initialize_system_from_experiment_number(exp_no=22)

S = ff.greedy_simultaneous(S, print_check_outer=True)