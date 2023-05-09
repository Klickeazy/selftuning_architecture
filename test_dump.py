import functionfile_speedygreedy as ff


if __name__ == "__main__":

    exp_section = 2
    exp_no = 7
    multiprocess_check = False

    if exp_section == 1:
        exp = ff.Experiment()
        S = ff.initialize_system_from_experiment_number(exp_no)
        S = ff.optimize_initial_architecture(S, print_check=True)
        ff.data_to_memory_gen_model(S)

    elif exp_section == 2:
        exp = ff.Experiment()
        S = ff.initialize_system_from_experiment_number(exp_no)
        S = ff.data_from_memory_gen_model(S.model_name)

        # S_fix = ff.simulate_fixed_architecture(S, print_check=True)
        # print(S_fix.model_name)
        # print(S_fix.trajectory.cost.true)

        S_tune = ff.simulate_self_tuning_architecture(S, number_of_changes_limit=1, multiprocess_check=False, print_check=True)
        print(S_tune.model_name)
        print(S_tune.trajectory.cost.true)
        print(S_tune.B.history_active_set)
        print(S_tune.C.history_active_set)

    else:
        raise Exception('Code not defined')

    print('Code run done')