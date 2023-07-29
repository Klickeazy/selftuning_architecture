from CodeHistory import functionfile_speedygreedy_old as ff

import matplotlib.pyplot as plt


if __name__ == "__main__":
    print('Code run start')

    exp_section = 3
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
        S_fix = ff.simulate_fixed_architecture(S)
        S_tune = ff.simulate_self_tuning_architecture(S, number_of_changes_limit=None, multiprocess_check=multiprocess_check)
        ff.data_to_memory_sim_model(S, S_fix, S_tune)

        # print(S_tune.trajectory.cost.true)
        # print(S_tune.trajectory.cost.predicted)

    elif exp_section == 3:
        exp = ff.Experiment()
        S = ff.initialize_system_from_experiment_number(exp_no)
        S, S_fix, S_tune = ff.data_from_memory_sim_model(S.model_name)

        for i in range(0, S.number_of_states):
            plt.plot(range(0, S_fix.sim.t_simulate), S_fix.trajectory.x[:, i], c='tab:blue')
            plt.plot(range(0, S_tune.sim.t_simulate), S_tune.trajectory.x[:, i], c='tab:green')
        # plt.legend()
        plt.show()

        # fix_norm = [np.linalg.norm(S_fix.trajectory.x[t, :]) for t in range(0, S_fix.sim.t_simulate)]
        # tune_norm = [np.linalg.norm(S_tune.trajectory.x[t, :]) for t in range(0, S_tune.sim.t_simulate)]
        #
        # plt.plot(range(0, S_fix.sim.t_simulate), fix_norm, c='tab:blue')
        # plt.plot(range(0, S_tune.sim.t_simulate), tune_norm, c='tab:green')
        # plt.show()

        print(S_tune.B.history_active_set)
        print(S_tune.C.history_active_set)

        # print(S_fix.trajectory.x)
        # print(S_tune.trajectory.x)
        # print(S_fix.trajectory.estimation_matrix[0])
        # print(S_fix.trajectory.estimation_matrix[1])
        # print(S_fix.trajectory.estimation_matrix[2])

    else:
        raise Exception('Code not defined')

    print('Code run done')
