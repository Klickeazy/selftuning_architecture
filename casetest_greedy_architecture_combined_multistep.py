import greedy_architecture_combined as gac
from copy import deepcopy as dc

if __name__ == "__main__":
    S = gac.System(graph_model={'type': 'ER', 'rho': 0.8}, architecture={'rand': True})
    S.optimal_control_feedback_wrapper()
    S.optimal_estimation_feedback_wrapper()
    S.cost_wrapper()

    T_sim = 100

    for t in range(0, T_sim):
        S.system_one_step_update_enhanced()
        S = dc(gac.greedy_simultaneous(S)['work_set'])
        S.cost_wrapper('estimate')
        S.cost_wrapper('true')

    # print(S.trajectory['cost']['estimate_total'])
    # print(S.trajectory['cost']['true_total'])
    # print(len(S.trajectory['cost']['estimate_total']))
    # print(len(S.trajectory['cost']['true_total']))

    S.cost_plot()
