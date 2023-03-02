import greedy_architecture_combined as gac
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np

n = 10
rho = 1.05
Tp = 10
n_arch = 2

S, S_fixed, S_tuning = gac.data_reading_sim_model(n, rho, Tp, n_arch)

plt_map = {'fixed': {'c': 'C0', 'line_style': 'solid', 'alpha': 0.5, 'zorder': 1},
               'tuning': {'c': 'C1', 'line_style': 'dashed', 'alpha': 0.5, 'zorder': 2},
               'marker': {'B': "o", 'C': "o"}}

fig = plt.figure(figsize=(10, 7), constrained_layout=True)
grid = fig.add_gridspec(5, 2)

ax_cost = fig.add_subplot(grid[0, 0])
ax_trajectory = fig.add_subplot(grid[1, 0], sharex=ax_cost)
ax_error = fig.add_subplot(grid[2, 0], sharex=ax_cost)
ax_architecture_B = fig.add_subplot(grid[3, 0], sharex=ax_cost)
ax_architecture_C = fig.add_subplot(grid[4, 0], sharex=ax_cost)

ax_network = fig.add_subplot(grid[1:4, 1])
ax_network.tick_params(axis='both', labelbottom=False, labelleft=False, bottom=False, top=False)

ax_timeline = fig.add_subplot(grid[:, 0], sharex=ax_cost, frameon=False)
ax_timeline.tick_params(axis='both', labelbottom=False, labelleft=False, bottom=False, top=False)

gac.cost_plots({'fixed': S_fixed.trajectory['cost']['true'], 'tuning': S_tuning.trajectory['cost']['true']}, S.model_name, ax_cost, plt_map=plt_map)
S_fixed.plot_trajectory(ax_in={'x': ax_trajectory, 'error': ax_error}, plt_map=plt_map, s_type='fixed')
S_tuning.plot_trajectory(ax_in={'x': ax_trajectory, 'error': ax_error}, plt_map=plt_map, s_type='tuning')
S_tuning.plot_architecture_history(ax_in={'B': ax_architecture_B, 'C': ax_architecture_C}, plt_map=plt_map)

S_tuning.plot_network(ax_in=ax_network)

s_w = 0.3
s_h = 0.025
b_w = 0.1
b_h = 0.05
ax_timeslide = fig.add_axes([((0.95+0.55-s_w)/2), 0.1, s_w, s_h])
timeslider = Slider(ax=ax_timeslide, label='Time', valmin=0, valmax=S.simulation_parameters['T_sim']+1, valinit=S.simulation_parameters['T_sim']+1, valstep=1)
ax_button = fig.add_axes([((0.95+0.55-b_w)/2), 0.05, b_w, b_h])
button = Button(ax=ax_button, label='Reset', hovercolor='0.975')


def update(t):
    ax_network.clear()
    ax_timeline.clear()
    S_tuning.plot_network(time_step=int(t), ax_in=ax_network)
    ax_timeline.axvline(int(t), alpha=0.5, linewidth=1, linestyle='dotted', c='k', zorder=-1)


def reset(event):
    timeslider.reset()


timeslider.on_changed(update)
button.on_clicked(reset)

plt.show()
