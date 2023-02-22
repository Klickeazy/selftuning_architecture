import shelve
import greedy_architecture_combined as gac

# Plot from data dump code
print('\n Data reading')
try:
    shelve_data = shelve.open('DataDumps/comparison_fixed_vs_selftuning_n50_rho1.05')
except (FileNotFoundError, IOError):
    raise Exception('test file not found')

for k in ['System', 'Fixed', 'SelfTuning']:
    if k not in shelve_data:
        raise Exception('Check data file')

S = shelve_data['System']
S_fixed = shelve_data['Fixed']
S_tuning = shelve_data['SelfTuning']
shelve_data.close()

if not isinstance(S, gac.System) or not isinstance(S_tuning, gac.System) or not isinstance(S_fixed, gac.System):
    raise Exception('Data type mismatch')

print('\n Plotting')
S_tuning.plot_architecture_history()
gac.trajectory_plots({'fixed': S_fixed.trajectory['x'], 'tuning': S_tuning.trajectory['x']}, {'fixed': S_fixed.trajectory['error'], 'tuning': S_tuning.trajectory['error']}, S.model_name)
gac.cost_plots({'fixed': S_fixed.trajectory['cost']['true'], 'tuning': S_tuning.trajectory['cost']['true']}, S.model_name)
