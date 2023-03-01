import shelve
import greedy_architecture_combined as gac

datadump_folderpath = 'C:/Users/kxg161630/Box/KarthikGanapathy_Research/SpeedyGreedyAlgorithm/DataDump/'

# Plot from data dump code
print('\n Data reading')

n = 50
rho = 1.5
Tp = 10
n_arch = 5

shelve_file = datadump_folderpath+'model_n'+str(n)+'_rho'+str(rho)+'_Tp'+str(Tp)+'_arch'+str(n_arch)
# shelve_file = datadump_folderpath+'model_n'+str(n)+'_rho'+str(rho)+'_Tp'+str(Tp)
try:
    shelve_data = shelve.open(shelve_file)
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
gac.combined_plot(S, S_fixed, S_tuning)
