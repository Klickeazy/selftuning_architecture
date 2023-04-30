import greedy_architecture_combined as gac
import shelve
import numpy as np
from copy import deepcopy as dc
import control
import socket
import pandas as pd

if socket.gethostname() == 'melap257805':
    datadump_folder_path = 'C:/Users/kxg161630/Box/KarthikGanapathy_Research/SpeedyGreedyAlgorithm/DataDump/'
else:
    datadump_folder_path = 'D:/Box/KarthikGanapathy_Research/SpeedyGreedyAlgorithm/DataDump/'


# parameter_keys = ['number_of_nodes', 'network_model', 'second_order']
# # 'second_order_network', 'initial_architecture_size', 'second_order_architecture', 'disturbance_model', 'simulation_model', 'architecture_constraint', 'rho', 'network_parameter', 'disturbance_step']
#
# data = {
#     "number_of_nodes": [20, 30, 50],
#     "network_model": ['rand', 'ER', 'BA'],
#     "second_order": [False, True, False]
# }
#
# df = pd.DataFrame(data)
#
# # Save the DataFrame as a CSV file
# df.to_csv(datadump_folder_path+"experiment_parameters.csv")

# parameter_keys = ['number_of_nodes', 'network_model', 'second_order', 'second_order_network', 'initial_architecture_size', 'second_order_architecture', 'disturbance_model', 'disturbance_step', 'disturbance_number', 'disturbance_magnitude', 'simulation_model', 'architecture_constraint', 'rho', 'network_parameter']
#
# parameter_values = [20, 'rand', False, 0, 2, None, None, None, None, None, None, 3, 3, None]

# p = pd.read_csv(datadump_folder_path + "parameter_table.csv")
#
# data = p.iloc[0]
# print([k for k in data])

values = [20, 'rand', False, 0, 2, None, 1, 1, None, None, None, None, None, 3, 3, None, 10]
keys = ['number_of_nodes', 'network_model', 'second_order', 'second_order_network', 'initial_architecture_size', 'second_order_architecture', 'W_scaling', 'V_scaling', 'disturbance_model', 'disturbance_step', 'disturbance_number', 'disturbance_magnitude', 'simulation_model', 'architecture_constraint', 'rho', 'network_parameter', 'prediction_time_horizon']

print('Keys: ', len(keys))
print('Values: ', len(values))

for i in range(0, min(len(values), len(keys))):
    print(keys[i], ':', values[i])

print("Code run done")
