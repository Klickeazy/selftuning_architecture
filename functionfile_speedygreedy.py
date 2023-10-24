# Copyright 2023, Karthik Ganapathy

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import numpy as np
import networkx as netx

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator, FuncFormatter, LogFormatter, LogLocator, NullFormatter

from time import process_time
from copy import deepcopy as dc
import pandas as pd
import dbm
import shelve
import itertools
import concurrent.futures

import os
import socket

from tqdm.autonotebook import tqdm

matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['legend.title_fontsize'] = 10
matplotlib.rcParams['legend.framealpha'] = 0.5
matplotlib.rcParams['lines.markersize'] = 5
matplotlib.rcParams['image.cmap'] = 'Blues'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'


# matplotlib.rcParams['axes.autolimit_mode'] = 'round_numbers'


# Error handling
class ArchitectureError(Exception):
    """Raise when architecture is not B or C"""

    def __init__(self, value='Check architecture type'):
        self.value = value

    def __str__(self):
        return repr(self.value)


class SecondOrderError(Exception):
    """Raise when second order or type is not specified accurately"""

    def __init__(self, value='Check second order type'):
        self.value = value

    def __str__(self):
        return repr(self.value)


class ClassError(Exception):
    """Raise when variable is not of the correct class/variable type"""

    def __init__(self, value='Check class data type'):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Experiment:
    """
    Manage experiment parameters - save/load from csv file
    Wrapper to simulate and plot objects of System class
    """

    def __init__(self):
        self.parameters_save_filename = "experiment_parameters.csv"  # File name for experiment parameters

        # Dictionary of parameter names and default value with data-type
        self.default_parameter_datatype_map = {'experiment_no': int(1),
                                               'test_model': str(None),
                                               'test_parameter': int(0),
                                               'number_of_nodes': int(20),
                                               'network_model': str('rand'),
                                               'network_parameter': float(0),
                                               'rho': float(1),
                                               'second_order': False,
                                               'second_order_network': int(0),
                                               'initial_architecture_size': int(5),
                                               'architecture_constraint_min': int(5),
                                               'architecture_constraint_max': int(5),
                                               'second_order_architecture': int(0),
                                               'Q_cost_scaling': float(1),
                                               'R_cost_scaling': float(1),
                                               'B_run_cost': float(1),
                                               'C_run_cost': float(1),
                                               'B_switch_cost': float(1),
                                               'C_switch_cost': float(1),
                                               'W_scaling': float(1),
                                               'V_scaling': float(1),
                                               'disturbance_model': str(None),
                                               'disturbance_step': int(0),
                                               'disturbance_number': int(0),
                                               'disturbance_magnitude': int(0),
                                               'prediction_time_horizon': int(10),
                                               'X0_scaling': float(1),
                                               'multiprocessing': False}

        self.parameter_keys = list(self.default_parameter_datatype_map.keys())  # Strip parameter names from dict
        self.parameter_datatypes = {k: type(self.default_parameter_datatype_map[k]) for k in
                                    self.default_parameter_datatype_map}  # Strip parameter data types from dict

        # Parameters and simulation systems for current experiment
        self.exp_no = 1
        self.parameter_values = []

        # System dictionaries: key = 0 for single exp, model_id for statistical
        self.S = {0: System()}  # Dictionary of generated systems
        self.S_1 = {}   # Dictionary of first system to compare
        self.S_2 = {}   # Dictionary of second system to compare
        self.process_pool_workers = None    # number of workers for processpool: none=max

        self.datadump_folder_path = 'DataDump/'

        self.image_save_folder_path = 'Images/'

        # Saved experiment data parse
        self.parameter_table = pd.DataFrame()  # Parameter table from csv
        self.experiments_list = []  # List of experiments
        self.read_table_from_file()

        # Plot parameters
        self.plot_title_check = True

        # Simulation function mapper
        self.experiment_modifications_mapper = {
            'fixed_vs_self_tuning': self.simulate_experiment_fixed_vs_self_tuning,
            'self_tuning_number_of_changes': self.simulate_experiment_self_tuning_number_of_changes,
            'self_tuning_prediction_horizon': self.simulate_experiment_self_tuning_prediction_horizon,
            'self_tuning_architecture_cost': self.simulate_experiment_self_tuning_architecture_cost,
            'pointdistribution_openloop': self.simulate_experiment_fixed_vs_self_tuning_pointdistribution_openloop,
            'self_tuning_architecture_cost_no_lim': self.simulate_experiment_self_tuning_architecture_cost_no_lim,
            'self_tuning_architecture_constraints': self.simulate_experiment_self_tuning_architecture_constraints
        }

        self.experiment_mapper_statistics = {
            'statistics_fixed_vs_self_tuning': 'fixed_vs_self_tuning',
            'statistics_self_tuning_number_of_changes': 'self_tuning_number_of_changes',
            'statistics_self_tuning_prediction_horizon': 'self_tuning_prediction_horizon',
            'statistics_self_tuning_architecture_cost': 'self_tuning_architecture_cost',
            'statistics_self_tuning_architecture_cost_no_lim': 'self_tuning_architecture_cost_no_lim',
            'statistics_pointdistribution_openloop': 'pointdistribution_openloop',
            'statistics_self_tuning_architecture_constraints': 'self_tuning_architecture_constraints'
        }

    def initialize_table(self) -> None:
        # Initialize parameter csv file from nothing - FOR BUGFIXING ONLY
        print('Initializing table with default parameters')
        self.parameter_values = [[k] for k in self.default_parameter_datatype_map.values()]
        self.parameter_table = pd.DataFrame(dict(zip(self.parameter_keys, self.parameter_values)))
        self.parameter_table.set_index(self.parameter_keys[0], inplace=True)
        self.write_table_to_file()

    def check_dimensions(self, print_check=False) -> None:
        # Ensure dimensions match
        if len(self.parameter_values) == len(self.parameter_datatypes) == len(self.parameter_keys):
            if print_check:
                print('Dimensions agree: {} elements'.format(len(self.parameter_keys)))
        else:
            raise Exception("Dimension mismatch - values: {}, datatype: {}, keys: {}".format(len(self.parameter_values), len(self.parameter_datatypes), len(self.parameter_keys)))

    def check_experiment_number(self) -> None:
        # Ensure experiment number is in the tables list
        if self.exp_no not in self.experiments_list:
            raise Exception('Invalid experiment number')

    def read_table_from_file(self) -> None:
        # Read table from file
        if not os.path.exists(self.parameters_save_filename):
            raise Warning('File does not exist')
        else:
            self.parameter_table = pd.read_csv(self.parameters_save_filename, index_col=0, dtype=self.parameter_datatypes)
            self.parameter_table.replace({np.nan: None}, inplace=True)
            self.experiments_list = self.parameter_table.index

    def read_parameters_from_table(self) -> None:
        # Read parameters from table
        if not isinstance(self.parameter_table, pd.DataFrame):
            raise Exception('Not a pandas frame')

        self.parameter_values = [self.exp_no] + [k for k in self.parameter_table.loc[self.exp_no]]

    def parameter_value_map(self) -> None:
        # Mapping list for parameters to manage default csv values to python values
        self.parameter_values = [list(map(d, [v]))[0] if v is not None else None for d, v in
                                 zip(self.parameter_datatypes, self.parameter_values)]

    def write_parameters_to_table(self) -> None:
        # Write a parameter dictionary to an experiment table - FOR BUGFIXING ONLY
        if len(self.parameter_values) == 0:
            raise Exception('No experiment parameters provided')
        self.check_dimensions()
        self.parameter_value_map()
        append_check = True
        for i in self.experiments_list:
            if [k for k in self.parameter_table.loc[i]][:] == self.parameter_values[1:]:
                print('Duplicate experiment at : {}'.format(i))
                append_check = False
                break
        if append_check:
            self.parameter_table.loc[max(self.experiments_list) + 1] = self.parameter_values[1:]
            self.experiments_list = self.parameter_table.index
            self.write_table_to_file()

    def write_table_to_file(self) -> None:
        # Write an experiment table to csv - FOR BUGFIXING ONLY
        self.parameter_table.to_csv(self.parameters_save_filename)
        print('Printing done')

    # def return_keys_values(self):
    #     return self.parameter_values, self.parameter_keys

    def display_test_parameters(self) -> None:
        # Print parameter table
        print(self.parameter_table)

    def initialize_system_from_experiment_number(self, exp_no=None) -> None:
        # Define model S[0] for given experiment number
        if exp_no is not None:
            self.exp_no = exp_no
        self.check_experiment_number()
        self.read_parameters_from_table()
        self.S[0] = System()
        self.S[0].initialize_system_from_experiment_parameters(self.parameter_values, self.parameter_keys)

        if self.S[0].sim.test_model not in self.experiment_modifications_mapper and self.S[0].sim.test_model not in self.experiment_mapper_statistics:
            raise Exception('Experiment not defined')

    def retrieve_experiment(self, exp_no=None) -> None:
        # Retrieve simulated experiment from datadump
        if exp_no is None:
            exp_no = self.exp_no
        self.initialize_system_from_experiment_number(exp_no)
        self.system_model_from_memory_sim_model(self.S[0].model_name)
        if self.S_1[0].plot is None:
            self.S_1[0].plot = PlotParameters()

        if self.S_2[0].plot is None:
            self.S_2[0].plot = PlotParameters()

    def system_model_to_memory_gen_model(self) -> None:
        # Store model generated from experiment parameters
        shelve_filename = self.datadump_folder_path + 'gen_' + self.S[0].model_name
        with shelve.open(shelve_filename, writeback=True) as shelve_data:
            shelve_data['s'] = self.S[0]
        print('\nShelving gen model: {}'.format(shelve_filename))

    def system_model_from_memory_gen_model(self, model, print_check=False):
        # Retrieve model generated from experiment parameters
        shelve_filename = self.datadump_folder_path + 'gen_' + model
        if print_check:
            print('\nReading gen model: {}'.format(shelve_filename))
        with shelve.open(shelve_filename, flag='r') as shelve_data:
            self.S[0] = shelve_data['s']
        if not isinstance(self.S[0], System):
            raise Exception('System model error')

    def system_model_to_memory_sim_model(self) -> None:
        # Store simulated models
        shelve_filename = self.datadump_folder_path + 'sim_' + self.S[0].model_name
        print('\nShelving sim model: {}'.format(shelve_filename))
        with shelve.open(shelve_filename, writeback=True) as shelve_data:
            shelve_data['s'] = self.S[0]
            shelve_data['s1'] = self.S_1[0]
            shelve_data['s2'] = self.S_2[0]

    def system_model_from_memory_sim_model(self, model):
        # Retrieve simulated models
        shelve_filename = self.datadump_folder_path + 'sim_' + model
        print('\nReading sim model: {}'.format(shelve_filename))
        with shelve.open(shelve_filename, flag='r') as shelve_data:
            self.S[0] = shelve_data['s']
            self.S_1[0] = shelve_data['s1']
            self.S_2[0] = shelve_data['s2']
        if not isinstance(self.S[0], System) or not isinstance(self.S_1[0], System) or not isinstance(self.S_2[0], System):
            raise Exception('Data type mismatch')
        self.S[0].plot = PlotParameters()
        self.S_1[0].plot = PlotParameters(1)
        self.S_2[0].plot = PlotParameters(2)

    def system_model_to_memory_statistics(self, model_id: int, print_check: bool = False) -> None:
        # Store simulated statistics model to memory
        shelve_filename = self.datadump_folder_path + 'statistics/' + self.S[0].model_name
        if not os.path.isdir(shelve_filename):
            os.makedirs(shelve_filename)
        shelve_filename = shelve_filename + '/model_' + str(model_id)
        with shelve.open(shelve_filename, writeback=True) as shelve_data:
            shelve_data['s'] = self.S[model_id]
            shelve_data['s1'] = self.S_1[model_id]
            shelve_data['s2'] = self.S_2[model_id]
        if print_check:
            print('\nShelving model: {}'.format(shelve_filename))

    def data_from_memory_statistics(self, model_id: int = None, print_check=False):
        # Retrieve simulated statistics model from memory
        if self.exp_no is None:
            raise Exception('Experiment not provided')

        shelve_filename = self.datadump_folder_path + 'statistics/' + self.S[0].model_name + '/model_' + str(model_id)
        with shelve.open(shelve_filename, flag='r') as shelve_data:
            self.S[model_id] = shelve_data['s']
            self.S_1[model_id] = shelve_data['s1']
            self.S_2[model_id] = shelve_data['s2']
        if not isinstance(self.S[model_id], System) or not isinstance(self.S_1[model_id], System) or not isinstance(self.S_2[model_id], System):
            raise Exception('Data type mismatch')
        if print_check:
            print('\nModel read done: {}'.format(shelve_filename))

    def dict_memory_clear(self, model_id : int = 0):
        # Clear up memory after using statistics model
        for S in (self.S, self.S_1, self.S_2):
            if model_id in S:
                del S[model_id]

    def simulate_experiment_wrapper(self, exp_no=None, print_check: bool = False) -> None:
        # Wrapper function to simulate experiment based on specified test_model
        self.initialize_system_from_experiment_number(exp_no=exp_no)

        print('\nSimulating Exp No: {}'.format(self.exp_no))

        if self.S[0].sim.test_model in self.experiment_modifications_mapper:
            self.simulate_experiment(print_check=print_check)

        elif self.S[0].sim.test_model in self.experiment_mapper_statistics:
            self.simulate_statistics_experiment(print_check=print_check)
        else:
            raise Exception('Experiment not defined')

    def simulate_experiment(self, statistics_model: int = 0, print_check: bool = False, tqdm_check: bool = True) -> None:
        # Experiment wrapper to create + modify, simulate and save test cases
        if self.S[0].sim.test_model in self.experiment_modifications_mapper:
            sim_function = self.experiment_modifications_mapper[self.S[0].sim.test_model]
        else:
            sim_function = self.experiment_modifications_mapper[self.experiment_mapper_statistics[self.S[0].sim.test_model]]

        if statistics_model > 0 and self.S[0].sim.test_model not in ['pointdistribution_openloop']:
            self.S[statistics_model] = System()
            self.initialize_system_from_experiment_number()

        if self.S[0].sim.test_model == 'pointdistribution_openloop' or self.S[0].sim.test_model in self.experiment_mapper_statistics:
            sim_function(statistics_model=statistics_model, print_check=print_check)
        else:
            sim_function(print_check=print_check)

        self.S_1[statistics_model].simulate(print_check=print_check, tqdm_check=tqdm_check)
        self.S_2[statistics_model].simulate(print_check=print_check, tqdm_check=tqdm_check)

        if statistics_model == 0:
            self.system_model_to_memory_sim_model()
            self.dict_memory_clear(model_id=0)
        else:
            self.system_model_to_memory_statistics(statistics_model)
            self.dict_memory_clear(model_id=statistics_model)

    def simulate_statistics_experiment(self, print_check: bool = False, start_idx: int = 1, number_of_samples: int = 100) -> None:
        # wrapper to simulate statistics experiments sequentially using a loop or in parallel using a ProcessPoolExecutor
        if self.S[0].sim.test_model == 'statistics_pointdistribution_openloop':
            self.system_model_to_memory_gen_model()
            idx_range = list(range(1, 1 + self.S[0].number_of_states))
        else:
            idx_range = list(range(start_idx, number_of_samples + start_idx))

        self.S[0].sim.test_model = self.experiment_mapper_statistics[self.S[0].sim.test_model]

        if self.S[0].sim.multiprocess_check:
            tqdm_check = False
            with tqdm(total=len(idx_range), ncols=100, desc='Model ID', leave=True) as pbar:
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.process_pool_workers) as executor:
                    for _ in executor.map(self.simulate_experiment, idx_range, itertools.repeat(print_check),
                                          itertools.repeat(tqdm_check)):
                        pbar.update()
        else:
            for test_no in tqdm(idx_range, desc='Simulations', ncols=100, position=0, leave=True):
                self.simulate_experiment(statistics_model=test_no, print_check=print_check, tqdm_check=True)

    def simulate_experiment_fixed_vs_self_tuning(self, statistics_model: int = 0, print_check: bool = False) -> None:
        # test_parameter is the number of optimizing greedy swap changes for self-tuning architecture
        if statistics_model > 0:
            self.S[statistics_model].initialize_system_from_experiment_parameters(self.parameter_values, self.parameter_keys)

        self.S[statistics_model].optimize_initial_architecture(print_check=print_check)

        self.S_1[statistics_model] = dc(self.S[statistics_model])
        self.S_1[statistics_model].sim.sim_model = "fixed"
        self.S_1[statistics_model].plot_name = 'fixed arch'
        self.S_2[statistics_model] = dc(self.S[statistics_model])
        self.S_2[statistics_model].sim.self_tuning_parameter = None if self.S_2[statistics_model].sim.test_parameter == 0 else self.S_2[statistics_model].sim.test_parameter
        self.S_2[statistics_model].sim.sim_model = "self_tuning"
        self.S_2[statistics_model].plot_name = 'self_tuning arch'

    def simulate_experiment_fixed_vs_self_tuning_pointdistribution_openloop(self, statistics_model: int = 0, print_check: bool = False) -> None:
        # test fixed vs self-tuning after setting the true initial state based on the statistics model
        # test_parameter is the number of optimizing greedy swap changes for self-tuning architecture
        self.S[statistics_model] = dc(self.S[0])
        self.S[statistics_model].initialize_trajectory(statistics_model - 1)

        self.S[statistics_model].optimize_initial_architecture(print_check=print_check)

        self.S_1[statistics_model] = dc(self.S[statistics_model])
        self.S_1[statistics_model].sim.sim_model = "fixed"
        self.S_1[statistics_model].plot_name = 'fixed arch'
        self.S_2[statistics_model] = dc(self.S[statistics_model])
        self.S_2[statistics_model].sim.sim_model = "self_tuning"
        self.S_2[statistics_model].sim.self_tuning_parameter = None if self.S[statistics_model].sim.test_parameter == 0 else self.S[statistics_model].sim.test_parameter
        self.S_2[statistics_model].plot_name = 'self_tuning arch'

    def simulate_experiment_self_tuning_number_of_changes(self, statistics_model: int = 0, print_check: bool = False) -> None:
        # compare 1 vs test_parameter number of changes for self-tuning architecture
        if statistics_model > 0:
            self.S[statistics_model].initialize_system_from_experiment_parameters(self.parameter_values, self.parameter_keys)

        self.S[statistics_model].sim.sim_model = "self_tuning"
        self.S[statistics_model].optimize_initial_architecture(print_check=print_check)

        self.S_1[statistics_model] = dc(self.S[statistics_model])
        self.S_1[statistics_model].sim.self_tuning_parameter = 1
        self.S_1[statistics_model].plot_name = 'self_tuning 1change'
        self.S_2[statistics_model] = dc(self.S[statistics_model])
        self.S_2[statistics_model].sim.self_tuning_parameter = None if self.S[statistics_model].sim.test_parameter == 0 else self.S[statistics_model].sim.test_parameter
        self.S_2[statistics_model].plot_name = 'self_tuning bestchange' if self.S_2[statistics_model].sim.self_tuning_parameter is None else f"self_tuning {self.S_2[statistics_model].sim.self_tuning_parameter}changes"

    def simulate_experiment_self_tuning_prediction_horizon(self, statistics_model: int = 0, print_check: bool = False) -> None:
        # compare Tp vs test_parameter*Tp simulation horizon for self-tuning architecture
        if statistics_model > 0:
            self.S[statistics_model].initialize_system_from_experiment_parameters(self.parameter_values, self.parameter_keys)

        self.S[statistics_model].sim.sim_model = "self_tuning"
        self.S[statistics_model].optimize_initial_architecture(print_check=print_check)

        self.S_1[statistics_model] = dc(self.S[statistics_model])
        self.S_1[statistics_model].plot_name = 'self_tuning Tp' + str(self.S_1[statistics_model].sim.t_predict)
        self.S_2[statistics_model] = dc(self.S[statistics_model])
        self.S_2[statistics_model].sim.t_predict *= 2 if self.S[statistics_model].sim.test_parameter is None else self.S[statistics_model].sim.test_parameter
        self.S_2[statistics_model].plot_name = 'self_tuning Tp' + str(self.S_2[statistics_model].sim.t_predict)

    def simulate_experiment_self_tuning_architecture_cost(self, statistics_model: int = 0, print_check: bool = False) -> None:
        # compare base vs test_parameter*base running and switching costs
        # Check if base-costs are non-zero for valid scaling
        if statistics_model > 0:
            self.S[statistics_model].initialize_system_from_experiment_parameters(self.parameter_values, self.parameter_keys)

        self.S[statistics_model].sim.sim_model = "self_tuning"

        self.S_1[statistics_model] = dc(self.S[statistics_model])
        self.S_1[statistics_model].optimize_initial_architecture(print_check=print_check)
        self.S_1[statistics_model].plot_name = 'self_tuning base arch cost'

        self.S_2[statistics_model] = dc(self.S[statistics_model])
        self.S_2[statistics_model].scalecost_by_test_parameter()
        self.S_2[statistics_model].optimize_initial_architecture(print_check=print_check)

    def simulate_experiment_self_tuning_architecture_cost_no_lim(self, statistics_model: int = 0, print_check: bool = False) -> None:
        # compare base vs test_parameter*base running and switching costs for unconstrainted architecture
        if statistics_model > 0:
            self.S[statistics_model].initialize_system_from_experiment_parameters(self.parameter_values, self.parameter_keys)

        self.S[statistics_model].sim.sim_model = "self_tuning"
        self.S[statistics_model].sim.self_tuning_parameter = None

        self.S_1[statistics_model] = dc(self.S[statistics_model])
        self.S_1[statistics_model].optimize_initial_architecture(print_check=print_check)
        self.S_1[statistics_model].plot_name = 'self_tuning base arch cost'

        self.S_2[statistics_model] = dc(self.S[statistics_model])
        self.S_2[statistics_model].scalecost_by_test_parameter()
        self.S_2[statistics_model].optimize_initial_architecture(print_check=print_check)

    def simulate_experiment_self_tuning_architecture_constraints(self, statistics_model: int = 0, print_check: bool = False) -> None:
        # compare [base,high] vs [test_parameter*base,high] architecture constraints
        if statistics_model > 0:
            self.S[statistics_model].initialize_system_from_experiment_parameters(self.parameter_values, self.parameter_keys)

        self.S[statistics_model].sim.sim_model = "self_tuning"
        self.S[statistics_model].sim.self_tuning_parameter = None

        # print('Check 1')
        self.S_1[statistics_model] = dc(self.S[statistics_model])
        self.S_1[statistics_model].optimize_initial_architecture(print_check=print_check)
        self.S_1[statistics_model].plot_name = r"self_tuning $|$arch$|\in [${},{}$]$".format(self.S_1[statistics_model].B.min, self.S_1[statistics_model].B.max)
        # print(f"1: {self.S_1[statistics_model].B.min} | {self.S_1[statistics_model].B.max}")

        self.S_2[statistics_model] = dc(self.S[statistics_model])
        self.S_2[statistics_model].architecture_limit_set(min_set=self.S_2[statistics_model].sim.test_parameter)
        self.S_2[statistics_model].optimize_initial_architecture(print_check=print_check)
        self.S_2[statistics_model].plot_name = r"self_tuning $|$arch$|\in [${},{}$]$".format(self.S_2[statistics_model].B.min, self.S_2[statistics_model].B.max)
        # print(f"2: {self.S_2[statistics_model].B.min} | {self.S_2[statistics_model].B.max}")

    def plot_experiment(self, exp_no=None) -> None:
        # Plotting wrapper for experiments
        self.initialize_system_from_experiment_number(exp_no=exp_no)

        print('\nPlotting Experiment No: {}'.format(self.exp_no))

        if self.S[0].sim.test_model in self.experiment_modifications_mapper:
            self.retrieve_experiment()
            self.plot_comparison_exp_no()
        elif self.S[0].sim.test_model in self.experiment_mapper_statistics:
            self.plot_statistics_exp_no()
        else:
            raise Exception('Experiment not defined')

    def plot_comparison_exp_no(self) -> None:
        # Plot wrapper for a single experiment
        fig = plt.figure(figsize=(6, 8), tight_layout=True)
        outer_grid = gs.GridSpec(2, 2, figure=fig, height_ratios=[1, 7], width_ratios=[1, 1])

        ax_exp_legend = fig.add_subplot(outer_grid[0, 0])
        ax_eval = fig.add_subplot(outer_grid[0, 1])

        time_grid = gs.GridSpecFromSubplotSpec(7, 1, subplot_spec=outer_grid[1, :], hspace=0.2, height_ratios=[1, 1, 1, 0.7, 1, 0.7, 0.7])

        ax_cost = fig.add_subplot(time_grid[0, 0])
        ax_state = fig.add_subplot(time_grid[1, 0], sharex=ax_cost)
        ax_B_scatter = fig.add_subplot(time_grid[2, 0], sharex=ax_cost)
        ax_B_count = fig.add_subplot(time_grid[3, 0], sharex=ax_cost)
        ax_C_scatter = fig.add_subplot(time_grid[4, 0], sharex=ax_cost)
        ax_C_count = fig.add_subplot(time_grid[5, 0], sharex=ax_cost)
        ax_compute_time = fig.add_subplot(time_grid[6, 0], sharex=ax_cost)

        self.S_1[0].plot_openloop_eigvals(ax_in=ax_eval)

        self.S_1[0].plot_cost(ax_in=ax_cost)
        self.S_2[0].plot_cost(ax_in=ax_cost, set_details_flag=True)
        ax_cost.tick_params(axis="x", labelbottom=False)
        # ax_cost.yaxis.set_major_locator(MaxNLocator(2))

        plt_title = None if not self.plot_title_check else f"Experiment No: {self.exp_no}"
        ax_exp_legend.legend(handles=[mpatches.Patch(color=self.S_1[0].plot.plot_parameters[self.S_1[0].plot.plot_system]['c'], label=self.S_1[0].plot_name),
                                      mpatches.Patch(color=self.S_2[0].plot.plot_parameters[self.S_2[0].plot.plot_system]['c'], label=self.S_2[0].plot_name)],
                             loc='center', ncol=1, title=plt_title)
        ax_exp_legend.axis('off')

        self.S_1[0].plot_states(ax_in=ax_state)
        self.S_2[0].plot_states(ax_in=ax_state, set_details_flag=True)
        ax_state.tick_params(axis="x", labelbottom=False)

        self.S_1[0].plot_architecture_history(arch='B', ax_in=ax_B_scatter)
        self.S_2[0].plot_architecture_history(arch='B', ax_in=ax_B_scatter)
        ax_B_scatter.set_ylabel('Actuator\nPosition\n' + r'$\mathcal{A}_t$')
        ax_B_scatter.tick_params(axis="x", labelbottom=False)

        self.S_1[0].plot_architecture_history(arch='C', ax_in=ax_C_scatter)
        self.S_2[0].plot_architecture_history(arch='C', ax_in=ax_C_scatter)
        ax_C_scatter.set_ylabel('Sensor\nPosition\n' + r'$S_t$')
        ax_C_scatter.tick_params(axis="x", labelbottom=False)

        for lim_val in [self.S[0].B.min, self.S[0].B.max]:
            ax_B_count.axhline(y=lim_val, color='tab:gray', ls='dashdot', alpha=0.5)
            ax_C_count.axhline(y=lim_val, color='tab:gray', ls='dashdot', alpha=0.5)

        self.S_1[0].plot_architecture_count(ax_in=ax_B_count, arch='B')
        self.S_2[0].plot_architecture_count(ax_in=ax_B_count, arch='B')
        ax_B_count.set_ylabel('Actuator\nCount\n' + r'$|\mathcal{A}_t|$')
        ax_B_count.tick_params(axis="x", labelbottom=False)

        self.S_1[0].plot_architecture_count(ax_in=ax_C_count, arch='C')
        self.S_2[0].plot_architecture_count(ax_in=ax_C_count, arch='C')
        ax_C_count.set_ylabel('Sensor\nCount\n' + r'$|S_t|$')
        ax_C_count.tick_params(axis="x", labelbottom=False)

        self.S_1[0].plot_compute_time(ax_in=ax_compute_time)
        self.S_2[0].plot_compute_time(ax_in=ax_compute_time)
        ax_compute_time.set_yscale('log')
        y_lims = list(ax_compute_time.get_ylim())
        ax_compute_time.locator_params(axis='y', subs=(1, ))
        ax_compute_time.set_ylim(10 ** np.floor(np.log10(y_lims[0])), 10 ** np.ceil(np.log10(y_lims[1])))
        # ax_compute_time.yaxis.set_major_locator(MaxNLocator(2))

        ax_compute_time.set_ylabel('Compute\nTime (s)')
        ax_compute_time.set_xlabel(r'Time $t$')
        ax_compute_time.set_xlim(0, self.S[0].sim.t_simulate)

        plt.show()

        save_path = self.image_save_folder_path + 'exp' + str(self.exp_no) + '.pdf'
        fig.savefig(save_path, dpi=fig.dpi)
        print('\nImage saved: {}\n'.format(save_path))

    def plot_statistics_exp_no(self) -> None:
        # Plot wrapper for statistics experiment
        self.initialize_system_from_experiment_number()
        sim_range = range(1, self.S[0].number_of_states + 1) if self.S[0].sim.test_model == 'statistics_pointdistribution_openloop' else range(1, 100 + 1)

        fig = plt.figure(tight_layout=True)
        grid_outer = gs.GridSpec(2, 1, figure=fig, height_ratios=[3, 1])
        grid_inner = gs.GridSpecFromSubplotSpec(2, 2, subplot_spec=grid_outer[0, 0], hspace=0.2, width_ratios=[1, 1], height_ratios=[1, 2])

        ax_exp_legend = fig.add_subplot(grid_inner[0, 0])
        ax_eigmodes = fig.add_subplot(grid_inner[0, 1])
        ax_cost = fig.add_subplot(grid_inner[1, :])

        grid_architecture = gs.GridSpecFromSubplotSpec(1, 5, subplot_spec=grid_outer[1, :], hspace=0)
        ax_architecture_B_count = fig.add_subplot(grid_architecture[0, 0])
        ax_architecture_C_count = fig.add_subplot(grid_architecture[0, 1], sharey=ax_architecture_B_count)
        ax_architecture_B_change = fig.add_subplot(grid_architecture[0, 2], sharey=ax_architecture_B_count)
        ax_architecture_C_change = fig.add_subplot(grid_architecture[0, 3], sharey=ax_architecture_B_count)
        ax_architecture_compute_time = fig.add_subplot(grid_architecture[0, 4], sharey=ax_architecture_B_count)

        cstyle = ['tab:blue', 'tab:orange', 'black']
        lstyle = ['dashdot', 'dashed']
        mstyle = ['o', '+', 'x']

        cost_min_1, cost_min_2 = np.inf * np.ones(self.S[0].sim.t_simulate), np.inf * np.ones(self.S[0].sim.t_simulate)
        cost_max_1, cost_max_2 = np.zeros(self.S[0].sim.t_simulate), np.zeros(self.S[0].sim.t_simulate)
        sample_cost_1, sample_cost_2 = np.zeros(self.S[0].sim.t_simulate), np.zeros(self.S[0].sim.t_simulate)
        compute_time_1, compute_time_2 = [], []
        sample_eig = np.zeros(self.S[0].number_of_states)
        arch_change_1 = {'B': [], 'C': []}
        arch_change_2 = {'B': [], 'C': []}
        arch_count_1 = {'B': [], 'C': []}
        arch_count_2 = {'B': [], 'C': []}

        m1_name, m2_name = '', ''

        sample_ID = np.random.choice(sim_range)

        for model_no in tqdm(sim_range, ncols=100, desc='Model ID'):
            self.data_from_memory_statistics(model_no)

            cost_min_1, cost_max_1, compute_time_1, arch_change_1, arch_count_1 = statistics_data_parser(self.S_1[model_no], cost_min_1, cost_max_1, compute_time_1, arch_change_1, arch_count_1)
            cost_min_2, cost_max_2, compute_time_2, arch_change_2, arch_count_2 = statistics_data_parser(self.S_2[model_no], cost_min_2, cost_max_2, compute_time_2, arch_change_2, arch_count_2)

            ax_eigmodes.scatter(range(1, self.S[model_no].number_of_states + 1), np.sort(np.abs(self.S[model_no].A.open_loop_eig_vals)),
                                marker=mstyle[0], s=10, color=cstyle[0], alpha=float(1 / self.S[0].number_of_states))

            if sample_ID == model_no:
                sample_cost_1 = list(
                    itertools.accumulate(self.S_1[model_no].list_from_dict_key_time(self.S_1[model_no].trajectory.cost.true)))
                sample_cost_2 = list(
                    itertools.accumulate(self.S_2[model_no].list_from_dict_key_time(self.S_2[model_no].trajectory.cost.true)))
                sample_eig = np.sort(np.abs(self.S[model_no].A.open_loop_eig_vals))
                m1_name = self.S_1[model_no].plot_name
                m2_name = self.S_2[model_no].plot_name

            self.dict_memory_clear(model_id=model_no)

        plt_title = None if not self.plot_title_check else f"Experiment No: {self.exp_no}"

        ax_exp_legend.legend(handles=[mpatches.Patch(color=cstyle[0], label=r'$M_1$:' + m1_name),
                                      mpatches.Patch(color=cstyle[1], label=r'$M_2$:' + m2_name)],
                             loc='center', title = plt_title)
        ax_exp_legend.axis('off')

        ax_cost.fill_between(range(0, self.S[0].sim.t_simulate), cost_min_1, cost_max_1, color=cstyle[0], alpha=0.4)
        ax_cost.fill_between(range(0, self.S[0].sim.t_simulate), cost_min_2, cost_max_2, color=cstyle[1], alpha=0.4)
        ax_cost.plot(range(0, self.S[0].sim.t_simulate), sample_cost_1, color=cstyle[2], ls=lstyle[0], linewidth=1)
        ax_cost.plot(range(0, self.S[0].sim.t_simulate), sample_cost_2, color=cstyle[2], ls=lstyle[1], linewidth=1)
        ax_cost.legend(handles=[mlines.Line2D([], [], color=cstyle[2], ls=lstyle[0], label='Sample ' + r'$M_1$'),
                                mlines.Line2D([], [], color=cstyle[2], ls=lstyle[1], label='Sample ' + r'$M_2$')],
                       loc='upper left', ncols=2)
        ax_cost.set_yscale('log')
        ax_cost.set_xlabel(r'Time $t$')
        ax_cost.set_ylabel(r'Cost $J_t$')
        ax_cost.set_xlim(0, self.S[0].sim.t_simulate)

        ax_eigmodes.scatter(range(1, self.S[0].number_of_states + 1), sample_eig, marker=mstyle[2], color=cstyle[2], s=10)
        ax_eigmodes.hlines(1, xmin=1, xmax=self.S[0].number_of_states, colors=cstyle[2], ls=lstyle[1])
        # ax_eigmodes.set_xlabel('Mode ' + r'$i$')
        ax_eigmodes.set_ylabel(r'$|\lambda_i(A)|$')
        ax_eigmodes.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False)
        ax_eigmodes.legend(
            handles=[mlines.Line2D([], [], color=cstyle[2], marker=mstyle[2], linewidth=0, label='Sample'),
                     mlines.Line2D([], [], color=cstyle[0], marker=mstyle[0], linewidth=0, label='Modes')],
            loc='upper left')

        a1 = ax_architecture_B_change.boxplot([arch_change_2['B'], arch_change_1['B']],
                                              labels=[r'$M_2$', r'$M_1$'], vert=False, widths=0.5)
        a2 = ax_architecture_C_change.boxplot([arch_change_2['C'], arch_change_1['C']],
                                              labels=[r'$M_2$', r'$M_1$'], vert=False, widths=0.5)
        a3 = ax_architecture_B_count.boxplot([arch_count_2['B'], arch_count_1['B']],
                                             labels=[r'$M_2$', r'$M_1$'], vert=False, widths=0.5)
        a4 = ax_architecture_C_count.boxplot([arch_count_2['C'], arch_count_1['C']],
                                             labels=[r'$M_2$', r'$M_1$'], vert=False, widths=0.5)
        a5 = ax_architecture_compute_time.boxplot([compute_time_2, compute_time_1],
                                                  labels=[r'$M_2$', r'$M_1$'], vert=False, widths=0.5)

        for ax in (ax_architecture_B_count, ax_architecture_C_count):
            ax.xaxis.set_major_locator(MaxNLocator(min_n_ticks=1, integer=True))

        for ax in (ax_architecture_B_change, ax_architecture_C_change):
            ax.xaxis.set_major_locator(MaxNLocator(min_n_ticks=1, integer=True))

        for bplot in (a1, a2, a3, a4, a5):
            for patch, color in zip(bplot['medians'], [cstyle[1], cstyle[0]]):
                patch.set_color(color)

        # for a in (ax_architecture_B_count, ax_architecture_C_count, ax_architecture_B_change, ax_architecture_C_change):
        #     x_lim_g = a.get_xlim()
        #     a.set_xlim(np.floor(x_lim_g[0]), np.ceil(x_lim_g[1]))

        for a in (ax_architecture_C_count, ax_architecture_B_change, ax_architecture_C_change, ax_architecture_compute_time):
            a.tick_params(axis='y', labelleft=False, left=False)

        ax_architecture_compute_time.set_xscale('log')
        x_lims = list(ax_architecture_compute_time.get_xlim())
        # ax_architecture_compute_time.locator_params(axis='x', subs=(1, ))
        ax_architecture_compute_time.set_xlim(10 ** np.floor(np.log10(x_lims[0])), 10 ** np.ceil(np.log10(x_lims[1])))
        # ax_architecture_compute_time.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=2))
        # ax_architecture_compute_time.xaxis.set_major_formatter(LogFormatter(labelOnlyBase=True))
        ax_architecture_compute_time.set_xticks([10 ** np.floor(np.log10(x_lims[0])),  10 ** np.ceil(np.log10(x_lims[1]))])
        ax_architecture_compute_time.xaxis.set_minor_formatter(NullFormatter())

        ax_architecture_B_count.set_xlabel('Avg ' + r'$\mathcal{A}_t$' + '\nSize')
        ax_architecture_C_count.set_xlabel('Avg ' + r'$S_t$' + '\nSize')
        ax_architecture_B_change.set_xlabel('Avg ' + r'$\mathcal{A}_t - \mathcal{A}_{t-1}$' + '\nChanges')
        ax_architecture_C_change.set_xlabel('Avg ' + r'$S_t - S_{t-1}$' + '\nChanges')
        ax_architecture_compute_time.set_xlabel('Avg Compute \n Time (s)')

        plt.show()

        save_path = self.image_save_folder_path + 'exp' + str(self.exp_no) + '.pdf'
        fig.savefig(save_path, dpi=fig.dpi)
        print('\nImage saved: {}\n'.format(save_path))


def coin_toss() -> bool:
    # Generate True/False with equal probability
    return np.random.default_rng().random() > 0.5


def compare_lists(list1: list, list2: list) -> dict:
    # Find elements unique to each list and common to both lists
    return {'only1': [k for k in list1 if k not in list2], 'only2': [k for k in list2 if k not in list1],
            'both': [k for k in list1 if k in list2]}


def architecture_iterator(arch=None) -> list:
    # Iterate over either B, C or both
    if type(arch) == list and len(arch) == 1:
        arch = arch[0]
    arch = [arch] if arch in ['B', 'C'] else ['B', 'C'] if (
                arch is None or arch == ['B', 'C']) else [] if arch == 'skip' else 'Error'
    if arch == 'Error':
        raise ArchitectureError
    return arch


def normalize_columns_of_matrix(A_mat: np.ndarray) -> np.ndarray:
    # Normalize columns of a matrix - used for eigenvector matrix normalization
    for i in range(0, np.shape(A_mat)[0]):
        if np.linalg.norm(A_mat[:, i]) != 0:
            A_mat[:, i] /= np.linalg.norm(A_mat[:, i])
    return A_mat


class PlotParameters:
    # Plot values and parameters - updated when simulated systems are read from memory
    def __init__(self, sys_stage: int = 0):
        self.plot_system = sys_stage
        self.predicted_cost, self.true_cost = [], []
        self.x_2norm, self.x_estimate_2norm, self.error_2norm = [], [], []
        self.plot_parameters = \
            {1: {'node': 'tab:blue', 'B': 'tab:orange', 'C': 'tab:green', 'm': 'x', 'c': 'tab:blue', 'ms': 20,
                 'ls': 'solid'},
             2: {'node': 'tab:blue', 'B': 'tab:orange', 'C': 'tab:green', 'm': 'o', 'c': 'tab:orange', 'ms': 20,
                 'ls': 'dashed'}}
        self.network_plot_limits = []
        self.B_history = [[], []]
        self.C_history = [[], []]
        self.states_graph, self.state_positions = netx.DiGraph(), {}
        self.actutator_positions, self.sensor_positions = {}, {}


class System:
    class Dynamics:
        def __init__(self):
            # Parameters assigned from function file
            self.rho = 1  # Scaling factor for eigenvalues
            self.network_model = 'rand'  # Network adjacency matrix
            self.network_parameter = 0  # Parameter for network adjacency matrix
            self.second_order = False  # Check for second order states - each node has 2 states associated with it
            self.second_order_scaling_factor = 1  # Scaling factor for second order equation
            self.second_order_network_type = 1  # Network type of second order states

            # Default parameter
            self.self_loop = True  # Self-loops within the adjacency matrix

            # Evaluated
            self.open_loop_eig_vals = 0  # Vector of second order states
            self.open_loop_eig_vecs = np.zeros((0, 0))
            self.adjacency_matrix = 0  # Adjacency matrix
            self.number_of_non_stable_modes = 0  # Number of unstable modes with magnitude >= 1
            self.A_mat = np.zeros((0, 0))  # Open-loop dynamics matrix
            self.A_augmented_mat = np.zeros((0, 0))  # Open-loop augmented dynamics matrix for current t

        def display_values(self) -> None:
            for var, value in vars(self).items():
                print(f"{var} = {value}")

    class Architecture:
        def __init__(self):
            # Parameters assigned from function file
            self.second_order_architecture_type = 0  # Type of second-order architecture - which states do it control or estimate
            self.min = 1  # Minimum architecture bounds
            self.max = 1  # Maximum architecture bounds
            self.R2 = 0  # Cost on running architecture
            self.R3 = 0  # Cost on switching architecture

            # Calculated terms
            self.available_indices = []  # Indices of available architecture
            self.available_vectors = {}  # Vectors associated with available architecture
            self.number_of_available = 0  # Count of available architecture
            self.active_set = []  # Set of indices of active architecture
            self.active_matrix = np.zeros((0, 0))  # Matrix of active architecture
            self.Q = np.zeros((0, 0))  # Cost on states/Process noise covariance
            self.R1 = np.zeros((0, 0))  # Cost on active actuators/Measurement noise covariance
            self.R1_reference = np.zeros((0, 0))  # Cost on available actuators
            self.indicator_vector_current = np.zeros(
                self.number_of_available)  # {1, 0} binary vector of currently active architecture - to compute running/switching costs
            self.indicator_vector_history = np.zeros(
                self.number_of_available)  # {1, 0} binary vector of previously active architecture - to compute switching costs
            self.history_active_set = {}  # Record of active architecture over simulation horizon
            self.change_count = 0  # Count of number of changes in architecture over simulation horizon
            self.active_count = []  # Count size of active architecture at each timestep of the simulation horizon
            self.recursion_matrix = {}  # Recursive cost matrix/estimation error covariance over the prediction horizon
            self.gain = {}  # Gains calculated over the prediction horizon for the fixed architecture

        def display_values(self) -> None:
            for var, value in vars(self).items():
                print(f"{var} = {value}")

    class Disturbance:
        def __init__(self):
            # Parameters assigned from function file
            self.W_scaling = 1  # Scaling factor for process noise covariance
            self.V_scaling = 1  # Scaling factor for measurement noise covariance
            self.noise_model = None  # Noise model for targeted un-modelled disturbances
            self.disturbance_step = 0  # Number of steps between un-modelled disturbances
            self.disturbance_number = 0  # Number of states affected by un-modelled disturbances
            self.disturbance_magnitude = 0  # Scaling factor of un-modelled disturbances

            # Calculated terms
            self.W = np.zeros((0, 0))  # Process noise covariance
            self.V = np.zeros((0, 0))  # Measurement noise covariance
            self.w_gen = {}  # Realization of process noise
            self.v_gen = {}  # Realization of measurement noise
            self.F_augmented = np.zeros((0, 0))  # augmented matrix for noise at current time

        def display_values(self) -> None:
            for var, value in vars(self).items():
                print(f"{var} = {value}")

    class Simulation:
        def __init__(self):
            # Parameters assigned from function file
            self.experiment_number = 0  # Experiment number based on parameter sheet
            self.t_predict = int(10)  # Prediction time horizon
            self.sim_model = None  # Simulation model of architecture
            self.self_tuning_parameter = 1  # Number of self tuning changes per step
            self.test_model = None  # Test case
            self.test_parameter = None  # Test case
            self.multiprocess_check = False  # Boolean to determine if multiprocess mapping is used or not for choices

            # Constant parameters
            self.t_simulate = int(100)  # Simulation time horizon
            self.t_current = 0  # Current time-step of simulation

        def display_values(self) -> None:
            for var, value in vars(self).items():
                print(f"{var} = {value}")

    class Trajectory:
        def __init__(self):
            # Parameters assigned from function file
            self.X0_scaling = 1  # Scaling factor of initial state

            # Calculated terms
            self.X0_covariance = np.zeros((0, 0))  # Initial state covariance
            self.x = {}  # True state trajectory
            self.x_estimate = {}  # Estimates state trajectory
            self.X_augmented = {}  # augmented state trajectory
            self.error = {}  # Estimation error trajectory
            self.control_cost_matrix = {}  # Control cost matrix at each timestep
            self.estimation_matrix = {}  # Estimation error covariance matrix at each timestep
            self.error_2norm = {}  # 2-norm of estimation error trajectory
            self.cost = System.Cost()  # Cost variables
            self.computation_time = {}  # Computation time for greedy optimization at each simulation timestep

        def display_values(self) -> None:
            for var, value in vars(self).items():
                print(f"{var} = {value}")

    class Cost:
        def __init__(self):
            # Parameters assigned from function file
            self.metric_control = 1  # Metric function to evaluate control costs
            self.metric_running = 1  # Metric function to evaluate running costs
            self.metric_switching = 1  # Metric function to evaluate switching costs

            # Calculated terms
            self.running = 0  # Running cost at current timestep
            self.switching = 0  # Switching cost at current timestep
            self.control = 0  # Control cost at current timestep
            self.predicted = {}  # Predicted total stage cost trajectory
            self.true = {}  # True total stage cost trajectory
            self.initial = []  # Costs for initial architecture optimization
            self.predicted_matrix = {}  # Cost matrix over the prediction horizon

        def display_values(self) -> None:
            for var, value in vars(self).items():
                print(f"{var} = {value}")

    def __init__(self):
        self.number_of_nodes = 20  # Number of nodes in the network
        self.number_of_states = 20  # Number of state in the network (affected by second_order dynamics)
        self.A = self.Dynamics()
        self.B = self.Architecture()
        self.C = self.Architecture()
        self.disturbance = self.Disturbance()
        self.sim = self.Simulation()
        self.trajectory = self.Trajectory()
        self.plot = None

        self.model_name = ''
        self.plot_name = None

    def copy_from_system(self, ref_sys):
        # Dict based deep copy from ref_sys
        if not isinstance(ref_sys, System):
            raise Exception('Check system')
        self.__dict__.update(ref_sys.__dict__)

    def initialize_system_from_experiment_parameters(self, experiment_parameters, experiment_keys) -> None:
        # Create model from experiment parameters

        parameters = dict(zip(experiment_keys, experiment_parameters))

        self.sim.experiment_number = parameters['experiment_no']
        self.number_of_nodes = parameters['number_of_nodes']
        self.number_of_states = parameters['number_of_nodes']
        self.A.rho = parameters['rho']
        self.A.network_model = parameters['network_model']
        self.A.network_parameter = parameters['network_parameter']

        self.adjacency_matrix_initialize()

        self.A.second_order = parameters['second_order']
        if self.A.second_order:
            self.number_of_states = self.number_of_nodes * 2
            self.A.second_order_network_type = parameters['second_order_network']
            self.B.second_order_architecture_type = parameters['second_order_architecture']
            self.C.second_order_architecture_type = parameters['second_order_architecture']

        self.rescale_wrapper()

        self.sim.t_predict = parameters['prediction_time_horizon']
        self.sim.multiprocess_check = parameters['multiprocessing']

        self.sim.test_model = None if parameters['test_model'] == 'None' else parameters['test_model']
        self.sim.test_parameter = None if parameters['test_parameter'] == 0 else int(parameters['test_parameter'])

        parameters['disturbance_model'] = None if parameters['disturbance_model'] == 'None' else parameters[
            'disturbance_model']

        self.disturbance.W_scaling = parameters['W_scaling']
        self.disturbance.V_scaling = parameters['V_scaling']
        self.disturbance.noise_model = parameters['disturbance_model']
        if self.disturbance.noise_model is not None:
            self.disturbance.disturbance_step = parameters['disturbance_step']
            self.disturbance.disturbance_number = parameters['disturbance_number']
            self.disturbance.disturbance_magnitude = parameters['disturbance_magnitude']
        self.initialize_disturbance()

        self.B.R2 = parameters['B_run_cost']
        self.C.R2 = parameters['C_run_cost']
        self.B.R3 = parameters['B_switch_cost']
        self.C.R3 = parameters['C_switch_cost']

        self.architecture_limit_set(min_set=parameters['architecture_constraint_min'],
                                    max_set=parameters['architecture_constraint_max'])

        self.initialize_available_vectors_as_basis_vectors()

        self.B.Q = np.identity(self.number_of_states) * parameters['Q_cost_scaling']
        self.B.R1_reference = np.identity(self.B.number_of_available) * parameters['R_cost_scaling']
        self.C.Q = self.disturbance.W
        self.C.R1_reference = self.disturbance.V

        self.initialize_random_architecture_active_set(parameters['initial_architecture_size'])
        self.architecture_update_to_history_indicator_matrix_from_active_set()

        self.trajectory.X0_scaling = parameters['X0_scaling']
        self.initialize_trajectory()
        self.prediction_gains()
        self.cost_prediction_wrapper()

        self.model_namer()

    def model_namer(self, namer_type=1, name_extension: str = None) -> None:
        if namer_type == 1:
            self.model_name = 'model_exp' + str(self.sim.experiment_number)
        elif namer_type == 2:
            self.model_name = 'model_n' + str(int(self.number_of_nodes)) + '_net' + self.A.network_model
            if self.A.rho is None:
                self.model_name = self.model_name + '_rhoNone'
            else:
                self.model_name = self.model_name + '_rho' + str(np.round(self.A.rho, decimals=2))
            if self.A.second_order:
                self.model_name = self.model_name + '_secondorder'
            self.model_name = self.model_name + '_arch' + str(self.B.max) + '_Tp' + str(self.sim.t_predict)
            if self.disturbance.noise_model is not None:
                self.model_name = self.model_name + '_' + self.disturbance.noise_model
            if self.sim.sim_model is not None:
                self.model_name = self.model_name + '_' + self.sim.sim_model
            if self.sim.test_model is not None:
                self.model_name = self.model_name + '_' + self.sim.test_model
        else:
            raise Exception('Not Implemented')
        if name_extension is not None:
            self.model_name = self.model_name + '_' + name_extension

    def display_all_parameters(self) -> None:
        print('Model name = {}'.format(self.model_name))
        print('Number of Nodes = {}'.format(self.number_of_nodes))
        print('Number of States = {}'.format(self.number_of_states))
        print('A')
        self.A.display_values()
        print('B')
        self.B.display_values()
        print('C')
        self.C.display_values()
        print('disturbances')
        self.disturbance.display_values()
        print('sim')
        self.sim.display_values()
        self.trajectory.display_values()

    def adjacency_matrix_initialize(self) -> None:
        # Create adjacency matrix based on network_model and network_parameters
        if self.A.network_model not in ['rand', 'ER', 'BA', 'path', 'cycle', 'eval_squeeze', 'eval_bound']:
            raise Exception('Network model not defined: {}'.format(self.A.network_model))

        connected_graph_check = False
        G = netx.Graph()

        while not connected_graph_check:

            if self.A.network_model == 'ER':
                if self.A.network_parameter is None:
                    self.A.network_parameter = 0.3
                elif self.A.network_parameter < 0 or self.A.network_parameter > 1:
                    raise Exception('Check network model parameter')
                G = netx.generators.random_graphs.erdos_renyi_graph(self.number_of_nodes, self.A.network_parameter)

            elif self.A.network_model == 'BA':
                if self.A.network_parameter is None:
                    self.A.network_parameter = 2
                elif self.A.network_parameter <= 0 or self.A.network_parameter >= self.number_of_nodes:
                    raise Exception('Check network model parameter')
                G = netx.generators.random_graphs.barabasi_albert_graph(self.number_of_nodes, self.A.network_parameter)

            elif self.A.network_model == 'rand':
                self.A.network_parameter = None
                A = np.random.rand(self.number_of_nodes, self.number_of_nodes)
                G = netx.from_numpy_array(A)

            elif self.A.network_model == 'path':
                G = netx.generators.classic.path_graph(self.number_of_nodes)

            elif self.A.network_model == 'cycle':
                G = netx.generators.classic.cycle_graph(self.number_of_nodes)

            elif self.A.network_model in ['eval_squeeze', 'eval_bound']:
                if self.A.network_parameter is None:
                    self.A.network_parameter = 0.1
                elif self.A.network_parameter < 0 or self.A.network_parameter > 1:
                    raise Exception('Check network model parameter')

                A = np.random.rand(self.number_of_nodes, self.number_of_nodes)
                A = A.T + A
                _, V_mat = np.linalg.eig(A)
                V_mat = normalize_columns_of_matrix(V_mat)

                if self.A.network_model == 'eval_squeeze':
                    e = 2 * self.A.network_parameter * (0.5 - np.random.default_rng().random(self.number_of_nodes))
                    e = [1 + k if coin_toss() else 1 - k for k in e]
                elif self.A.network_model == 'eval_bound':
                    e = 1 - self.A.network_parameter * np.random.default_rng().random(self.number_of_nodes)
                else:
                    raise Exception('Check Network model')

                self.A.self_loop = None
                self.A.rho = None

                e = np.array([i * -1 if coin_toss() else i for i in e])
                A = V_mat @ np.diag(e) @ np.linalg.inv(V_mat)
                G = netx.from_numpy_array(A)

            else:
                raise Exception('Check Network model')

            connected_graph_check = netx.algorithms.components.is_connected(G)

        self.A.adjacency_matrix = netx.to_numpy_array(G)

        if self.A.self_loop:
            self.A.adjacency_matrix += np.identity(self.number_of_nodes)

    def evaluate_modes(self) -> None:
        # Find open loop eigenvalues, eigenvectors and stable/unstable modes
        self.A.open_loop_eig_vals = np.sort(np.abs(np.linalg.eigvals(self.A.A_mat)))
        _, self.A.open_loop_eig_vecs = np.linalg.eig(self.A.A_mat)
        self.A.open_loop_eig_vecs = normalize_columns_of_matrix(self.A.open_loop_eig_vecs)
        self.A.number_of_non_stable_modes = len([e for e in self.A.open_loop_eig_vals if e >= 1])

    def rescale(self) -> None:
        # Scale dynamics matrix to rho*normalized_spectrum
        if self.A.rho is not None:
            self.A.A_mat = self.A.rho * self.A.adjacency_matrix / np.max(
                np.abs(np.linalg.eigvals(self.A.adjacency_matrix)))
        else:
            self.A.A_mat = self.A.adjacency_matrix

    def second_order_matrix(self) -> None:
        # second order matrix of either first order or second order states
        if self.A.second_order:
            if self.A.second_order_network_type == 1:

                self.A.A_mat = np.block([[self.A.A_mat, np.zeros_like(self.A.A_mat)],
                                         [self.A.second_order_scaling_factor * np.identity(self.number_of_nodes),
                                          self.A.second_order_scaling_factor * np.identity(self.number_of_nodes)]])
            elif self.A.second_order_network_type == 2:
                self.A.A_mat = np.block([[np.identity(self.number_of_nodes), np.zeros_like(self.A.A_mat)],
                                         [self.A.second_order_scaling_factor * np.identity(self.number_of_nodes),
                                          self.A.second_order_scaling_factor * self.A.A_mat]])
            else:
                raise SecondOrderError()

    def rescale_wrapper(self) -> None:
        # wrapper to define dynamics matrix, rescale and evaluate open-loop dynamics
        self.rescale()
        self.second_order_matrix()
        self.evaluate_modes()

    def initialize_active_matrix(self, arch=None) -> None:
        # Initialize architecture matrix as zeros
        for a in architecture_iterator(arch):
            if a == 'B':
                self.B.active_matrix = np.zeros((self.number_of_states, len(self.B.active_set)))
            else:  # self.architecture_type == 'C':
                self.C.active_matrix = np.zeros((len(self.C.active_set), self.number_of_states))

    def initialize_available_vectors_as_basis_vectors(self, arch=None) -> None:
        # Set all available architecture as normalized basis vectors of 1s and 0s - rows or column vectors suitably
        for a in architecture_iterator(arch):
            set_mat = np.identity(self.number_of_states)

            if a == 'B':
                if self.A.second_order:
                    if self.B.second_order_architecture_type == 1:
                        set_mat = set_mat[:, :self.number_of_nodes]
                    elif self.B.second_order_architecture_type == 2:
                        set_mat = set_mat[:, self.number_of_nodes:]
                    else:
                        raise SecondOrderError
                self.B.available_vectors = {i: set_mat[:, i] for i in range(0, self.number_of_nodes)}
                self.B.available_indices = [i for i in self.B.available_vectors]
                self.B.number_of_available = len(self.B.available_indices)
                self.B.indicator_vector_history = np.zeros(self.B.number_of_available, dtype=int)
                self.B.indicator_vector_current = np.zeros(self.B.number_of_available, dtype=int)

            else:  # a == 'C'
                if self.A.second_order:
                    if self.C.second_order_architecture_type == 1:
                        set_mat = set_mat[:self.number_of_nodes, :]
                    elif self.C.second_order_architecture_type == 2:
                        set_mat = set_mat[self.number_of_nodes:, :]
                    else:
                        raise SecondOrderError
                self.C.available_vectors = {i: set_mat[i, :] for i in range(0, self.number_of_nodes)}
                self.C.available_indices = [i for i in self.C.available_vectors]
                self.C.number_of_available = len(self.B.available_indices)
                self.C.indicator_vector_history = np.zeros(self.C.number_of_available, dtype=int)
                self.C.indicator_vector_current = np.zeros(self.C.number_of_available, dtype=int)

    def initialize_random_architecture_active_set(self, initialize_random: int, arch=None) -> None:
        # Randomize initial architecture of fixed size from available choices
        for a in architecture_iterator(arch):
            if a == 'B':
                self.B.active_set = list(np.sort(
                    np.random.default_rng().choice(self.B.available_indices, size=initialize_random, replace=False)))
            else:  # a == 'C'
                self.C.active_set = list(np.sort(
                    np.random.default_rng().choice(self.C.available_indices, size=initialize_random, replace=False)))

    def initialize_disturbance(self) -> None:
        # Generate realizations of normally distributed process and measurement noises
        # Generate realizations of unmodelled process and/or measurement noise
        self.disturbance.W = np.identity(self.number_of_states) * self.disturbance.W_scaling
        self.disturbance.V = np.identity(self.number_of_states) * self.disturbance.V_scaling

        self.disturbance.w_gen = {
            t: np.random.default_rng().multivariate_normal(np.zeros(self.number_of_states), self.disturbance.W) for t in
            range(0, self.sim.t_simulate)}
        self.disturbance.v_gen = {
            t: np.random.default_rng().multivariate_normal(np.zeros(self.number_of_states), self.disturbance.V) for t in
            range(0, self.sim.t_simulate)}

        if self.disturbance.noise_model in ['process', 'measurement', 'combined']:
            if self.disturbance.disturbance_number == 0 or self.disturbance.disturbance_magnitude == 0 or self.disturbance.disturbance_step == 0:
                raise Exception('Check disturbance parameters')
            for t in range(0, self.sim.t_simulate, self.disturbance.disturbance_step):
                if self.disturbance.noise_model in ['process', 'combined']:
                    self.disturbance.w_gen[t][
                        np.random.default_rng().choice(self.number_of_states, self.disturbance.disturbance_number,
                                                       replace=False)] = self.disturbance.disturbance_magnitude * np.array(
                        [coin_toss() for _ in range(0, self.disturbance.disturbance_number)])
                if self.disturbance.noise_model in ['measurement', 'combined']:
                    self.disturbance.v_gen[t] = self.disturbance.disturbance_magnitude * np.array(
                        [coin_toss() for _ in range(0, self.number_of_states)])

    def initialize_trajectory(self, x0_idx=None) -> None:
        # Initialize true, estimate and error state vectors
        # Initializes along specified eigenvector of open-loop dynamics
        self.trajectory.X0_covariance = np.identity(self.number_of_states) * self.trajectory.X0_scaling

        if x0_idx is None:
            self.trajectory.x = {0: np.random.default_rng().multivariate_normal(np.zeros(self.number_of_states),
                                                                                self.trajectory.X0_covariance)}
        else:
            self.trajectory.x = {0: self.A.open_loop_eig_vecs[:, x0_idx] * self.trajectory.X0_scaling}

        self.trajectory.x_estimate = {0: np.random.default_rng().multivariate_normal(np.zeros(self.number_of_states),
                                                                                     self.trajectory.X0_covariance)}

        self.trajectory.X_augmented = {0: np.concatenate((self.trajectory.x[0], self.trajectory.x_estimate[0]))}

        self.trajectory.control_cost_matrix = {}
        self.trajectory.estimation_matrix = {0: np.identity(self.number_of_states)}

        self.trajectory.error = {0: self.trajectory.x[0] - self.trajectory.x_estimate[0]}
        self.trajectory.error_2norm = {0: np.linalg.norm(self.trajectory.error[0])}

    def architecture_limit_set(self, arch=None, min_set: int = None, max_set: int = None) -> None:
        # Sets specific values for lower and upper bounds on architecture
        for a in architecture_iterator(arch):
            if a == 'B':
                self.B.min = self.B.number_of_available if self.B.min is None else min_set if min_set is not None else self.B.min
                self.B.max = self.B.number_of_available if self.B.max is None else max_set if max_set is not None else self.B.max
            else:  # a == 'C'
                self.C.min = self.C.number_of_available if self.C.min is None else min_set if min_set is not None else self.C.min
                self.C.max = self.C.number_of_available if self.C.max is None else max_set if max_set is not None else self.C.max

    def architecture_limit_mod(self, arch=None, min_mod: int = None, max_mod: int = None) -> None:
        # Modifies lower and upper bounds on architecture
        min_mod = 0 if min_mod is None else min_mod
        max_mod = 0 if max_mod is None else max_mod
        for a in architecture_iterator(arch):
            if a == 'B':
                self.B.min = self.B.min + min_mod
                self.B.max = self.B.max + max_mod
            else:  # a == 'C'
                self.C.min = self.C.min + min_mod
                self.C.max = self.C.max + max_mod

    def architecture_update_to_matrix_from_active_set(self, arch=None) -> None:
        # Update architecture matrices and corresponding noise and cost matrices based on the active set
        # Used for recursion calculations of cost/error matrices and gains
        self.initialize_active_matrix(arch)
        for a in architecture_iterator(arch):
            if a == 'B':
                if len(self.B.active_set) > 0:
                    self.B.active_set = list(np.sort(self.B.active_set))
                    for k in range(0, len(self.B.active_set)):
                        self.B.active_matrix[:, k] = self.B.available_vectors[self.B.active_set[k]]
                    self.B.R1 = self.B.R1_reference[self.B.active_set, :][:, self.B.active_set]
            else:  # a == 'C'
                if len(self.C.active_set) > 0:
                    self.C.active_set = list(np.sort(self.C.active_set))
                    for k in range(0, len(self.C.active_set)):
                        self.C.active_matrix[k, :] = self.C.available_vectors[self.C.active_set[k]]
                    self.C.R1 = self.C.R1_reference[self.C.active_set, :][:, self.C.active_set]

    def architecture_update_to_indicator_from_active_set(self, arch=None) -> None:
        # Update architecture indicator vectors from active set
        # Used for architecture running and switching cost calculations
        for a in architecture_iterator(arch):
            if a == 'B':
                self.B.indicator_vector_history = np.zeros(self.B.number_of_available, dtype=int)
                self.B.indicator_vector_current = np.zeros(self.B.number_of_available, dtype=int)
                self.B.indicator_vector_current[self.B.history_active_set[self.sim.t_current]] = 1
                if self.sim.t_current >= 1:
                    self.B.indicator_vector_history[self.B.history_active_set[self.sim.t_current - 1]] = 1
                else:
                    self.B.indicator_vector_history = dc(self.B.indicator_vector_current)

            else:  # a == 'C'
                self.C.indicator_vector_history = np.zeros(self.C.number_of_available, dtype=int)
                self.C.indicator_vector_current = np.zeros(self.C.number_of_available, dtype=int)
                self.C.indicator_vector_current[self.C.history_active_set[self.sim.t_current]] = 1
                if self.sim.t_current >= 1:
                    self.C.indicator_vector_history[self.C.history_active_set[self.sim.t_current - 1]] = 1
                else:
                    self.C.indicator_vector_history = dc(self.C.indicator_vector_current)

    def architecture_update_to_history_from_active_set(self, arch=None) -> None:
        # Update record of architecture from active set
        for a in architecture_iterator(arch):
            if a == 'B':
                self.B.history_active_set[self.sim.t_current] = self.B.active_set
            else:  # a == 'C'
                self.C.history_active_set[self.sim.t_current] = self.C.active_set

    def architecture_update_to_history_indicator_matrix_from_active_set(self, arch=None) -> None:
        # Wrapper for all updates from active set
        self.architecture_update_to_history_from_active_set(arch)
        self.architecture_update_to_matrix_from_active_set(arch)
        self.architecture_update_to_indicator_from_active_set()

    def architecture_duplicate_active_set_from_system(self, reference_system, update_check=True) -> None:
        if not isinstance(reference_system, System):
            raise ClassError
        for a in ['B', 'C']:
            if a == 'B':
                self.B.active_set = dc(reference_system.B.active_set)
            else:
                self.C.active_set = dc(reference_system.C.active_set)
        if update_check:
            self.architecture_update_to_history_indicator_matrix_from_active_set()

    def architecture_compare_active_set_to_system(self, reference_system) -> bool:
        # Compare active architecture of two systems
        if not isinstance(reference_system, System):
            raise ClassError
        return set(self.B.active_set) == set(reference_system.B.active_set) and set(self.C.active_set) == set(
            reference_system.C.active_set)

    def architecture_limit_check(self) -> bool:
        # Check if architecture sets are within lower and upper bounds
        if self.B.min <= len(self.B.active_set) <= self.B.max and self.C.min <= len(self.C.active_set) <= self.C.max:
            return True
        else:
            return False

    def architecture_compute_active_set_changes(self, reference_system) -> int:
        # Compute the number of add/swap/drop changes between two architecture sets
        if not isinstance(reference_system, System):
            raise ClassError
        B_compare = compare_lists(self.B.active_set, reference_system.B.active_set)
        C_compare = compare_lists(self.C.active_set, reference_system.C.active_set)
        number_of_changes = max(len(B_compare['only1']), len(B_compare['only2'])) + max(len(C_compare['only1']),
                                                                                        len(C_compare['only2']))
        return number_of_changes

    def architecture_count_number_of_sim_changes(self) -> None:
        # Compare total number of active set changes over simulation time horizon
        self.B.change_count, self.C.change_count = 0, 0
        if self.sim.sim_model == "self_tuning":
            for t in range(1, self.sim.t_simulate):
                compare_B = compare_lists(self.B.history_active_set[t - 1], self.B.history_active_set[t])
                compare_C = compare_lists(self.C.history_active_set[t - 1], self.C.history_active_set[t])
                self.B.change_count += max(len(compare_B['only2']), len(compare_B['only1']))
                self.C.change_count += max(len(compare_C['only2']), len(compare_C['only1']))

    def architecture_display(self) -> None:
        # Print architecture if required
        print('B: {}'.format(self.B.active_set))
        print('C: {}'.format(self.C.active_set))

    def cost_architecture_running(self) -> None:
        # Compute architecture running costs based on given cost type as scalar or vector
        if self.trajectory.cost.metric_running == 0 or (self.B.R2 == 0 and self.C.R2 == 0):
            self.trajectory.cost.running = 0
        elif self.trajectory.cost.metric_running == 1 or (type(self.B.R2) == int and type(self.C.R2) == int):
            self.trajectory.cost.running = np.linalg.norm(self.B.indicator_vector_current,
                                                          ord=0) * self.B.R2 + np.linalg.norm(
                self.C.indicator_vector_current, ord=0) * self.C.R2
        elif self.trajectory.cost.metric_running == 2 or (
                np.shape(self.B.R2) == (len(self.B.active_set), len(self.B.active_set)) and np.shape(self.C.R2) == (
        len(self.C.active_set), len(self.C.active_set))):
            self.trajectory.cost.running = self.B.indicator_vector_current.T @ self.B.R2 @ self.B.indicator_vector_current + self.C.indicator_vector_current.T @ self.C.R2 @ self.C.indicator_vector_current
        else:
            print(self.B.R2)
            raise Exception('Check running cost metric')

    def cost_architecture_switching(self) -> None:
        # Compute architecture switching costs based on given cost type as scalar or vector
        if self.trajectory.cost.metric_switching == 0 or (self.B.R3 == 0 and self.C.R3 == 0):
            self.trajectory.cost.switching = 0
        elif self.trajectory.cost.metric_switching == 1 or (type(self.B.R3) == int and type(self.C.R3) == int):
            self.trajectory.cost.switching = np.linalg.norm(
                self.B.indicator_vector_current - self.B.indicator_vector_history, ord=0) * self.B.R3 + np.linalg.norm(
                self.C.indicator_vector_current - self.C.indicator_vector_history, ord=0) * self.C.R3
        elif self.trajectory.cost.metric_switching == 2 or (
                np.shape(self.B.R3) == (len(self.B.active_set), len(self.B.active_set)) and np.shape(self.C.R3) == (len(self.C.active_set), len(self.C.active_set))):
            self.trajectory.cost.switching = (self.B.indicator_vector_current - self.B.indicator_vector_history).T @ self.B.R2 @ (self.B.indicator_vector_current - self.B.indicator_vector_history) + (self.C.indicator_vector_current - self.C.indicator_vector_history).T @ self.C.R2 @ (self.C.indicator_vector_current - self.C.indicator_vector_history)
        else:
            raise Exception('Check switching cost metric')

    def prediction_gains(self, arch=None) -> None:
        # Compute only the gains with architecture changes - minimizes computation at each iteration
        for a in architecture_iterator(arch):
            if a == 'B':
                self.prediction_control_gain()
            else:  # if a == 'C'
                self.prediction_estimation_gain()

    def prediction_control_gain(self) -> None:
        # Control gains over the prediction time horizon
        self.B.recursion_matrix = {self.sim.t_predict: self.B.Q}
        for t in range(self.sim.t_predict - 1, -1, -1):
            self.B.gain[t] = np.linalg.inv((self.B.active_matrix.T @ self.B.recursion_matrix[
                t + 1] @ self.B.active_matrix) + self.B.R1) @ self.B.active_matrix.T @ self.B.recursion_matrix[
                                 t + 1] @ self.A.A_mat
            self.B.recursion_matrix[t] = (self.A.A_mat.T @ self.B.recursion_matrix[t + 1] @ self.A.A_mat) - (
                        self.A.A_mat.T @ self.B.recursion_matrix[t + 1] @ self.B.active_matrix @ self.B.gain[
                    t]) + self.B.Q

    def prediction_estimation_gain(self) -> None:
        # Estimation gains over the prediction time horizon
        self.C.recursion_matrix = {0: self.trajectory.estimation_matrix[self.sim.t_current]}
        for t in range(0, self.sim.t_predict):
            self.C.gain[t] = self.C.recursion_matrix[t] @ self.C.active_matrix.T @ np.linalg.inv(
                (self.C.active_matrix @ self.C.recursion_matrix[t] @ self.C.active_matrix.T) + self.C.R1)
            self.C.recursion_matrix[t + 1] = (self.A.A_mat @ self.C.recursion_matrix[t] @ self.A.A_mat.T) - (
                        self.A.A_mat @ self.C.gain[t] @ self.C.active_matrix @ self.C.recursion_matrix[
                    t] @ self.A.A_mat.T) + self.C.Q

    def cost_prediction(self) -> None:
        # Cost over the prediction horizon
        A_augmented_mat = {}
        W_augmented = {}
        F_augmented = {}
        Q_augmented = {}

        W_mat = np.block([[self.disturbance.W, np.zeros((self.number_of_states, len(self.C.active_set)))],
                          [np.zeros((len(self.C.active_set), self.number_of_states)),
                           self.disturbance.V[:, self.C.active_set][self.C.active_set, :]]])

        for t in range(0, self.sim.t_predict):
            BKt = self.B.active_matrix @ self.B.gain[t]
            ALtC = self.A.A_mat @ self.C.gain[t] @ self.C.active_matrix

            A_augmented_mat[t] = np.block([[self.A.A_mat, -BKt],
                                           [ALtC, self.A.A_mat - ALtC - BKt]])

            F_augmented[t] = np.block(
                [[np.identity(self.number_of_states), np.zeros((self.number_of_states, len(self.C.active_set)))],
                 [np.zeros((self.number_of_states, self.number_of_states)), self.A.A_mat @ self.C.gain[t]]])
            W_augmented[t] = F_augmented[t] @ W_mat @ F_augmented[t].T

            Q_augmented[t] = np.block([[self.B.Q, np.zeros((self.number_of_states, self.number_of_states))],
                                       [np.zeros((self.number_of_states, self.number_of_states)),
                                        self.B.gain[t].T @ self.B.R1 @ self.B.gain[t]]])

        Q_augmented[self.sim.t_predict] = np.block(
            [[self.B.Q, np.zeros((self.number_of_states, self.number_of_states))],
             [np.zeros((self.number_of_states, self.number_of_states)),
              np.zeros((self.number_of_states, self.number_of_states))]])

        self.A.A_augmented_mat = A_augmented_mat[0]
        self.disturbance.F_augmented = F_augmented[0]

        self.trajectory.cost.predicted_matrix = {self.sim.t_predict: Q_augmented[self.sim.t_predict]}
        self.trajectory.cost.control = 0

        for t in range(self.sim.t_predict - 1, -1, -1):
            self.trajectory.cost.control += np.trace(self.trajectory.cost.predicted_matrix[t + 1] @ W_augmented[t])
            self.trajectory.cost.predicted_matrix[t] = A_augmented_mat[t].T @ self.trajectory.cost.predicted_matrix[
                t + 1] @ A_augmented_mat[t]

        if self.trajectory.cost.metric_control == 1:
            x_estimate_stack = np.squeeze(np.tile(self.trajectory.x_estimate[self.sim.t_current], (1, 2)))
            self.trajectory.cost.control += (
                        x_estimate_stack.T @ self.trajectory.cost.predicted_matrix[0] @ x_estimate_stack)
        elif self.trajectory.cost.metric_control == 2:
            self.trajectory.cost.control += np.max(np.linalg.eigvals(self.trajectory.cost.predicted_matrix[0]))
        else:
            raise Exception('Check control cost metric')

    def cost_true(self) -> None:
        # True cost incurred at current timestep
        Q_mat = np.block([[self.B.Q, np.zeros((self.number_of_states, self.number_of_states))],
                          [np.zeros((self.number_of_states, self.number_of_states)),
                           self.B.gain[0].T @ self.B.R1 @ self.B.gain[0]]])
        self.trajectory.cost.control = 0
        if self.trajectory.cost.metric_control == 1:
            x_estimate_stack = np.squeeze(np.tile(self.trajectory.x_estimate[self.sim.t_current], (1, 2)))
            self.trajectory.cost.control += (x_estimate_stack.T @ Q_mat @ x_estimate_stack)
        elif self.trajectory.cost.metric_control == 2:
            self.trajectory.cost.control += np.max(np.linalg.eigvals(self.trajectory.cost.predicted_matrix[0]))
        else:
            raise Exception('Check control cost metric')

    def cost_prediction_wrapper(self, evaluate_gains='skip') -> None:
        # Wrapper to update only the required gains and compute predicted costs
        self.prediction_gains(evaluate_gains)
        self.cost_prediction()
        self.cost_architecture_running()
        self.cost_architecture_switching()
        self.trajectory.cost.predicted[
            self.sim.t_current] = self.trajectory.cost.control + self.trajectory.cost.running + self.trajectory.cost.switching

    def cost_true_wrapper(self) -> None:
        # Wrapper to update only the required gains and compute true costs
        self.cost_true()
        self.cost_architecture_running()
        self.cost_architecture_switching()
        self.trajectory.cost.true[self.sim.t_current] = self.trajectory.cost.control + self.trajectory.cost.running

    # choice format: {target_architecture, target_node, +/- (select/reject), resultant_system}
    def available_choices_selection(self) -> list:
        if len(self.B.active_set) >= self.B.min and len(self.C.active_set) >= self.C.min:
            # If minimum number of actuators AND sensors are active
            choices = [{'arch': 'skip', 'idx': None, 'change': None}]
        else:
            choices = []

        # High selection priority of actuators or sensors if < minimum
        if len(self.B.active_set) < self.B.min or len(self.C.active_set) < self.C.min:
            if len(self.B.active_set) < self.B.min:
                choices.extend([{'arch': 'B', 'idx': i, 'change': '+'} for i in
                                compare_lists(self.B.active_set, self.B.available_indices)['only2']])
            if len(self.C.active_set) < self.C.min:
                choices.extend([{'arch': 'C', 'idx': i, 'change': '+'} for i in
                                compare_lists(self.C.active_set, self.C.available_indices)['only2']])
        else:  # Low selection priority of actuators or sensors if >= min and < maximum
            if len(self.B.active_set) < self.B.max:
                choices.extend([{'arch': 'B', 'idx': i, 'change': '+'} for i in
                                compare_lists(self.B.active_set, self.B.available_indices)['only2']])
            if len(self.C.active_set) < self.C.max:
                choices.extend([{'arch': 'C', 'idx': i, 'change': '+'} for i in
                                compare_lists(self.C.active_set, self.C.available_indices)['only2']])

        return choices

    def available_choices_rejection(self) -> list:
        if len(self.B.active_set) <= self.B.max and len(self.C.active_set) <= self.C.max:
            # If maximum number of actuators AND sensors are active
            choices = [{'arch': 'skip', 'idx': None, 'change': None}]
        else:
            choices = []

        # High rejection priority of actuators or sensors if >= maximum
        if len(self.B.active_set) > self.B.max or len(self.C.active_set) > self.C.max:
            if len(self.B.active_set) > self.B.max:
                choices.extend([{'arch': 'B', 'idx': i, 'change': '-'} for i in self.B.active_set])
            if len(self.C.active_set) > self.C.max:
                choices.extend([{'arch': 'C', 'idx': i, 'change': '-'} for i in self.C.active_set])
        else:  # Low rejection priority of actuators or sensors if > minimum and <= maximum
            if len(self.B.active_set) > self.B.min:
                choices.extend([{'arch': 'B', 'idx': i, 'change': '-'} for i in self.B.active_set])
            if len(self.C.active_set) > self.C.min:
                choices.extend([{'arch': 'C', 'idx': i, 'change': '-'} for i in self.C.active_set])

        return choices

    def architecture_update_active_set_from_choices(self, architecture_change_parameters) -> None:
        # Update active set for a specific choice
        if architecture_change_parameters['arch'] == 'B':
            if architecture_change_parameters['change'] == '+':
                self.B.active_set.append(architecture_change_parameters['idx'])
                self.B.active_set = list(np.sort(np.array(self.B.active_set)))
            elif architecture_change_parameters['change'] == '-':
                self.B.active_set = [k for k in self.B.active_set if k != architecture_change_parameters['idx']]
        elif architecture_change_parameters['arch'] == 'C':
            if architecture_change_parameters['change'] == '+':
                self.C.active_set.append(architecture_change_parameters['idx'])
                self.C.active_set = list(np.sort(np.array(self.C.active_set)))
            elif architecture_change_parameters['change'] == '-':
                self.C.active_set = [k for k in self.C.active_set if k != architecture_change_parameters['idx']]
        elif architecture_change_parameters == {'arch': 'skip', 'idx': None, 'change': None}:
            pass
        else:
            raise Exception('Invalid update choice')

    def evaluate_cost_for_choice(self, choice_parameter) -> dict:
        # Wrapper to update active set and evaluate costs
        S_choice = dc(self)
        S_choice.architecture_update_active_set_from_choices(choice_parameter)
        S_choice.architecture_update_to_history_indicator_matrix_from_active_set()
        S_choice.cost_prediction_wrapper(choice_parameter['arch'])
        return_values = dc(choice_parameter)
        return_values.update({'system': S_choice, 'cost': S_choice.trajectory.cost.predicted[S_choice.sim.t_current]})
        return return_values

    def one_step_system_update(self) -> None:
        # Update state and architecture trajectories and costs based

        self.trajectory.X_augmented[self.sim.t_current + 1] = (self.A.A_augmented_mat @ self.trajectory.X_augmented[
            self.sim.t_current]) + (self.disturbance.F_augmented @ np.concatenate((self.disturbance.w_gen[
                                                                                       self.sim.t_current],
                                                                                   self.disturbance.v_gen[
                                                                                       self.sim.t_current][
                                                                                       self.C.active_set])))

        self.trajectory.x[self.sim.t_current + 1] = self.trajectory.X_augmented[self.sim.t_current + 1][
                                                    0:self.number_of_states]

        self.trajectory.x_estimate[self.sim.t_current + 1] = self.trajectory.X_augmented[self.sim.t_current + 1][
                                                             self.number_of_states:]

        self.trajectory.error[self.sim.t_current + 1] = self.trajectory.x[self.sim.t_current + 1] - \
                                                        self.trajectory.x_estimate[self.sim.t_current + 1]

        self.trajectory.error_2norm[self.sim.t_current + 1] = np.linalg.norm(
            self.trajectory.error[self.sim.t_current + 1])

        self.trajectory.estimation_matrix[self.sim.t_current + 1] = self.C.recursion_matrix[1]
        self.trajectory.control_cost_matrix[self.sim.t_current] = self.B.recursion_matrix[0]

        self.sim.t_current += 1
        self.architecture_update_to_history_indicator_matrix_from_active_set()
        self.prediction_gains()
        self.cost_prediction_wrapper()

    def generate_architecture_history_points(self, arch=None) -> None:
        # Scatter plot (x,y) for architecture history
        for a in architecture_iterator(arch):
            if a == 'B':
                for t in self.B.history_active_set:
                    for node in self.B.history_active_set[t]:
                        self.plot.B_history[0].append(t)
                        self.plot.B_history[1].append(node + 1)
            else:  # a == 'C'
                for t in self.C.history_active_set:
                    for node in self.C.history_active_set[t]:
                        self.plot.C_history[0].append(t)
                        self.plot.C_history[1].append(node + 1)

    def architecture_active_count(self, arch=None) -> None:
        # populate list of size of active architecture at each time step
        for a in architecture_iterator(arch):
            if a == 'B':
                if self.sim.sim_model == 'self_tuning':
                    self.B.active_count = [len(self.B.history_active_set[i]) for i in
                                           range(0, len(self.B.history_active_set))]
                elif self.sim.sim_model == "fixed":
                    self.B.active_count = [len(self.B.active_set)] * len(self.B.history_active_set)
                else:
                    raise Exception('Invalid test model')

            else:  # if arch == 'C'
                if self.sim.sim_model == 'self_tuning':
                    self.C.active_count = [len(self.C.history_active_set[i]) for i in
                                           range(0, len(self.C.history_active_set))]
                elif self.sim.sim_model == "fixed":
                    self.C.active_count = [len(self.C.active_set)] * len(self.C.history_active_set)
                else:
                    raise Exception('Invalid test model')

    def plot_architecture_history(self, arch=None, ax_in=None) -> None:
        if ax_in is None:
            fig = plt.figure()
            grid = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(grid[0, 0])
        else:
            ax = ax_in

        x_val, y_val = [], []
        for a in architecture_iterator(arch):
            if a == 'B':
                self.generate_architecture_history_points(arch=a)
                x_val = self.plot.B_history[0]
                y_val = self.plot.B_history[1]
            else:  # a=='C'
                self.generate_architecture_history_points(arch=a)
                x_val = self.plot.C_history[0]
                y_val = self.plot.C_history[1]
        ax.scatter(x_val, y_val, s=10, alpha=0.7,
                   marker=self.plot.plot_parameters[self.plot.plot_system]['m'],
                   c=self.plot.plot_parameters[self.plot.plot_system]['c'])
        ax.set_ylim([-1, self.number_of_states + 2])
        ax.set_yticks([1, self.number_of_states])
        ax.grid(visible=True, which='major', axis='x')

        if ax_in is None:
            plt.show()

    def plot_compute_time(self, ax_in=None) -> None:
        if ax_in is None:
            fig = plt.figure()
            grid = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(grid[0, 0])
        else:
            ax = ax_in

        compute_time = self.list_from_dict_key_time(self.trajectory.computation_time)
        ax.scatter(range(0, len(compute_time)), compute_time,
                   color=self.plot.plot_parameters[self.plot.plot_system]['c'], marker='o', s=5)
        ax.grid(visible=True, which='major', axis='x')

        if ax_in is None:
            plt.show()

    def plot_architecture_count(self, ax_in=None, arch=None) -> None:
        if ax_in is None:
            fig = plt.figure()
            grid = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(grid[0, 0])
        else:
            ax = ax_in

        x_val, y_val = [], []
        for a in architecture_iterator(arch):
            self.architecture_active_count(arch=a)
            if a == 'B':
                x_val, y_val = range(0, len(self.B.active_count)), self.B.active_count
                size_arch = [self.B.min, self.B.max]
            else:  # a == 'C'
                x_val, y_val = range(0, len(self.C.active_count)), self.C.active_count
                size_arch = [self.C.min, self.C.max]

            ax.plot(x_val, y_val, color=self.plot.plot_parameters[self.plot.plot_system]['c'], alpha=0.7)

            ax.set_ylim(size_arch[0] - 0.5, size_arch[1] + 0.5)
            ax.set_yticks(size_arch)
        ax.grid(visible=True, which='major', axis='x')

        if ax_in is None:
            plt.show()

    def plot_cost(self, cost_type=None, ax_in=None, set_details_flag=False) -> None:
        if ax_in is None:
            fig = plt.figure()
            grid = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(grid[0, 0])
        else:
            ax = ax_in

        if cost_type is None:
            cost_type = ['true', 'predict']

        cost, ls, legend_handler = [], '', []

        for t in cost_type:
            if t == 'true':
                cost = list(itertools.accumulate(self.list_from_dict_key_time(self.trajectory.cost.true)))
                ls, labeler = 'solid', 'Cumulate Cost'
            elif t == 'predict':
                cost = self.list_from_dict_key_time(self.trajectory.cost.predicted)
                ls, labeler = 'dashed', 'Predict'
            else:
                raise Exception('Check argument')

            ax.plot(range(0, self.sim.t_simulate), cost, c=self.plot.plot_parameters[self.plot.plot_system]['c'],
                    linestyle=ls, alpha=0.7)
            legend_handler.append(mlines.Line2D([], [], color='black', ls=ls, label=labeler))

        ax.set_yscale('log')
        ax.grid(visible=True, which='major', axis='x')
        if set_details_flag:
            ax.set_ylabel('Cost\n' + r'$J_t$')
            leg1 = ax.legend(handles=legend_handler, loc='upper left', ncol=1)
            ax.add_artist(leg1)

        if ax_in is None:
            ax.set_xlabel(r'Time $t$')
            ax.set_ylabel(r'Cost')
            plt.show()

    def plot_openloop_eigvals(self, ax_in=None) -> None:
        if ax_in is None:
            fig = plt.figure()
            grid = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(grid[0, 0])
        else:
            ax = ax_in

        ax.scatter(range(1, self.number_of_states + 1), np.sort(np.abs(self.A.open_loop_eig_vals)),
                   marker='x', s=10, c='black', alpha=0.7)
        ax.axhline(y=1, color='tab:gray', ls='dashdot', alpha=0.5)
        # ax.set_ylim(np.min(np.abs(self.A.open_loop_eig_vals)), np.max(np.abs(self.A.open_loop_eig_vals)))
        ax.set_ylabel(r'$|\lambda_i(A)|$')
        ax.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False)

        if ax_in is None:
            ax.set_xlabel('Mode')
            plt.show()

    def plot_states(self, ax_in=None, state_marker=None, set_details_flag=False) -> None:
        if ax_in is None:
            fig = plt.figure()
            grid = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(grid[0, 0])
        else:
            ax = ax_in

        if state_marker is None:
            state_marker = ['state', 'estimate', 'error']

        legend_handler = []

        for state in state_marker:
            # x, ls = [], ''
            if state == 'state':
                x = self.ndarray_from_dict_key_states(self.trajectory.x)
                ls = 'solid'
                labeler = r'$|x(t)|_2$'
            elif state == 'estimate':
                x = self.ndarray_from_dict_key_states(self.trajectory.x_estimate)
                ls = 'dashdot'
                labeler = r'$|\hat{x}(t)|_2$'
            elif state == 'error':
                x = self.ndarray_from_dict_key_states(self.trajectory.error)
                ls = 'dashed'
                labeler = r'$|x(t) - \hat{x}(t)|_2$'
            else:
                raise Exception('Check iterator')

            x_norm = [np.linalg.norm(x[:, i]) for i in range(0, self.sim.t_simulate)]
            ax.plot(range(0, self.sim.t_simulate), x_norm, color=self.plot.plot_parameters[self.plot.plot_system]['c'],
                    ls=ls, alpha=0.8)
            legend_handler.append(mlines.Line2D([], [], color='black', ls=ls, label=labeler))

        if set_details_flag:
            ax.set_ylabel('States')
            ax.set_yscale('log')
            ax.legend(handles=legend_handler, loc='upper left', ncol=1)
            ax.grid(visible=True, which='major', axis='x')

        if ax_in is None:
            if set_details_flag:
                ax.set_xlabel('Time')
            plt.show()

    def list_from_dict_key_time(self, v: dict) -> list:
        # Simulation-time indexed sorted list from dictionary
        ret_list = []
        for t in range(0, self.sim.t_simulate):
            ret_list.append(v[t])
        return ret_list

    def ndarray_from_dict_key_states(self, v: dict) -> np.ndarray:
        # Simulation-time index updated numpy array from dictionary
        ret_list = np.empty((self.number_of_states, self.sim.t_simulate))
        for t in range(0, self.sim.t_simulate):
            ret_list[:, t] = v[t]
        return ret_list

    def scalecost_by_test_parameter(self) -> None:
        # Check for non-zero base costs for valuable scaling
        cost_scale = 0 if self.sim.test_parameter is None else self.sim.test_parameter
        self.B.R2 = self.B.R2 * cost_scale
        self.B.R3 = self.B.R3 * cost_scale
        self.C.R2 = self.C.R2 * cost_scale
        self.C.R3 = self.C.R3 * cost_scale
        if cost_scale == 0:
            self.plot_name = 'self_tuning free arch'
        else:
            self.plot_name = 'self_tuning ' + str(cost_scale) + 'scale arch cost'

    def optimize_initial_architecture(self, print_check: bool = False):
        # Optimize architecture using simultaneous greedy architecture assuming no switching costs - free offline optimization
        if print_check:
            print('Optimizing design-time architecture from:')
            self.architecture_display()

        t_predict_ref = dc(self.sim.t_predict)
        self.sim.t_predict = 2 * t_predict_ref
        self.trajectory.cost.metric_control = 2
        BR3, CR3 = dc(self.B.R3), dc(self.C.R3)
        self.B.R3 *= 0
        self.C.R3 *= 0
        self.prediction_gains()
        self.cost_prediction_wrapper()

        self.greedy_simultaneous(print_check_inner=print_check, print_check_outer=print_check)

        self.sim.t_predict = dc(t_predict_ref)
        self.trajectory.cost.metric_control = 1
        self.B.R3 = dc(BR3)
        self.C.R3 = dc(CR3)
        self.prediction_gains()
        self.cost_prediction_wrapper()

        if print_check:
            print('Design-time architecture optimized to:')
            self.architecture_display()

    def simulate(self, print_check: bool = False, tqdm_check: bool = False):
        # Simulation wrapper for both fixed and self-tuning architecture based on sim_model
        if self.sim.sim_model == "fixed":
            self.simulate_fixed_architecture(print_check=print_check, tqdm_check=tqdm_check)
        elif self.sim.sim_model == "self_tuning":
            self.simulate_self_tuning_architecture(print_check=print_check, tqdm_check=tqdm_check)
        else:
            raise Exception('Invalid sim model')

    def simulate_fixed_architecture(self, print_check: bool = False, tqdm_check: bool = True):
        # Fixed architecture - Only update trajectories over time
        self.model_namer()

        if print_check:
            print('Simulating Fixed Architecture')

        if tqdm_check:
            with tqdm(total=self.sim.t_simulate, ncols=100, desc='Fixed (P_ID:' + str(os.getpid()) + ')',
                      leave=False) as pbar:
                for _ in range(0, self.sim.t_simulate):
                    t_start = process_time()
                    self.cost_true_wrapper()
                    self.trajectory.computation_time[self.sim.t_current] = process_time() - t_start
                    self.one_step_system_update()
                    pbar.update()
        else:
            for _ in range(0, self.sim.t_simulate):
                t_start = process_time()
                self.cost_true_wrapper()
                self.trajectory.computation_time[self.sim.t_current] = process_time() - t_start
                self.one_step_system_update()

        if print_check:
            print('Fixed Architecture Simulation: DONE')

    def simulate_self_tuning_architecture(self, print_check: bool = False, tqdm_check: bool = True):
        # Self-tuning architecture - optimize architecture and update trajectory at each time-step
        self.model_namer()

        if print_check:
            print('Simulating Self-Tuning Architecture')

        if tqdm_check:
            with tqdm(total=self.sim.t_simulate, ncols=100, desc='Self-Tuning (P_ID:' + str(os.getpid()) + ')',
                      leave=False) as pbar:
                for t in range(0, self.sim.t_simulate):
                    t_start = process_time()
                    if t > 0:
                        self.greedy_simultaneous(print_check_outer=print_check)
                    self.cost_true_wrapper()
                    self.trajectory.computation_time[self.sim.t_current] = process_time() - t_start
                    self.one_step_system_update()
                    pbar.update()
        else:
            for t in range(0, self.sim.t_simulate):
                t_start = process_time()
                if t > 0:
                    self.greedy_simultaneous(print_check_outer=print_check)
                self.cost_true_wrapper()
                self.trajectory.computation_time[self.sim.t_current] = process_time() - t_start
                self.one_step_system_update()

        if print_check:
            print('Self-Tuning Architecture Simulation: DONE')

    def greedy_selection(self, print_check: bool = False):
        # Uses priority system to determine choices
        # Limits number of changes per iteration and maximum number of selections
        # Implements early exit
        exit_condition = 0
        work_sys = dc(self)
        arch_ref = dc(self)
        cost_improvement = [work_sys.trajectory.cost.predicted[work_sys.sim.t_current]]

        if print_check:
            print('Initial architecture')
            work_sys.architecture_display()

        number_of_changes, number_of_choices, selection_check = 0, 0, True
        while selection_check:
            work_iteration = dc(work_sys)
            choices = work_iteration.available_choices_selection()
            if print_check:
                print('Iteration: {}'.format(number_of_changes))
                work_iteration.architecture_display()

            if len(choices) == 0:
                # Exit if there are no available selections or the only option is no change
                selection_check = False
                exit_condition = 1
                if print_check:
                    print('Selection exit 1: No available selections')

            elif len(choices) == 1 and choices[0]['arch'] == 'skip':
                # Exit if the only option is no change
                selection_check = False
                exit_condition = 2
                if print_check:
                    print('Selection exit 2: No valuable selections')

            else:
                number_of_choices += len(choices)
                # evaluations = list(map(work_iteration.evaluate_cost_for_choice, choices))
                evaluations = cost_mapper(work_iteration, choices)
                # smallest_cost = min(d['cost'] for d in evaluations)
                # index_of_smallest_cost = evaluations.index(min(d for d in evaluations if d['cost'] == smallest_cost))

                index_of_smallest_cost = min(range(len(evaluations)),
                                             key=lambda change: evaluations[change].get('cost', float('inf')))
                smallest_cost = evaluations[index_of_smallest_cost]['cost']

                if print_check:
                    print('Choice evaluations')
                    for i in range(0, len(evaluations)):
                        print('{}: {} | {} | {} | {}'.format(i, evaluations[i]['arch'], evaluations[i]['idx'],
                                                             evaluations[i]['change'], evaluations[i]['cost']))
                    print('Smallest cost: {} @ {}'.format(smallest_cost, index_of_smallest_cost))
                    evaluations[index_of_smallest_cost]['system'].architecture_display()

                if evaluations[index_of_smallest_cost]['arch'] == 'skip':
                    # Exit if the best option is no change
                    selection_check = False
                    exit_condition = 5
                    if print_check:
                        print('Selection exit 5: No valuable selections')

                else:
                    work_sys = dc(evaluations[index_of_smallest_cost]['system'])
                    cost_improvement.append(smallest_cost)
                    number_of_changes = arch_ref.architecture_compute_active_set_changes(work_sys)
                    # number_of_changes += 1

                    if self.sim.self_tuning_parameter is not None and number_of_changes >= self.sim.self_tuning_parameter:
                        # Exit if maximum number of changes have been completed
                        selection_check = False
                        exit_condition = 3
                        if print_check:
                            print('Selection exit 3: Maximum selections done')

                    elif number_of_changes == 0:
                        # Trigger 2 should always activate before this
                        selection_check = False
                        exit_condition = 4
                        if print_check:
                            print('Selection exit 4: No selections done')

        # work_sys.trajectory.computation_time[work_sys.sim.t_current] = time.time() - t_start
        if print_check:
            print('Number of iterations: {}'.format(number_of_changes))
            work_sys.architecture_display()
            # print('Computation time: {}\nCost Improvement: {}'.format(work_sys.trajectory.computation_time[work_sys.sim.t_current], cost_improvement))

        if not work_sys.architecture_limit_check():
            print('\nB: {} | C: {} | max: {} | min: {} | Exit condition: {}'.format(len(work_sys.B.active_set),
                                                                                    len(work_sys.C.active_set),
                                                                                    work_sys.B.max, work_sys.B.min,
                                                                                    exit_condition))
            raise Exception('Architecture limits failed')

        self.copy_from_system(work_sys)

    def greedy_rejection(self, print_check: bool = False):
        # Uses priority system to determine choices
        # Limits number of changes per iteration and maximum number of rejections
        # Implements early exit
        exit_condition = 0
        work_sys = dc(self)
        cost_improvement = [work_sys.trajectory.cost.predicted[work_sys.sim.t_current]]

        if print_check:
            print('Initial architecture')
            work_sys.architecture_display()

        number_of_changes, number_of_choices, rejection_check = 0, 0, True
        while rejection_check:
            work_iteration = dc(work_sys)
            choices = work_iteration.available_choices_rejection()
            if print_check:
                print('Iteration: {}'.format(number_of_changes))
                work_iteration.architecture_display()

            if len(choices) == 0:
                # Exit if there are no available rejections or the only option is no change
                rejection_check = False
                exit_condition = 1
                if print_check:
                    print('Rejection exit 1: No available rejections')

            elif len(choices) == 1 and choices[0]['arch'] == 'skip':
                # Exit if the only option is no change
                rejection_check = False
                exit_condition = 2
                if print_check:
                    print('Rejection exit 2: No valuable rejections')

            else:
                number_of_choices += len(choices)
                evaluations = cost_mapper(work_iteration, choices)

                index_of_smallest_cost = min(range(len(evaluations)),
                                             key=lambda change: evaluations[change].get('cost', float('inf')))
                smallest_cost = evaluations[index_of_smallest_cost]['cost']

                if print_check:
                    print('Choice evaluations')
                    for i in range(0, len(evaluations)):
                        print('{}: {} | {} | {} | {}'.format(i, evaluations[i]['arch'], evaluations[i]['idx'],
                                                             evaluations[i]['change'], evaluations[i]['cost']))
                    print('Smallest cost: {} @ {}'.format(smallest_cost, index_of_smallest_cost))
                    evaluations[index_of_smallest_cost]['system'].architecture_display()

                if evaluations[index_of_smallest_cost]['arch'] == 'skip':
                    # Exit if the best option is no change
                    rejection_check = False
                    exit_condition = 5
                    if print_check:
                        print('Rejection exit 5: No valuable rejections')

                else:
                    work_sys = dc(evaluations[index_of_smallest_cost]['system'])
                    cost_improvement.append(smallest_cost)
                    number_of_changes = self.architecture_compute_active_set_changes(work_sys)

                    if self.sim.self_tuning_parameter is not None and number_of_changes >= self.sim.self_tuning_parameter:
                        # Exit if maximum number of changes have been completed
                        rejection_check = False
                        exit_condition = 3
                        if print_check:
                            print('Rejection exit 3: Maximum rejections done')

                    elif number_of_changes == 0:
                        # Trigger 2 should always activate before this
                        rejection_check = False
                        exit_condition = 4
                        if print_check:
                            print('Rejection exit 4: No valuable rejections')

        # work_sys.trajectory.computation_time[work_sys.sim.t_current] = time.time() - t_start
        if print_check:
            print('Number of iterations: {}'.format(number_of_changes))
            work_sys.architecture_display()
            # print('Computation time: {}\nCost Improvement: {}'.format(work_sys.trajectory.computation_time[work_sys.sim.t_current], cost_improvement))

        if not work_sys.architecture_limit_check():
            print('\nB: {} | C: {} | max: {} | min: {} | Exit condition: {}'.format(len(work_sys.B.active_set),
                                                                                    len(work_sys.C.active_set),
                                                                                    work_sys.B.max, work_sys.B.min,
                                                                                    exit_condition))
            raise Exception('Architecture limits failed')

        self.copy_from_system(work_sys)

    def greedy_simultaneous(self, number_of_changes_per_iteration: int = None, print_check_outer: bool = False, print_check_inner: bool = False):
        # Implements sequential greedy selection rejection
        # Uses early exit conditions if optimal solution is reached
        # Otherwise requires limit on maximum number of changes to architecture
        # Implements a safety check if more than number_of_states iterations of changes
        work_sys = dc(self)
        cost_improvement = [work_sys.trajectory.cost.predicted[work_sys.sim.t_current]]
        swap_limit_mod = 1 if number_of_changes_per_iteration is None else number_of_changes_per_iteration
        work_sys.sim.self_tuning_parameter = 2 * swap_limit_mod
        safety_counter = 0
        exit_condition = 0

        if print_check_outer:
            print('Initial architecture')
            work_sys.architecture_display()

        number_of_changes, number_of_choices, simultaneous_check = 0, 0, True
        while simultaneous_check:
            force_swap = dc(work_sys)
            # force_swap.sim.self_tuning_parameter = 2 * swap_limit_mod
            force_swap.architecture_limit_mod(min_mod=swap_limit_mod, max_mod=swap_limit_mod)
            force_swap.greedy_selection(print_check=print_check_inner)

            if print_check_outer:
                print('After force selection')
                force_swap.architecture_display()

            force_swap.architecture_limit_mod(min_mod=-swap_limit_mod, max_mod=-swap_limit_mod)
            force_swap.greedy_rejection(print_check=print_check_inner)

            cost_improvement.append(force_swap.trajectory.cost.predicted[force_swap.sim.t_current])

            if print_check_outer:
                print('After force rejection')
                force_swap.architecture_display()

            if work_sys.architecture_compare_active_set_to_system(
                    force_swap):  # Comparison of previous to current iteration of architecture update
                simultaneous_check = False
                exit_condition = 1
                if print_check_outer:
                    print('Swap exit 1: No more valuable forced swaps')

            else:
                number_of_changes = self.architecture_compute_active_set_changes(
                    force_swap)  # Comparison of initial reference to current architecture
                work_sys = dc(force_swap)
                safety_counter += 1

                if self.sim.self_tuning_parameter is not None and number_of_changes >= 2 * self.sim.self_tuning_parameter:
                    simultaneous_check = False
                    exit_condition = 2
                    if print_check_outer:
                        print('Swap exit 2: Maximum swaps done')
                elif number_of_changes == 0:
                    simultaneous_check = False
                    exit_condition = 3
                    if print_check_outer:
                        print('Swap exit 3: No valuable swaps or cyclic swaps')

                if safety_counter > work_sys.number_of_states:
                    raise Exception('Triggered safety constraint - too many changes - check system')

        # work_sys.trajectory.computation_time[work_sys.sim.t_current] = time.time() - t_start
        if print_check_outer:
            work_sys.architecture_display()
            # print('Computation time: {}\nCost Improvement: {}'.format(work_sys.trajectory.computation_time[work_sys.sim.t_current], cost_improvement))

        if not work_sys.architecture_limit_check():
            print('\nB: {} | C: {} | max: {} | min: {} | Exit condition: {}'.format(len(work_sys.B.active_set),
                                                                                    len(work_sys.C.active_set),
                                                                                    work_sys.B.max, work_sys.B.min,
                                                                                    exit_condition))
            raise Exception('Architecture limits failed')

        self.copy_from_system(work_sys)


def cost_mapper(S: System, choices) -> list:
    # Mapper for function to evaluate cost for each choice
    evaluation = list(map(S.evaluate_cost_for_choice, choices))
    return evaluation


def element_wise_min_max(v_ref_min, v_ref_max, v):
    v_ret_min = [min(e) for e in zip(v_ref_min, v)]
    v_ret_max = [max(e) for e in zip(v_ref_max, v)]

    return v_ret_min, v_ret_max


def statistics_data_parser(S: System, cost_min, cost_max, compute_time, arch_change, arch_count):
    # Data parser function for pulling plotting info from simulated statistics models
    cost_min, cost_max = element_wise_min_max(cost_min, cost_max, list(itertools.accumulate(S.list_from_dict_key_time(S.trajectory.cost.true))))
    compute_time.append(np.average(S.list_from_dict_key_time(S.trajectory.computation_time)))
    S.architecture_count_number_of_sim_changes()
    S.architecture_active_count()

    arch_change['B'].append(float(S.B.change_count / S.sim.t_simulate))
    arch_change['C'].append(float(S.C.change_count / S.sim.t_simulate))

    arch_count['B'].append(np.average(S.B.active_count))
    arch_count['C'].append(np.average(S.C.active_count))

    return cost_min, cost_max, compute_time, arch_change, arch_count


def run_experiment(exp_no=None, run_check: bool = True, plot_check: bool = True):
    # Primary caller to simulate/plot an experiment
    Exp = Experiment()
    if run_check:
        Exp.simulate_experiment_wrapper(exp_no=exp_no)

    try:
        if plot_check:
            Exp.plot_experiment(exp_no=exp_no)
    except dbm.error:
        print('No code run data for plotting')


if __name__ == "__main__":
    print('Function file run check')
