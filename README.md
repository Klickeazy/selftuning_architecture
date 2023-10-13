# Architecture Selection for Self-Tuning Optimal Control of Networks

Supporting code for "Architecture Selection for Self-Tuning Optimal Control of Networks" with code for greedy-swapping and extended numerical analysis on the algorithm's optimization parameters.

Paper authors: Karthik Ganapathy, Iman Shames, Mathias Hudoba de Badyn, Tyler Summers

Code author: Karthik Ganapathy

Check latest release for working code

## Python package requirements
The code is created and tested on Python 3.11.3.

Additional package requirements are provided in the [requirements.txt](requirements.txt) file

## Run code
The base execution code is provided in [run_code.py](run_code.py) for *exp_no=[1,9]* corresponding to the examples included in the main paper.

The Jupyter notebook [numerical_analysis.ipynb](numerical_analysis.ipynb) contains more extensive numerical analysis and explanations for the experiment parameters and test cases.

## DataDump
Pre-generated images for all provided experiments are available in the Images folder. Simulation data for all provided experiments are available in: [selftuning_architecture_datadump](https://github.com/Klickeazy/selftuning_architecture_datadump).

This is recommended for the statistical experiments which have long execution times.

After cloning this repository, copy the contents of the DataDump folder from the linked repository to the DataDump folder of this repository to run the visualizer code without needing to run the simulations.
