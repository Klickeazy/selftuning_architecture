# Speedy Greedy Algorithm

Simultaneous greedy selection and rejection algorithm on multi-metric costs for run-time (online) architecture selection in networks

## Architecture Selection

For a given system $$x_{t+1} = Ax_t + B_S u_t$$





## Greedy Algorithm for Architecture Selection

Exhaustive search for combinatorial optimization of solutions is not computationally tractable for large scale problems. The Greedy algorithm sacrifices guarantees of global optimality (of combinatorial problems) for significant improvements in computation power requirements to find a locally optimal solution as below:
- Greedy selection: Initialized with nothing and iteratively select the most optimal choice at every iteration
- Greedy rejection: Initialized using all available choices and iteratively drop the least optimal choice at every iteration





### Conda Environment - Python 3.10.8 packages
Updated on: 4th Jan 2023

Packages
- Numpy
- Scipy
- Networkx
- Matplotlib


##### Option 1 - build from [Py3_10.yml](PyEnvironment\Py3_10.yml)
Navigate terminal to folder and create environment using:    

    conda env create -f Py3_10.yml

You can export an environment using:

    conda env export > Py3_10.yml

Note: _Py3_10_ is the name of the environment. You may choose to rename this.


##### Option 2 - build from Anaconda backup package [Py3_10_condabackup.yaml](PyEnvironment/Py3_10_condabackup.yaml)

Use _Import_ in Anaconda Navigator > Environment and navigate to file