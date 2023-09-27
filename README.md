# Architecture Selection for Self-Tuning Optimal Control of Networks



Simultaneous greedy selection and rejection algorithm on multi-metric costs for run-time (online) architecture selection in networks

## Architecture Selection

Consider a graph $\mathcal{G}=(\mathcal{V},\mathcal{E})$, with $\mathcal{V}$ nodes connected by $\mathcal{E}$ edges. The dynamics of the network system are
$$x_{t+1} = Ax_t + B_{S_t} u_t + w_t$$
$$y_t = C_{S^{'}_t} x_t + v_t$$
where at time $t$:
- $x_t$ is the vector of states
- $y_t$ is the vector of measurements
- $u_t$ is the vector of inputs
- $w_t, v_t$ are i.i.d. zero-mean process and measurment noises
- $A$ is the dynamics derived from the weighted adjacency matrix of network $\mathcal{G}$
- $B_S$ is the control input matrix formed by columns of actuators in $S_t$. For all available actuators $\mathcal{B}$, $S_t \subseteq \mathcal{B}$ is the active set of actuators at the current time step
- $C_{S'_ t}$ is the measurement matrix formed by rows of sensors in $S'_ t$. For all available sensors $\mathcal{C}$, $S'_ t \subseteq \mathcal{C}$ is the active set of sensors at the current time step




## Greedy Algorithm for Architecture Selection

Exhaustive search for combinatorial optimization of solutions is not computationally tractable for large scale problems. The Greedy algorithm sacrifices guarantees of global optimality (of combinatorial problems) for significant improvements in computation power requirements to find a locally optimal solution as below:
- Greedy selection: Initialized with nothing and iteratively select the most optimal choice at every iteration
- Greedy rejection: Initialized using all available choices and iteratively drop the least optimal choice at every iteration






