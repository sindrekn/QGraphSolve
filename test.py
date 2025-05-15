import preform_sweeps as PS
import numpy as np

# Reqierd variables
# SA: 
# Number of nodes (int)
N = 100
# Path to the graph file (string)
graph_path = 'graph.csv' 
# Number of sweeps (int)
sweeps = N**2

# SQA (in addition to SA):
# NUmber of Trotter slices (int)
P = 64
# ----------------------------
# Additional variables for SA
# Number of runs (int)
runs = 1
# Initial state (In SA needs to in size of N, while in SQA it needs to be in size of [N, P])
state = 0
# Beta schedule (np array of size sweeps, else 0)
beta = 0
# Energy return (bool)
energy_return = True
# Parallel execution (bool)
# This will only work if prefom_sweeps.py is run as main program
parallel = False

# Additional variables for SQA (in addition to SA)
# runs=1, state=0, PT=1, beta=0, alpha=1, mu=0, Jts=0, tcs=0, global_moves=False, energy_return=False, correlation=False, parallel=False
# The temparature in SQA (Can be negelected if using beta) (int)
PT = 1.1
# The alpha parameter in SQA (decding the annealing schedule) (float)
alpha = 2
# The mu parameter in SQA (deciding the annealing schedule) (float between 0 and 4)
mu = 0
# The Jt parameter in SQA (deciding the annealing schedule) (np array of size sweeps, else 0)
Jts = 0
# The tc parameter in SQA (deciding the annealing schedule) (np array of size sweeps, else 0)
tcs = 0
# Global moves on/off in SQA (bool)
global_moves = False
# The correlation function return/nonreturn in SQA (bool)
correlation = False

states, energies, correlations = PS.run_SQA(graph_path, N, P, sweeps, runs=runs, state=state, PT=PT, beta=beta, alpha=alpha, mu=mu, Jts=Jts, tcs=tcs, global_moves=global_moves, energy_return=energy_return, correlation=correlation, parallel=parallel)

# State[run][Trotter slice][Node]
# Energies[run][Trotter slice][sweep]
# Correlations[run][sweep]

print(f'SQA: {energies[0][30]}')

states, energies = PS.run_SA(graph_path, N, sweeps, runs=runs, state=state, beta=beta, energy_return=energy_return, parallel=parallel)

# State[run][Node]
# Energies[run][sweep]

print(f'SA: {energies[0]}')

