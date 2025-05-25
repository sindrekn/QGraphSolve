import preform_sweeps as PS
import os


def main(N = 20, graph_num = 1, graph_path = '', sweeps = 0, P=64, runs = 1, state = 0, beta = 0, energy_return = True, parallel = False, PT = 1.1, alpha = 2, mu = 0, Jts = 0, tcs = 0, global_moves = False, correlation = False, algo = 'SQA'): 
    # Reqierd variables
    # SA: 
    # N: Number of nodes (int)
    # graph_path: Path to the graph file (string)
    # graph_num: The graph number to use (int)

    # ----------------------------
    # Graph path:
    # If no path is provided, it will be constructed based on the current working directory

    if graph_path == "":
        # Find the path to the parent directory
        current_path = os.getcwd()
        parent_path = os.path.dirname(current_path)
        # Path to graph CSV files
        graph_path = os.path.join(parent_path, 'ExGraphs', 'SmallGraphs', f'Graphs{N}Nodes', f'RandomGraph{graph_num}.csv')

    # Number of sweeps (int)
    if sweeps == 0:
        sweeps = N**2

    # SQA (in addition to SA):
    # P: Number of Trotter slices (int)
    # ----------------------------
    # Additional variables for SA

    # runs: Number of runs (int)
    # state: Initial state (In SA needs to in size of N, while in SQA it needs to be in size of [N, P])

    # beta: Beta schedule (np array of size sweeps, else 0)
    
    # energy_return: nergy return (bool)

    # parallel: Parallel execution (bool)
    # This will only work if prefom_sweeps.py is run as main program

    # Additional variables for SQA (in addition to SA)
    # runs=1, state=0, PT=1, beta=0, alpha=1, mu=0, Jts=0, tcs=0, global_moves=False, energy_return=False, correlation=False, parallel=False
    # PT: The temparature in SQA (Can be negelected if using beta) (int)

    # alpha: The alpha parameter in SQA (decding the annealing schedule) (float)

    # mu: The mu parameter in SQA (deciding the annealing schedule) (float between 0 and 4)

    # Jts: The Jt parameter in SQA (deciding the annealing schedule) (np array of size sweeps, else 0)
    
    # tcs: The tc parameter in SQA (deciding the annealing schedule) (np array of size sweeps, else 0)
    
    # global_moves: Global moves on/off in SQA (bool)
    
    # correlation: The correlation function return/nonreturn in SQA (bool)

    # ----------------------------
    # Run the SQA algorithm
    # The function will return the states, energies and correlations
    # states: [run][Trotter slice][Node]
    # energies: [run][Trotter slice][sweep]
    # correlations: [run][sweep]

    if algo == 'SQA':
        states, energies, correlations = PS.run_SQA(graph_path, N, P, sweeps, runs=runs, state=state, PT=PT, beta=beta, alpha=alpha, mu=mu, Jts=Jts, tcs=tcs, global_moves=global_moves, energy_return=energy_return, correlation=correlation, parallel=parallel)

        print(f'SQA: {energies[0][30]}')

        # To find cut value: 
        import SQA_MC_Engine as SQA
        csr_data, csr_index, csr_ptr = SQA.sparse_csr_generator(N, graph_path)

        cut = SQA.CutValue(csr_data, csr_index, csr_ptr, states[0][30], N)
        print(f'Cut value: {cut}')
    else:
        # Run the SA algorithm
        # The function will return the states and energies
        # states: [run][Node]
        # energies: [run][sweep]
        # correlations: [run][sweep] (if energy_return is True)

        states, energies = PS.run_SA(graph_path, N, sweeps, runs=runs, state=state, beta=beta, energy_return=energy_return, parallel=parallel)

        print(f'SA: {energies[0]}')

