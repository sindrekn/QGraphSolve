# crank_nicholson_cpu.py
# -----------------------------------
# CPU implementation of the Crank-Nicholson method for quantum annealing
# -----------------------------------

import numpy as np
from scipy.sparse import kron, identity, csr_matrix
from scipy.sparse.linalg import eigsh, gmres
import pandas as pd
import time
import os

# ----------------------
# Utility functions
# ----------------------
def apply_operator(operator, target_node, num_nodes):
    """
    Applies a single-node operator to the target node in an N-node system using Kronecker products.
    """
    I = identity(2, format='csr')
    result = I if target_node != 0 else operator
    for i in range(1, num_nodes):
        result = kron(result, operator if i == target_node else I, format='csr')
    return result

# ----------------------
# Hamiltonian construction
# ----------------------
def Hamiltonian(N, graph_path):
    """
    Constructs the driver (H_D) and problem (H_P) Hamiltonians for the system.
    """
    sigma_x = csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
    sigma_z = csr_matrix(np.array([[1, 0], [0, -1]], dtype=float))

    # Load coupling constants from CSV file
    df = pd.read_csv(graph_path)
    coupling_constants = {(df['Node1'][i], df['Node2'][i]): df['Weight'][i] 
                        for i in range(len(df)) if df['Node1'][i] < N and df['Node2'][i] < N}

    # Driver Hamiltonian: sum of sigma_x on each node
    H_D = sum(apply_operator(sigma_x, n, N) for n in range(N))
    # Problem Hamiltonian: sum over weighted sigma_z sigma_z couplings
    H_P = sum(J * apply_operator(sigma_z, i, N).dot(apply_operator(sigma_z, j, N)) 
            for (i, j), J in coupling_constants.items())

    return H_D, H_P

# ----------------------
# Main simulation routine
# ----------------------
def CPU_diagonalization(N, ta, delta_s, alpha, beta, graph_path): 
    """
    Runs the Crank-Nicholson time evolution and computes success probability and minimum gap.
    """
    # Calculate the number of time steps and where to calculate the minimum gap
    t_total = int(ta / delta_s)
    t_mid = int(t_total / 2)

    # Annealing schedules
    x = np.linspace(0, 1, t_total)
    Gamma_s = -alpha*(beta*(x - 0.5)**3 + (1-beta/4)*x + beta/8) + alpha
    J_s = alpha*(beta*(x - 0.5)**3 + (1-beta/4)*x + beta/8)

    # Identity matrix for the full Hilbert space
    In = identity(2**N, format='csr')

    # Construct Hamiltonians
    H_D, H_P = Hamiltonian(N, graph_path)

    # Initialize the wave function in the ground state of H_D
    psi = eigsh(-H_D, k=1, which='SA')[1][:, 0]

    # Time evolution loop
    for t in range(1, t_total):
        # Hamiltonian at time s
        H_s = -Gamma_s[t] * H_D - J_s[t] * H_P

        # Crank-Nicholson update (using GMRES solver)
        A = In - 0.5j * delta_s * H_s
        B = In + 0.5j * delta_s * H_s
        psi, info = gmres(B, A.dot(psi))

        # Ensure normalization
        psi = psi / np.linalg.norm(psi)

    # Calculate the minimum energy gap at the midpoint
    H_mid = -Gamma_s[t_mid] * H_D - J_s[t_mid] * H_P
    eigenvalues, eigenvectors = eigsh(H_mid, k=3, which='SA')
    min_diff = eigenvalues[2] - eigenvalues[0]

    # Diagonalize final Hamiltonian for success probability
    eigenvalues, eigenvectors = eigsh(H_s, k=3, which='SA')
    # Probability to end in one of the adiabatic ground states
    success_probabilities = np.abs(eigenvectors[:, 0].conj().dot(psi))**2
    success_probabilities += np.abs(eigenvectors[:, 1].conj().dot(psi))**2

    return success_probabilities, min_diff, psi

# ----------------------
# Main loop over graphs
# ----------------------
def main(N = 8, graph=1, ta=14, delta_s=0.02, alpha=1, beta=0, graph_path="", save=False): 
    # ----------------------
    # Parameters and paths
    # ----------------------
    # N: int = 16  # Number of nodes in the graph
    # graph: int = 1  # Graph index to use
    # ta: float = 14  # Total annealing time
    # delta_s: float = 0.02  # Time step for the simulation
    # alpha: float = 1  # Annealing parameter
    # beta: float = 0  # Annealing parameter
    # graph_path: str = 'path/to/graph.csv'  # Path to the graph CSV file
    # save: bool = False  # Whether to save the results to a CSV file

    # ----------------------
    # Find the path to the parent directory
    # ----------------------
    current_path = os.getcwd()
    parent_path = os.path.dirname(current_path)

    save_path = os.path.join(parent_path, 'ExGraphs', 'Results', 'SD')

    if graph_path == "":  # If no path is provided, construct it based on the current working directory

        # Path to graph CSV files and save directory
        # (Update these paths as needed for your system)
        graph_path = os.path.join(parent_path, 'ExGraphs', 'SmallGraphs', f'Graphs{N}Nodes', f'RandomGraph{graph}.csv')

    # ----------------------
    # DataFrame for results
    # ----------------------
    Data = pd.DataFrame(columns=['Graph', 'Success probability for GS', 'min_diff', 'Time'])

    # Start timing
    time_start = time.time()

    # Run the simulation
    success_probabilities, min_diff, psi = CPU_diagonalization(N, ta, delta_s, alpha, beta, graph_path)

    # End timing
    time_run = time.time() - time_start

    # Save the results to the DataFrame
    new_data_row = pd.DataFrame([[graph, success_probabilities, min_diff, time_run]], 
                                columns=['Graph', 'Success probability for GS', 'min_diff', 'Time'])
    if Data.empty:
        Data = new_data_row
    else: 
        Data = pd.concat([Data, new_data_row], ignore_index=True)

    # Print results for this graph
    print(f"Graph {graph} completed")
    print(f"Ground state probability: {success_probabilities}")
    print(f"Minimum energy gap: {min_diff}")
    print(f"Time: {time_run}")

    # Save the DataFrame to CSV after each graph
    if save:
        Data.to_csv(save_path + f'/Data_Node{N}_Graph{graph}_Time{ta}.csv', index=False)
    
main()


