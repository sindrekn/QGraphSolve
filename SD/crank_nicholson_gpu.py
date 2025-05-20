# crank_nicholson_gpu.py
# -----------------------------------
# GPU implementation of the Crank-Nicholson method for quantum annealing
# -----------------------------------

import cupy as cp
from cupyx.scipy.sparse import kron, identity, csr_matrix
from cupyx.scipy.sparse.linalg import eigsh, gmres
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import time

# Suppress future warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------
# Parameters and paths
# ----------------------
N = 16  # Number of qubits

ta = 7  # Total annealing time
delta_s = 0.02  # Time step size
alpha = 1  # Schedule parameter
beta = 0   # Schedule parameter

# Path to graph CSV files and save directory
# (Update these paths as needed for your system)
graph_path = f'C:/Users/sindr/Documents/MasterResults/Diagonalization/RandomInteraction/Graphs{N}Nodes/'
save_path = f'C:/Users/sindr/Documents/MasterResults/Diagonalization/RandomInteraction/LinearSchedule/{N}Nodes/'

# ----------------------
# DataFrame for results
# ----------------------
Data = pd.DataFrame(columns=['Graph', 'Success probability for GS', 'min_diff', 'Time'])
time_steps = np.linspace(0, ta, int(ta/delta_s))

# ----------------------
# Utility functions
# ----------------------
def apply_operator(operator, target_qubit, num_qubits):
    """
    Applies a single-qubit operator to the target qubit in an N-qubit system using Kronecker products (GPU version).
    """
    I = identity(2, format='csr')
    result = I if target_qubit != 0 else operator
    for i in range(1, num_qubits):
        result = kron(result, operator if i == target_qubit else I, format='csr')
    return result

# ----------------------
# Hamiltonian construction
# ----------------------
def Hamiltonian(N, graph_path):
    """
    Constructs the driver (H_D) and problem (H_P) Hamiltonians for the system (GPU version).
    """
    sigma_x = csr_matrix(cp.array([[0, 1], [1, 0]], dtype=float))
    sigma_z = csr_matrix(cp.array([[1, 0], [0, -1]], dtype=float))

    # Load coupling constants from CSV file
    df = pd.read_csv(graph_path)
    coupling_constants = {(df['Node1'][i], df['Node2'][i]): df['Weight'][i] 
                        for i in range(len(df)) if df['Node1'][i] < N and df['Node2'][i] < N}

    # Driver Hamiltonian: sum of sigma_x on each qubit
    H_D = sum(apply_operator(sigma_x, n, N) for n in range(N))
    # Problem Hamiltonian: sum over weighted sigma_z sigma_z couplings
    H_P = sum(J * apply_operator(sigma_z, i, N).dot(apply_operator(sigma_z, j, N)) 
            for (i, j), J in coupling_constants.items())

    return H_D, H_P

# ----------------------
# Main simulation routine
# ----------------------
def GPU_diagonalization(N, ta, delta_s, alpha, beta, graph_path): 
    """
    Runs the Crank-Nicholson time evolution and computes success probability and minimum gap (GPU version).
    """
    # Calculate the number of time steps
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

        # Normalize the wave function
        psi = psi / cp.linalg.norm(psi)

    # Calculate the minimum energy gap at the midpoint
    H_mid = -Gamma_s[t_mid] * H_D - J_s[t_mid] * H_P
    eigenvalues, eigenvectors = eigsh(H_mid, k=3, which='SA')
    min_diff = eigenvalues[2] - eigenvalues[0]

    # Diagonalize final Hamiltonian for success probability
    eigenvalues, eigenvectors = eigsh(H_s, k=3, which='SA')
    # Probability to end in one of the adiabatic ground states
    success_probabilities = cp.abs(eigenvectors[:, 0].conj().dot(psi))**2
    success_probabilities += cp.abs(eigenvectors[:, 1].conj().dot(psi))**2

    return success_probabilities, min_diff, psi

# ----------------------
# Main loop over graphs
# ----------------------
for graph in range(1, 101): 
    # Path to the current graph CSV
    path = graph_path + f'RandomGraph{graph}.csv'
    
    # Start timing
    time_start = time.time()
    
    # Run the simulation
    success_probabilities, min_diff, psi = GPU_diagonalization(N, ta, delta_s, alpha, beta, path)
    
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
    Data.to_csv(save_path + f'/DataNode{N}Time{ta}Linear.csv', index=False)


