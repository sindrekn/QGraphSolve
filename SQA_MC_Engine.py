import numpy as np
import pandas as pd
from numba import njit

@njit
def Hamiltonian(Jt, tc, state, csr_data, csr_index, csr_inptr, P, N):
    # Compute the enrgy of the system
    E = 0
    # Sum over all Trotter slices
    for k in range(P): 
        # Sum over all vertices
        for i in range(N): 
            # Sum over all edges
            for j in range(csr_inptr[i], csr_inptr[i+1]):
                E -= Jt * csr_data[j] * state[csr_index[j], k] * state[i, k]
            E -= tc * state[i, k] * state[i, k-1]      
    return E

@njit
def Hamiltonian_Slice(state, csr_data, csr_index, csr_inptr, N):
    # Use spasre matrix to calculate the total energy of one slice
    E = 0
    for i in range(N):
        for j in range(csr_inptr[i], csr_inptr[i+1]):
            E -= csr_data[j] * state[csr_index[j]] * state[i]
    return E

@njit
def CutValue(csr_data, csr_index, csr_inptr, state, N):
    # Compute the cut value of the state
    cut = 0
    for i in range(N):
        for j in range(csr_inptr[i], csr_inptr[i+1]):
            cut += 0.5* (-csr_data[j]) * (1 - state[csr_index[j]] * state[i])
    return cut

@njit
def correlation_function(state, P, N): 
    # Initialize the total correlation value
    C = 0

    # Loop over all vertices (spins)
    for i in range(N): 
        # Initialize the correlation for the current node replicas
        C_i = 0

        # Loop over all pairs of Trotter slices (k1, k2)
        for k1 in range(P): 
            # Loop over the next two Trotter slices (k2)
            for k2 in range(k1 + 1, k1 + 3): 
                # Compute the product of spins in slices k1 and k2 for node i
                C_i += state[i, k1] * state[i, k2 % P]

        # Normalize the correlation for the current node by the number of slice pairs
        C += C_i / (P * (3 - 1))

    # Normalize the total correlation by the number of vertices
    C /= N

    # Return the average correlation
    return C

@njit
def dE_local(Jt, tc, state, csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, P, i, k): 
    # Compute the change in energy for a local
    dE = 0

    # Sum over all vertices in the Trotter slice
    for j in range(csr_inptr[i], csr_inptr[i+1]):
        dE -= csr_data[j] * state[csr_index[j], k]
    for j in range(csc_inptr[i], csc_inptr[i+1]):
        dE -= csc_data[j] * state[csc_index[j], k]
    dE *= 2 * Jt

    # Compute the change in energy for neighboring Trotter slices
    if k+1 < P: 
        dE -= 2 * tc * (state[i, k+1] + state[i, k-1])
    else: 
        dE -= 2 * tc * (state[i, 0] + state[i, k-1])
    return dE * state[i, k]

@njit
def local_move(Jt, tc, state, csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, P, beta, E, i, k):
    # One move along one Trotter slice (local move)
    # Flip the spin
    state[i, k] *= -1

    # Compute the change in energy
    dE = dE_local(Jt, tc, state, csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, P, i, k)

    # Accept or reject the move
    if dE > 0:
        exp_term = np.exp(-beta * dE / P)
        if np.random.random() < exp_term:
            E += dE
        else:
            state[i, k] *= -1
    else:
        E += dE

    return state, E

@njit
def dE_global(Jt, state, csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, P, i):
    # Compute the change in energy for a move with sparse matrix
    dE = 0

    # Sum over all Trotter slices
    for k in range(P):

        # Compute the change in energy foreach Trotter slice
        for j in range(csr_inptr[i], csr_inptr[i+1]):
            dE -= csr_data[j] * state[csr_index[j], k] * state[i, k]
        for j in range(csc_inptr[i], csc_inptr[i+1]):
            dE -= csc_data[j] * state[csc_index[j], k] * state[i, k]

    return dE * 2 * Jt

@njit
def global_move(Jt, state, csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, P, beta, E, i): 
    # One move along all the Trotter slices (global move)
    # Flip the spins along all the Trotter slices
    for k in range(P): 
        state[i, k] *= -1

    # Compute the change in energy
    dE = dE_global(Jt, state, csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, P, i)

    # Accept or reject the move
    if dE > 0:
        exp_term = np.exp(-beta * dE / P)
        if np.random.random() < exp_term:
            E += dE
        else:
            for k in range(P): 
                state[i, k] *= -1
    else:
        E += dE

    return state, E

@njit
def metropolis(Jt, tc, state, csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, P, N, beta, E): 
    # One sweep of the configuration
    # Perform the local and global moves N*P times
    i = 0
    k = 0

    for move in range(N*P): 
        if move % (N*P) == 0:
            i = 0
            k = 0

        # Update the state and energy
        state, E = local_move(Jt, tc, state, csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, P, beta, E, i, k)
        k += 1
        
        if (move + 1) % P == 0:
            i += 1
            k = 0

    return state, E

@njit
def metropolis_w_global(Jt, tc, state, csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, P, N, beta, E): 
    # One sweep of the configuration
    # Perform the local and global moves N*P times
    i = 0
    k = 0

    for move in range(N*P): 
        if move % (N*P) == 0:
            i = 0
            k = 0

        # Update the state and energy
        state, E = local_move(Jt, tc, state, csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, P, beta, E, i, k)
        k += 1
    
        if move % (N*P) == 0:
            state, E = global_move(Jt, state, csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, P, beta, E, i)
        
        if (move + 1) % P == 0:
            i += 1
            k = 0

    return state, E

@njit
def preform_SQA_sweeps(csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, P, N, beta, tcs, Jts, state, global_moves, energy_return, correlation): 
    # Initialize the Spin Glass properties
    sweeps = len(tcs)

    # Preper the energy array, where each row is the energy of a Trotter slice
    Energies = np.zeros((P, sweeps), dtype=np.float64)

    # Preper the correlation function array, where each row is the correlation function of a Trotter slice
    C_m_row = np.zeros(sweeps, dtype=np.float64)

    # Compute the initial energy
    E = Hamiltonian(Jts[0], tcs[0], state, csr_data, csr_index, csr_inptr, P, N)

    # Perform the n sweeps
    for sweep in range(sweeps):
        # Perform a sweep of the Metropolis algorithm
        if global_moves: 
            # Perform SQA with global and local moves
            state, E = metropolis_w_global(Jts[sweep], tcs[sweep], state, csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, P, N, beta[sweep], E)
        else: 
            # Perform SQA with only local moves
            state, E = metropolis(Jts[sweep], tcs[sweep], state, csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, P, N, beta[sweep], E)

        if energy_return: 
            # Compute the energy of each Trotter slice
            for k in range(P): 
                Energies[k, sweep] = Hamiltonian_Slice(state[:, k], csr_data, csr_index, csr_inptr, N)
        
        if correlation:
            # Compute the finite correlation function
            C_m_row[sweep] = correlation_function(state, P, N)
            # Store the correlation function for this run

    return state, Energies, C_m_row

def get_spin_arr(N, P):
    # Generate a random spin array with N vertices and P Trotter slices
    state = np.zeros((N, P), dtype=np.int8)

    # Generate the Trotter slices
    for k in range(P): 
        for i in range(N): 
            if np.random.rand() < 0.5: 
                state[i, k] = 1
            else:
                state[i, k] = -1
    return state

def update_fields(t, alpha, mu):
    # Linear to saddle point schedule for the fields
    gamma = -alpha*(mu*(t - 0.5)**3 + (1-mu/4)*t + mu/8) + alpha
    Jt = alpha*(mu*(t - 0.5)**3 + (1-mu/4)*t + mu/8)
    return gamma, Jt

def Trotter_coeffient(P, gamma, beta):
    # Update the Trotter coefficient, but keep it finite, aka at 
    # gamma < 10^-10 we set it constant to 10^-8
    if gamma > 10**-20:
        tc = -0.5 * (P / beta) * np.log(np.tanh(gamma*beta/P))
    else:
        tc = -0.5 * (P / beta) * np.log(np.tanh(10**-20*beta/P))
    return tc

def sparse_csr_generator(N, input_file): 
    # Generate a sparse CSR matrix from the GSET dataset
    # Make the csv file into a pandas dataframe
    df = pd.read_csv(input_file)
    
    # Initialize the sparse CSR matrix
    csr_index = []
    csr_ptr = [0]
    csr_data = []

    i = 0
    last_row = 0

    # Move through every row in the dataframe
    for line in df.iterrows(): 
        # Check if the row is less than N and greater than M
        if line[1]['Node1'] <= N:
            row = line[1]['Node1']

            # Add error if the row is less than 0
            if row < 0:
                raise ValueError('Row index less than 0')

            # Make a pointer from row to colum indices
            if row == last_row:
                None
            else:
                for _ in range(row - last_row):
                    csr_ptr.append(i)
                last_row = row
            
            # Check if the column is less than N and greater than M, then add the data to the CSR matrix
            if line[1]['Node2'] <= N:
                csr_data.append(line[1]['Weight'])
                csr_index.append(line[1]['Node2'])
                i += 1

    # This makes sure that the last row is included in the csr_ptr (works only for 0 diagonal matrices)
    for _ in range(len(csr_ptr), N + 1):
        csr_ptr.append(i)

    # Convert the CSR matrix to numpy arrays and return them
    csr_data = np.array(csr_data, dtype=np.int32)
    csr_index = np.array(csr_index, dtype=np.int32)
    csr_ptr = np.array(csr_ptr, dtype=np.int32)
    return csr_data, csr_index, csr_ptr

def csr_to_csc(csr_data, csr_index, csr_ptr, N):
    # Generate a CSC matrix from a CSR matrix
    # Number of non-zero elements
    nnz = len(csr_data)
    
    # Initialize CSC arrays
    csc_data = np.zeros(nnz, dtype=csr_data.dtype)
    csc_index = np.zeros(nnz, dtype=csr_index.dtype)
    csc_indptr = np.zeros(N + 1, dtype=csr_ptr.dtype)
    
    # Step 1: Count the number of non-zero elements in each column
    col_counts = np.zeros(N, dtype=np.int32)
    for i in range(N):
        for j in range(csr_ptr[i], csr_ptr[i + 1]):
            col_counts[csr_index[j]] += 1
    
    # Step 2: Compute csc_indptr (column pointers)
    csc_indptr[0] = 0
    for i in range(1, N + 1):
        csc_indptr[i] = csc_indptr[i - 1] + col_counts[i - 1]
    
    # Step 3: Fill csc_csr_data and csc_csr_index using the CSR csr_data
    current_positions = np.zeros(N, dtype=np.int32)
    for i in range(N):
        for j in range(csr_ptr[i], csr_ptr[i + 1]):
            col = csr_index[j]
            pos = csc_indptr[col] + current_positions[col]
            csc_data[pos] = csr_data[j]
            csc_index[pos] = i
            current_positions[col] += 1
    
    return csc_data, csc_index, csc_indptr

def setup(graphpath, N): 
    # Generate the sparse matrix and set up inital state
    csr_data, csr_index, csr_inptr = sparse_csr_generator(N, graphpath)
    csc_data, csc_index, csc_inptr = csr_to_csc(csr_data, csr_index, csr_inptr, N)
    return csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr