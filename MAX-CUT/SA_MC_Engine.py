import numpy as np
from numba import njit

@njit
def CutValue(csr_data, csr_index, csr_inptr, state, N):
    # Compute the cut value of the state
    cut = 0
    for i in range(N):
        for j in range(csr_inptr[i], csr_inptr[i+1]):
            cut += 0.5* (-csr_data[j]) * (1 - state[csr_index[j]] * state[i])
    return cut

@njit
def Hamiltonian(state, csr_data, csr_index, csr_inptr, N):
    # Use spasre matrix to calculate the total energy of the system
    E = 0
    # Sum over affected nodes
    for i in range(N):
        for j in range(csr_inptr[i], csr_inptr[i+1]):
            E -= csr_data[j] * state[csr_index[j]] * state[i]
    return E

@njit
def dE(state, csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, n):
    # Compute the change in energy for a move with sparse matrix
    dE = 0
    for i in range(csr_inptr[n], csr_inptr[n+1]):
        dE -= csr_data[i] * state[csr_index[i]]
    for i in range(csc_inptr[n], csc_inptr[n+1]):
        dE -= csc_data[i] * state[csc_index[i]]
    return dE * 2 * state[n]

@njit
def metropolis(state, csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, beta, E, i):
    # One move of the Metropolis algorithm
    # Choose a random spin vertice
    # Flip the spin
    state[i] *= -1

    # Compute the change in energy
    dE_ = dE(state, csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, i)

    # Accept or reject the move
    if dE_ > 0:
        exp_term = np.exp(-beta * dE_)
        if np.random.random() < exp_term:
            E += dE_
        else:
            state[i] *= -1
    else:
        E += dE_

    return state, E

@njit
def preform_SA_sweeps(csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, N, beta, state, energy_return):
    # Initialize the Spin Glass properties
    sweeps = len(beta)
    E = Hamiltonian(state, csr_data, csr_index, csr_inptr, N)
    Es = np.zeros(sweeps, dtype=np.float64)

    # Perform the MCS
    for sweep in range(sweeps):
        # Perform a sweep of the Metropolis algorithm
        i = 0
        for move in range(N):
            state, E = metropolis(state, csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, beta[sweep], E, i)
            i += 1
        if energy_return:
            Es[sweep] = E

    return state, Es

def beta_schedule(beta_i, beta_f, nsweeps):
    # Initialize the beta schedule
    beta_sc = []
    delta = (beta_f - beta_i)/(nsweeps - 1)

    # Update the beta schedule
    for i in range(nsweeps): 
        beta_sc.append(beta_i + delta*i)

    return beta_sc

def get_spin_array(N): 
    # Generate an inital state for the simulation
    state = np.zeros(N, dtype=np.int8)
    for i in range(N): 
        if np.random.rand() < 0.5: 
            state[i] = 1
        else:
            state[i] = -1
    return state






