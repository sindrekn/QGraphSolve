import numpy as np
from multiprocessing import Pool
import SQA_MC_Engine as SQA
import SA_MC_Engine as SA
import TestVariables as TV

def set_up_SQA(graphpath, N, P, sweeps, state, PT, beta, alpha, mu, Jts, tcs): 
    # Validate the input parameters using the TestVariables module
    TV.path(graphpath)  # Check if the graph path is valid
    TV.test_N(N)        # Validate the number of spins
    TV.test_P(P)        # Validate the number of Trotter slices
    TV.test_sweeps(sweeps)  # Validate the number of sweeps
    TV.test_PT(PT)      # Validate the temperature parameter
    TV.test_alpha(alpha)  # Validate the alpha parameter
    TV.test_mu(mu)      # Validate the mu parameter
    TV.test_beta(beta)  # Validate the beta parameter
    TV.test_state(state)  # Validate the state parameter
    TV.test_Jts(Jts)    # Validate the interaction strengths
    TV.test_tcs(tcs)    # Validate the Trotter coefficients

    # Initialize the beta array if it is not provided as a numpy array
    if not isinstance(beta, np.ndarray):
        beta = np.ones(sweeps, dtype=np.float64) * P / PT  # Create a beta array with constant values

    # Validate and initialize Jts and tcs arrays
    if isinstance(Jts, np.ndarray): 
        if isinstance(tcs, np.ndarray): 
            pass  # Both Jts and tcs are valid numpy arrays
        else:
            raise ValueError('If Jts is a numpy array, tcs must also be a numpy array')
    else:
        # Initialize Jts and tcs as zero arrays if they are not provided
        Jts = np.zeros(sweeps, dtype=np.float64)
        tcs = np.zeros(sweeps, dtype=np.float64)
        # Compute Jts and tcs values for each sweep
        for sweep in range(sweeps): 
            gamma, Jt = SQA.update_fields(sweep, alpha, mu)  # Update gamma and Jt based on the sweep
            tc = SQA.Trotter_coeffient(P, gamma, beta[sweep])  # Compute the Trotter coefficient
            Jts[sweep] = Jt  # Store the interaction strength
            tcs[sweep] = tc  # Store the Trotter coefficient

    # Set up the sparse matrix representation of the graph
    csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr = SQA.setup(graphpath, N)

    # Return all the initialized and computed variables
    return csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, state, beta, tcs, Jts

def set_up_SA(graphpath, N, sweeps, state, beta): 
    # Validate the input parameters using the TestVariables module
    TV.path(graphpath)  # Check if the graph path is valid
    TV.test_N(N)        # Validate the number of spins
    TV.test_sweeps(sweeps)  # Validate the number of sweeps
    TV.test_state(state)  # Validate the state parameter
    TV.test_beta(beta)  # Validate the beta parameter

    # Initialize the beta array if it is not provided as a numpy array
    if not isinstance(beta, np.ndarray):
        beta = SA.beta_schedule(0.01, 10, sweeps)  # Create a beta schedule from 0 to 1

    # Set up the sparse matrix representation of the graph
    csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr = SQA.setup(graphpath, N)

    # Return all the initialized and computed variables
    return csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, state, beta

def SQA_state_init(csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, P, N, beta, tcs, Jts, state, global_moves, energy_return, correlation): 
    # Set up the initial state if not provided
    if not isinstance(state, np.ndarray): 
        state = SQA.get_spin_arr(N, P)  # Generate a random state
    
    return SQA.preform_SQA_sweeps(csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, P, N, beta, tcs, Jts, state, global_moves, energy_return, correlation)

def SA_state_init(csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, N, beta, state, energy_return): 
    # Set up the initial state if not provided
    if not isinstance(state, np.ndarray): 
        state = SA.get_spin_array(N)  # Generate a random state
    
    return SA.preform_SA_sweeps(csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, N, beta, state, energy_return)

def run_SQA(graphpath, N, P, sweeps, runs=1, state=0, PT=1, beta=0, alpha=1, mu=0, Jts=0, tcs=0, global_moves=False, energy_return=False, correlation=False, parallel=False): 
    csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, state, beta, tcs, Jts = set_up_SQA(graphpath, N, P, sweeps, state, PT, beta, alpha, mu, Jts, tcs)  # Set up the SQA parameters

    TV.test_boolean(global_moves)  # Validate the global_moves parameter
    TV.test_boolean(energy_return)  # Validate the energy_return parameter
    TV.test_boolean(correlation)   # Validate the correlation parameter

    if parallel: 
        with Pool() as pool: 
            # Use multiprocessing to run the SQA algorithm in parallel
            results = pool.starmap(SQA_state_init, [(csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, P, N, beta, tcs, Jts, state, global_moves, energy_return, correlation) for run in range(runs)])
    else: 
        # Run SQA sequentially
        results = [SQA_state_init(csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, P, N, beta, tcs, Jts, state, global_moves, energy_return, correlation) for run in range(runs)]

    # Unpack the results
    states, energies, correlations = zip(*results)

    return states, energies, correlations  # Return the states, energies, and correlations for each run

def run_SA(graphpath, N, sweeps, runs=1, state=0, beta=0, energy_return=False, parallel=False):
    csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, state, beta = set_up_SA(graphpath, N, sweeps, state, beta)  
    
    TV.test_boolean(energy_return)  # Validate the energy_return parameter

    # Set up the SA parameters
    if parallel: 
        with Pool() as pool: 
            # Use multiprocessing to run the SA algorithm in parallel
            results = pool.starmap(SA_state_init, [(csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, N, beta, state, energy_return) for run in range(runs)])
    else: 
        # Run SA sequentially
        results = [SA_state_init(csr_data, csr_index, csr_inptr, csc_data, csc_index, csc_inptr, N, beta, state, energy_return) for run in range(runs)]
    
    states, energies = zip(*results)  # Unpack the results
    return states, energies  # Return the states and energies for each run

