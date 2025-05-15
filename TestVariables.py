import numpy as np
import os

def path(GraphPath): 
    # Check if the graph file exists
    if not os.path.isfile(GraphPath): 
        raise ValueError('The graph file does not exist')

def test_N(N):
    # Check if the number of spins is a positive integer
    if N <= 0: 
        raise ValueError('The number of spins must be a positive integer')

def test_P(P):
    # Check if the number of Trotter slices is a positive integer
    if P <= 0 or not isinstance(P, int): 
        raise ValueError('The number of Trotter slices must be a positive integer')

def test_sweeps(sweeps):
    # Check if the number of sweeps is a positive integer
    if sweeps <= 0 or not isinstance(sweeps, int): 
        raise ValueError('The number of sweeps must be a positive integer')

def test_PT(PT):
    # Check if the temperature is a positive float
    if PT <= 0: 
        raise ValueError('The temperature must be a positive float')

def test_alpha(alpha):
    # Check if the alpha parameter is a positive float
    if alpha <= 0: 
        raise ValueError('The alpha parameter must be a positive float')

def test_mu(mu):
    # Check if the mu parameter is a positive float
    if mu < 0 or mu > 4: 
        raise ValueError('The mu parameter must be a positive float between 0 and 4')

def test_beta(beta):
    # Check if the beta parameter is a positive float
    if not isinstance(beta, np.ndarray): 
        if beta != 0: 
            raise ValueError('The beta parameter must be a numpy array or 0 to be generated')

def test_state(state):
    # Check if the state is a numpy array
    if not isinstance(state, np.ndarray): 
        if state != 0:
            raise ValueError('The state must be a numpy array or 0 to generate a random state')
    
def test_Jts(Jts):
    # Check if the interaction strengths are a numpy array
    if not isinstance(Jts, np.ndarray): 
        if Jts != 0:
            raise ValueError('The interaction strengths must be a numpy array or 0 to be generated')
    
def test_tcs(tcs):
    # Check if the Trotter coefficients are a numpy array
    if not isinstance(tcs, np.ndarray): 
        if tcs != 0:
            raise ValueError('The Trotter coefficients must be a numpy array or 0 to be generated')

def test_boolean(boolean): 
    # Check if the boolean parameter is a boolean
    if not isinstance(boolean, bool): 
        raise ValueError('The boolean parameter must be a boolean')
    

