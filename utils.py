# Gaussian Process Functions  
## Defines functions for computing GP predictions 

import logging
import numpy as np
import scipy as SP
import scipy.linalg as linalg

def jitChol(A, maxTries=10, warning=True):
    """ 
    Jitter Cholesky Function 
    Applies jitter to cholesky decomposition when eigenvalues are negative
    Modified from http://gpy.readthedocs.io/en/deploy/index.html
    """
    warning = True
    jitter = 0
    i = 0

    while(True):
        try: # Check if A is positive definite 
            if jitter == 0:
                jitter = abs(SP.trace(A))/A.shape[0]*1e-6
                LC = linalg.cholesky(A, lower=False) 
                return LC.T 
            else:
                if warning:
                    logging.error("Adding jitter of %f in jitChol()." % jitter)
                LC = linalg.cholesky(A+jitter*SP.eye(A.shape[0]), lower = False) 
                return LC.T 
        except linalg.LinAlgError: # If non-positive definite, apply jitter 
            if i<maxTries:
                jitter = jitter*10
            else:
                raise linalg.LinAlgError
        i += 1

def mu(k, alpha):
    """ Mean Function """
    return np.dot(k, alpha)

def sigma(k, V):
    """ Variance Function """
    array = np.asarray(np.sum(V*V))
    array = array.astype(float)
    return k - array
