# Gaussian Process Functions  
## Defines functions for computing GP predictions 

import logging
import numpy as np
import scipy as SP
import scipy.linalg as linalg

def process_data(X, Y, indicators, IDs, tr_IDs, te_IDs):
    """
    Processes data 
    Returns training and testing data 
    """
    X_tr, Y_tr = None, None 
    for tr in tr_IDs:
        tr_rows = np.where(IDs == tr)[0]

        X_tr = X[tr_rows, :] if X_tr is None else np.vstack((X_tr, X[tr_rows, :]))
        Y_tr = Y[tr_rows, :] if Y_tr is None else np.vstack((Y_tr, Y[tr_rows, :]))

    X_te, Y_te, ind_te, ID_te = None, None, None, None 
    for te in te_IDs:
        te_rows = np.where(IDs == te)[0]

        X_te = X[te_rows, :] if X_te is None else np.vstack((X_te, X[te_rows, :]))
        Y_te = Y[te_rows, :] if Y_te is None else np.vstack((Y_te, Y[te_rows, :]))
        ind_te = indicators[te_rows, :] if ind_te is None else np.vstack((ind_te, indicators[te_rows, :]))
        ID_te = IDs[te_rows, :] if ID_te is None else np.vstack((ID_te, IDs[te_rows, :]))

    return X_tr, Y_tr, X_te, Y_te, ind_te, ID_te 

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

if __name__ == '__main__':
    pass