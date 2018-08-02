

#######################################################
## Personalized Gaussian Process - Methods & Classes ## 
#######################################################

import numpy as np
import scipy as SP
import scipy.linalg as linalg
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import logging

def square_dist(ls, X, X2):

    X = X / ls #divide the X tensor with the lengthscales

    X_square = np.sum(np.square(X), axis = 1) #get the sum of the square of all values in X

    if X2 is None: #if there is no X2 value...

        #get the product of X and X and add the sum of the squares to it
        X_transpose = np.transpose(X)
        dist = -2 * np.matmul(X, X_transpose)
        dist += np.reshape(X_square, (-1, 1)) + np.reshape(X_square, (1, -1))
        return dist

    X2 = X2 / ls #if there is an X2, divide by lengthscales

    X2_square = np.sum(np.square(X2), axis = 1) #get the sum of the squares of all of the values in X2

    dist = -2 * np.matmul(X, np.transpose(X2)) #get the product of X and X2

    dist += np.reshape(X_square, (-1, 1)) + np.reshape(X2_square, (1, -1)) #add the sum of the squares of the values in X and X2

    return dist

def K(ls, var, X, X2=None):
    """
    Kernel function
    Return: variance times exponential of squared distance between X and X2
    """
    return var * np.exp(-square_dist(ls, X, X2) / 2)

def Kdiag(var, X):
    """
    Return: tensor with just the variances along 0th dimension of X
    """
    return np.full((1, X.shape[0]), float(var))

#modified from http://gpy.readthedocs.io/en/deploy/index.html
def jitChol(A, maxTries=10, warning=True):
    """
    Applies jitter to cholesky decomposition when eigenvalues are negative
    """
    warning = True
    jitter = 0
    i = 0

    while(True):
        try: #check if A is positive definite 
            if jitter == 0:
                jitter = abs(SP.trace(A))/A.shape[0]*1e-6
                LC = linalg.cholesky(A, lower=False) 
                return LC.T 
            else:
                if warning:
                    logging.error("Adding jitter of %f in jitChol()." % jitter)
                LC = linalg.cholesky(A+jitter*SP.eye(A.shape[0]), lower = False) 
                return LC.T 
        except linalg.LinAlgError: #if non-positive definite, apply jitter 
            if i<maxTries:
                jitter = jitter*10
            else:
                raise linalg.LinAlgError
        i += 1

def mu(k, alpha):
    """
    Return: mean 
    """
    return np.dot(k, alpha)

def sigma(k, V):
    """
    Return: variance 
    """
    array = np.asarray(np.sum(V*V))
    array = array.astype(float)
    return k - array

def call_source(model, xtest):
    """
    Return: mean and variance predicted from source model 
    
    Input: source model, test input data
    Output: source model mean, source model variance, model parameters
    """
    #store mean and variance of xtest from source model (based on GPFlow)
    m_s, s_s = model.predict_y(xtest)
    
    #obtain model parameters 
    m_params = model.read_values()

    return m_s, s_s, m_params

def call_adapt(x_a, y_a, x_s, y_s, xtest, m_s, s_s, ls, var, sn2):
    """
    Return: mean and variance predicted from adaptation model 
    
    Input: adaptation data, source data, test data, mean and variance from source model, model parameters 
    Output: adapted model mean, adapted model variance
    """
    
    #ADAPTATION
    m_adapt = np.array([])
    s_adapt = np.array([])
    
    #COMPUTATIONS BEFORE FOR LOOP
    K_ts_star_all = K(ls, var, x_s, xtest)
    print('k ts star all shape', K_ts_star_all.shape, type(K_ts_star_all))
    K_tt_all = K(ls, var, x_a)
    K_t_star_all = K(ls, var, x_a, xtest)
    
    K_s = K(ls, var, x_s)
    dim_K_s = K_s.shape[0]
    L_arg = K_s + sn2*np.identity(dim_K_s) 
    L = jitChol(L_arg) 
    
    alpha_denom = np.linalg.lstsq(L,y_s)[0]
    alpha = np.linalg.lstsq(L.transpose(),alpha_denom)[0]

    for i in range(1, len(x_a)+1): #1 to 20 (or 1 to 21 exclusive)

        if i == 1:
            m_adapt = np.append(m_adapt, [m_s[0]]) #mean predicted from X1
            s_adapt = np.append(s_adapt, [s_s[0]]) #var predicted from X1
            
        print('ENTERING FOR LOOP:', i)
        #adaptation data for subject
        y_a_patient = y_a[0:i] #0:1 to 0:20

        #ADAPTATION CALCULATIONS
        #K_ts, K_tt
        K_ts = K_ts_star_all[:,:i]
        K_tt = K_tt_all[:i,:i]

        #alpha_adapt
        V = np.linalg.lstsq(L, K_ts)[0]
        mu_t = mu(K_ts.transpose(), alpha)
        K_tt_dim = K_tt.shape[0]
        C_t = K_tt - np.dot(V.transpose(), V) + sn2*np.identity(K_tt_dim)
        L_adapt = jitChol(C_t)
        alpha_adapt = SP.linalg.cho_solve((L_adapt, True), y_a_patient - mu_t)

        #V_adapt
        K_t_star = K_t_star_all[:i, i:i+1]
        K_ts_star = K_ts_star_all[:, i:i+1]
        V_star = np.linalg.lstsq(L, K_ts_star)[0]
        V_dot = np.dot(V_star.transpose(), V)
        C_t_star = K_t_star - V_dot.transpose() 
        V_adapt = np.linalg.lstsq(L_adapt, C_t_star)[0]

        add_adapt = np.dot(C_t_star.transpose(), alpha_adapt)
        m_adapt = np.vstack((m_adapt, [m_s[i] + add_adapt[0]]))
        
        s_adapt_ele = sigma(s_s[i], V_adapt)
        s_adapt = np.vstack((s_adapt, [s_adapt_ele]))
    
    return m_adapt, s_adapt 

def call_target(x_a, y_a, xtest, m_s, s_s, ls, var, sn2):
    """
    Return: mean and variance predicted from target model 
    
    Input: adaptation data, source data, test data, mean and variance from source model, model parameters 
    Output: target model mean, target model variance
    """

    #TARGET 
    m_target = np.array([])
    s_target = np.array([])
    
    #COMPUTATIONS BEFORE FOR LOOP 
    K_ts_star_all = K(ls, var, x_a, xtest)  # 20 x 21, x_a x xtest
    k_star_star_all = Kdiag(var, xtest)
    K_s_all = K(ls, var, x_a)  # 20 x 20
    
    for i in range(1, len(x_a) + 1): #1 to 20 (or 1 to 21 exclusive)

        if i == 1:
            m_target = np.append(m_target, [m_s[0]]) #mean predicted from X2
            s_target = np.append(s_target, [s_s[0]]) #var predicted from X2
        
        print('ENTERING FOR LOOP:', i)
        
        #adaptation data for subject
        y_a_patient = y_a[0:i] #0:1 to 0:20

        #CALCULATION of target mean and variance
        #K_ts_star: 1x1, 2x1, 3x1, etc... 
        K_ts_star = K_ts_star_all[:i, i:i+1]

        #k_star_star: 1x1, 1x1, 1x1, etc...
        k_star_star = np.array([k_star_star_all[0][i]])

        #V_star
        #K_s: 1x1, 2x2, 3x3, etc... 
        K_s = K_s_all[:i, :i]
        dim_K_s = K_s.shape[0]
        L_arg = K_s + sn2*np.identity(dim_K_s)
        L = jitChol(L_arg)
        V_star = np.linalg.lstsq(L, K_ts_star)[0]

        #m_target
        alpha_denom = np.linalg.lstsq(L, y_a_patient)[0]
        alpha = np.linalg.lstsq(L.transpose(), alpha_denom)[0]
        m_target_ele = mu(K_ts_star.transpose(), alpha)

        #s_target
        s_target_ele = sigma(k_star_star, V_star)

        #constructing mean, variance array
        m_target = np.vstack((m_target, m_target_ele))
        s_target = np.vstack((s_target, s_target_ele[0]))

    return m_target, s_target

def call_pgp(m, x_s, y_s, x_a, y_a, xtest, k):
    """
    Calls personalized gaussian process 
    
    Input: optimized model, source data, adaptation data, test input data, kernel
    Output: m_s, s_s, m_a, s_a, m_t, s_t, g_t, model parameters
    """
    #SOURCE MODEL - mean, variance, parameters
    print('----- CALL SOURCE MODEL -----')
    m_s, s_s, m_params = call_source(m, xtest) #predictions done on X1->Y1 to X21->Y21
        
    #get model parameters - as values
    ls = m.read_values()['GPR/kern/lengthscales']
    var = m.read_values()['GPR/kern/variance']
    lik = float(m.likelihood.variance.value)
    sn2 = lik 
    
    #make s_s into nx1 vector
    length = len(s_s)
    s_s = np.mean(s_s, axis = 1).reshape((length, 1))
    
    #ADAPTATION MODEL
    print('----- CALL ADAPTATION MODEL -----')
    m_a, s_a = call_adapt(x_a, y_a, x_s, y_s, xtest, m_s, s_s, ls, var, sn2) #prediction for Y1 taken from source model, Y2->Y21 from adaptation model
                              
    #TARGET MODEL
    print('----- CALL TARGET MODEL -----')
    m_t, s_t = call_target(x_a, y_a, xtest, m_s, s_s, ls, var, sn2)

    #JOINT MODEL
    print('----- CALL JOINT MODEL -----')
    m_j = (m_a + m_t)/2
    s_j = (s_a + s_t)/2

    kern_lengthscale = m_params['GPR/kern/lengthscales']
    kern_variance = m_params['GPR/kern/variance']
    likelihood_variance = m_params['GPR/likelihood/variance']

    values = {'source model mu': m_s, 'source model sigma': s_s, 'adapted model mu': m_a, 'adapted model sigma': s_a, 'target model mu': m_t, 'target model sigma': s_t, 'joint model mu': m_j, 'joint model sigma': s_j, 'param - kernel ls': kern_lengthscale, 'param - kernel var': kern_variance, 'param - likelihood var': likelihood_variance}

    return values
