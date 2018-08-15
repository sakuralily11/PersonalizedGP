# Gaussian Process Classes 
## Defines personalizedGP and targetGP 

from utils import *

import logging
import gpflow 
import numpy as np
import scipy.linalg as linalg
import tensorflow as tf

class baseGP(object):
    def __init__(self, X, Y, kernel):
        self.X = X
        self.Y = Y 
        self.kernel = kernel 

    def train(self, X_tr, Y_tr, sGP=None, **kwargs): 
        """ Trains source GP if not provided """ 
        if sGP is None: 
        # Build sGP model 
            self.sourceGP = gpflow.models.GPR(X_tr, Y_tr, kern=self.kernel)
            self.sourceGP.likelihood.variance = np.exp(2*np.log(np.sqrt(0.1*np.var(Y_tr))))
            max_x = np.amax(X_tr, axis=0)
            min_x = np.amin(X_tr, axis=0)
            self.sourceGP.kern.lengthscales = np.array(np.median(max_x - min_x))
            self.sourceGP.kern.variance = np.var(Y_tr)

            # Optimize model 
            self.sourceGP.compile()
            opt = gpflow.train.ScipyOptimizer()
            opt.minimize(self.sourceGP)
        else:
            self.sourceGP = sGP 

        # Store parameters 
        pathname = '/'.join(list(self.sourceGP.read_trainables().keys())[0].split('/')[:-2])
        self.ls = self.sourceGP.read_values()['{}/kern/lengthscales'.format(pathname)]
        self.var = self.sourceGP.read_values()['{}/kern/variance'.format(pathname)]
        self.lik = float(self.sourceGP.likelihood.variance.value)

        return None 
        
    def __square_dist(self, X1, X2):
        X1 = X1 / self.ls 
        X_square = np.sum(np.square(X1), axis = 1) 
        if X2 is None: # If there is no X2 value...
            # Get product of X and X and add the sum of the squares to it
            X_transpose = np.transpose(X1)
            dist = -2 * np.matmul(X1, X_transpose)
            dist += np.reshape(X_square, (-1, 1)) + np.reshape(X_square, (1, -1))
            return dist

        X2 = X2 / self.ls # If there is an X2, divide by ls
        X2_square = np.sum(np.square(X2), axis = 1) 
        dist = -2 * np.matmul(X1, np.transpose(X2)) 
        dist += np.reshape(X_square, (-1, 1)) + np.reshape(X2_square, (1, -1)) 

        return dist

    def __K(self, X1, X2=None):
        """ Kernel Function """
        return self.var * np.exp(-self.__square_dist(X1, X2) / 2)

    def __Kdiag(self, X1):
        """ 
        K-diagonal Function 
        Return: tensor with just the variances along 0th dimension of X 
        """
        return np.full((1, X1.shape[0]), float(self.var))        

class personalizedGP(baseGP):
    def __init__(self, X, Y, kernel, sGP=None, **kwargs):
        super().__init__(X, Y, kernel)

        self.sGP = sGP 

    def train(self, X_tr, Y_tr, X_ad, Y_ad, new_patient=True, **kwargs): 
        """ 
        Trains personalizedGP 

        PARAMETERS
        X_tr: array of training data features 
        Y_tr: array of training data labels 
        X_ad: array of adaptation data features 
        Y_ad: array of adaptation data labels 
        """
        baseGP.train(self, X_tr, Y_tr, self.sGP) # trains sGP if necessary 

        self.X_tr = X_tr if new_patient else np.vstack((self.X_tr, X_tr))
        self.Y_tr = Y_tr if new_patient else np.vstack((self.Y_tr, Y_tr))

        self.X_ad = X_ad if new_patient else np.vstack((self.X_ad, X_ad))
        self.Y_ad = Y_ad if new_patient else np.vstack((self.Y_ad, Y_ad)) 

        self.K_tt_all = self._baseGP__K(self.X_ad)
        
        K_s = self._baseGP__K(self.X_tr)
        L_arg = K_s + self.lik*np.identity(K_s.shape[0]) 
        self.L = jitChol(L_arg) 

        alpha_denom = np.linalg.lstsq(self.L, self.Y_tr, rcond=None)[0]
        self.alpha = np.linalg.lstsq(self.L.transpose(),alpha_denom, rcond=None)[0]

        return None 

    def predict(self, X_te, sGP_predictions=None, v1=1, **kwargs):
        """ 
        Predicts on personalizedGP 
        
        PARAMETERS 
        X_te: array of testing data features 
        sGP_predictions: tuple of sGP mean and variance predictions 
        v1: int indicating visit to start predicting from 
        """ 
        if sGP_predictions is None: 
            m_s, s_s = self.sourceGP.predict_y(X_te)
        else:
            m_s, s_s = sGP_predictions

        m_adapt, s_adapt = None, None
        
        K_ts_star_all = self._baseGP__K(self.X_tr, X_te)
        K_t_star_all = self._baseGP__K(self.X_ad, X_te)

        start = v1 if v1 == 1 else v1-1 
        for i in range(start, len(self.X_ad)+1): 

            if i == 1:
                m_adapt = m_s[0:1, :]
                s_adapt = s_s[0:1, 0:1]
            
            y_a_patient = self.Y_ad[0:i] # adaptation data for subject

            # K_ts, K_tt
            K_ts = K_ts_star_all[:,:i]
            K_tt = self.K_tt_all[:i,:i]

            # alpha_adapt
            V = np.linalg.lstsq(self.L, K_ts, rcond=None)[0]
            mu_t = mu(K_ts.transpose(), self.alpha)
            C_t = K_tt - np.dot(V.transpose(), V) + self.lik*np.identity(K_tt.shape[0])
            L_adapt = jitChol(C_t)
            alpha_adapt = linalg.cho_solve((L_adapt, True), y_a_patient - mu_t)

            # V_adapt
            K_t_star = K_t_star_all[:i, i:i+1]
            K_ts_star = K_ts_star_all[:, i:i+1]
            V_star = np.linalg.lstsq(self.L, K_ts_star, rcond=None)[0]
            V_dot = np.dot(V_star.transpose(), V)
            C_t_star = K_t_star - V_dot.transpose() 
            V_adapt = np.linalg.lstsq(L_adapt, C_t_star, rcond=None)[0]

            add_adapt = np.dot(C_t_star.transpose(), alpha_adapt)
            m_adapt = np.vstack((m_adapt, m_s[i:i+1, :] + add_adapt))
            
            s_adapt_ele = sigma(s_s[i], V_adapt)
            s_adapt = np.vstack((s_adapt, s_adapt_ele[0:1]))
        
        return m_adapt, s_adapt 

class targetGP(baseGP):
    def __init__(self, X, Y, kernel, sGP=None, **kwargs):
        super().__init__(X, Y, kernel)

        self.sGP = sGP 

    def train(self, X_tr, Y_tr, X_t, Y_t, new_patient=True, **kwargs): 
        """ 
        Trains targetGP  

        PARAMETERS
        X_tr: array of training data features 
        Y_tr: array of training data labels 
        X_t: array of target data features 
        Y_t: array of target data labels 
        """
        baseGP.train(self, X_tr, Y_tr, self.sGP) # trains sGP if necessary 

        self.X_tr = X_tr if new_patient else np.vstack((self.X_tr, X_tr))
        self.Y_tr = Y_tr if new_patient else np.vstack((self.Y_tr, Y_tr))

        self.X_t = X_t if new_patient else np.vstack((self.X_t, X_t))
        self.Y_t = Y_t if new_patient else np.vstack((self.Y_t, Y_t)) 

        self.K_s_all = self._baseGP__K(self.X_t) 

        return None 

    def predict(self, X_te, sGP_predictions=None, v1=1, **kwargs):
        """ 
        Predicts on targetGP  
        
        PARAMETERS 
        X_te: array of testing data features 
        sGP_predictions: tuple of sGP mean and variance predictions 
        v1: int indicating visit to start predicting from 
        """ 
        if sGP_predictions is None: 
            m_s, s_s = self.sourceGP.predict_y(X_te)
        else:
            m_s, s_s = sGP_predictions

        m_target, s_target = None, None

        K_ts_star_all = self._baseGP__K(self.X_t, X_te) 
        k_star_star_all = self._baseGP__Kdiag(X_te) 

        start = v1 if v1 == 1 else v1-1 
        for i in range(start, len(self.X_t) + 1): 

            if i == 1:
                m_target = m_s[0:1, :]
                s_target = s_s[0:1, 0:1]
            
            y_a_patient = self.Y_t[0:i] # target data for subject

            # Calculation of target mean and variance 
            # K_ts_star 
            K_ts_star = K_ts_star_all[:i, i:i+1]

            # k_star_star 
            k_star_star = np.array([k_star_star_all[0][i]])

            # V_star
            K_s = self.K_s_all[:i, :i]
            L_arg = K_s + self.lik*np.identity(K_s.shape[0])
            L = jitChol(L_arg)
            V_star = np.linalg.lstsq(L, K_ts_star, rcond=None)[0]

            alpha_denom = np.linalg.lstsq(L, y_a_patient, rcond=None)[0]
            alpha = np.linalg.lstsq(L.transpose(), alpha_denom, rcond=None)[0]
            m_target_ele = mu(K_ts_star.transpose(), alpha)

            s_target_ele = sigma(k_star_star, V_star)

            m_target = np.vstack((m_target, m_target_ele))
            s_target = np.vstack((s_target, s_target_ele[0]))

        return m_target, s_target

if __name__ == '__main__':
    pass