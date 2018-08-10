

##########################################
## Personalized Gaussian Process - DEMO ## 
##########################################

import numpy as np
import gpflow 

from call_pgp import *
from evaluation_metrics import *

X = np.random.uniform(low = 0.0, high = 1.0, size = (1000, 10))
Y = np.random.uniform(low = 5.0, high = 10.0, size = (1000, 3))

tr_ind_source = list(range(0, 800))
adapt_ind = list(range(800, 1000))

#extract source features and labels 
#step may vary if dealing with multiple patient data 
x_s = X[[tr_ind_source]]
y_s = Y[[tr_ind_source]]

#extract all adaptation data 
x_a = X[[adapt_ind]][:-1, :]
y_a = Y[[adapt_ind]][:-1, :]

#extract test data
xtest = X[[adapt_ind]]
ytest = Y[[adapt_ind]]

#creating RBF kernel 
d = np.shape(X)[1]
k = gpflow.kernels.RBF(d)

#TRAINING AND PREDICTING SOURCE MODEL 
#creating gp model m with x_s, y_s variables and kernel k 
m = gpflow.models.GPR(x_s, y_s, kern = k)
#initialize hyperparameters 
m.likelihood.variance = np.exp(2*np.log(np.sqrt(0.1*np.var(y_s))))
max_x = np.amax(x_s, axis=0)
min_x = np.amin(x_s, axis=0)
m.kern.lengthscales = np.array(np.median(max_x - min_x))
m.kern.variance = np.var(y_s)
#optimizing model 
m.compile()
opt = gpflow.train.ScipyOptimizer()
opt.minimize(m)

############################################################################
#Stores output, specify key to retrieve values, i.e. out['source model mu']#
############################################################################

#out.keys() = dict_keys(['source model mu', 'source model sigma', 'adapted model mu', 'adapted model sigma', 'target model mu', 'target model sigma', 'param - kernel ls', 'param - kernel var', 'param - likelihood var'])

out = call_pgp(m, x_s, y_s, x_a, y_a, xtest, k)

g_t = ytest

m_s = out['source model mu']
m_a = out['adapted model mu']
m_t = out['target model mu']
m_j = out['joint model mu']

s_s = out['source model sigma']
s_a = out['adapted model sigma']
s_t = out['target model sigma']
s_j = out['joint model sigma']

#error - source model 
e_s = calcMAE(g_t, m_s)
w_s = calcWES(g_t, m_s, s_s)
print('SOURCE MODEL - MEAN ABSOLUTE ERROR:', e_s)
print('SOURCE MODEL - WEIGHTED ERROR SCORE:', w_s)

#error - adaptation model 
e_a = calcMAE(g_t, m_a)
w_a = calcWES(g_t, m_a, s_a)
print('ADAPTATION MODEL - MEAN ABSOLUTE ERROR:', e_a)
print('ADAPTATION MODEL - WEIGHTED ERROR SCORE:', w_a)

#error - target model 
e_t = calcMAE(g_t, m_t)
w_t = calcWES(g_t, m_t, s_t)
print('TARGET MODEL - MEAN ABSOLUTE ERROR:', e_t)
print('TARGET MODEL - WEIGHTED ERROR SCORE:', w_t)

#error - joint model 
e_j = calcMAE(g_t, m_j)
w_j = calcWES(g_t, m_j, s_j)
print('JOINT MODEL - MEAN ABSOLUTE ERROR:', e_j)
print('JOINT MODEL - WEIGHTED ERROR SCORE:', w_j)
