# Personalized Gaussian Process - DEMO 

import os 
import gpflow 
import numpy as np

from GP import *
from evaluation_metrics import *

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV_DIR = os.path.join(CURRENT_DIR, 'ts_demo_data.csv')
# DATA_CSV_DIR = os.path.join(CURRENT_DIR, 'adni_adas13_100_fl1_l4.csv')

data = np.genfromtxt(DATA_CSV_DIR, delimiter=',')

IDs = data[:, :1]
X = data[:, 1:-8]
Y = data[:, -8:-4]
indicators = data[:, -4:]

unique_IDs = np.unique(list(map(lambda x:int(x[0]), IDs)))

# Loop for 10 folds: 
mean = {'sGP': None, 'pGP': None, 'tGP': None, 'joint': None} 
variance = {'sGP': None, 'pGP': None, 'tGP': None, 'joint': None}
for i in range(10):
    te_IDs = [unique_IDs[i]]
    # te_IDs = IDs[i*10:i*10+10]
    tr_IDs = np.setdiff1d(unique_IDs, te_IDs)

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

    ## Source GP 
    # Create RBF kernel and GP model instance 
    k = gpflow.kernels.RBF(input_dim=X_tr.shape[1])
    sGP = gpflow.models.GPR(X_tr, Y_tr, kern = k)
    # Initialize hyperparameters 
    sGP.likelihood.variance = np.exp(2*np.log(np.sqrt(0.1*np.var(Y_tr))))
    sGP.kern.lengthscales = np.array([np.median(np.amax(X_tr, axis=0)-np.amin(X_tr, axis=0))])[0]
    sGP.kern.variance = np.var(Y_tr)
    # Optimize model 
    sGP.compile()
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(sGP, maxiter=30)
    # Predict 
    m_s, s_s = sGP.predict_y(X_te)

    ## Personalized GP & Target GP 
    # Create personalizedGP & targetGP instance 
    pGP = personalizedGP(X_tr=X_tr, Y_tr=Y_tr, kernel=k, GP=sGP)
    tGP = targetGP(X_tr=X_tr, Y_tr=Y_tr, kernel=k, GP=sGP)
    # Train and predict  
    m_ad, s_ad = None, None 
    m_t, s_t = None, None 
    for te in te_IDs: 
        te_rows = np.where(ID_te == te)[0] 

        X_ad_patient = X_te[te_rows][:-1, :]
        Y_ad_patient = Y_te[te_rows][:-1, :]
        X_te_patient = X_te[te_rows]

        sGP_patient_predictions = sGP.predict_y(X_te_patient)

        m_ad_patient, s_ad_patient = pGP.predict(X_ad=X_ad_patient, Y_ad=Y_ad_patient, X_te=X_te_patient, sGP_predictions=sGP_patient_predictions)
        m_t_patient, s_t_patient = tGP.predict(X_t=X_ad_patient, Y_t=Y_ad_patient, X_te=X_te_patient, sGP_predictions=sGP_patient_predictions)

        m_ad = m_ad_patient if m_ad is None else np.vstack((m_ad, m_ad_patient))
        s_ad = s_ad_patient if s_ad is None else np.vstack((s_ad, s_ad_patient))

        m_t = m_t_patient if m_t is None else np.vstack((m_t, m_t_patient))
        s_t = s_t_patient if s_t is None else np.vstack((s_t, s_t_patient))

    ## Joint GP 
    m_j = (m_ad + m_t)/2
    s_j = (s_ad + s_t)/4

    mean['sGP'] = m_s if mean['sGP'] is None else np.vstack((mean['sGP'], m_s))
    mean['pGP'] = m_ad if mean['pGP'] is None else np.vstack((mean['pGP'], m_ad))
    mean['tGP'] = m_t if mean['tGP'] is None else np.vstack((mean['tGP'], m_t))
    mean['joint'] = m_j if mean['joint'] is None else np.vstack((mean['joint'], m_j))

    variance['sGP'] = s_s if variance['sGP'] is None else np.vstack((variance['sGP'], s_s))
    variance['pGP'] = s_ad if variance['pGP'] is None else np.vstack((variance['pGP'], s_ad))
    variance['tGP'] = s_t if variance['tGP'] is None else np.vstack((variance['tGP'], s_t))
    variance['joint'] = s_j if variance['joint'] is None else np.vstack((variance['joint'], s_j))

print('sGP:', np.mean(np.abs(mean['sGP'] - Y), axis=0))
print('pGP:', np.mean(np.abs(mean['pGP'] - Y), axis=0))
print('tGP:', np.mean(np.abs(mean['tGP'] - Y), axis=0))
print('jointGP:', np.mean(np.abs(mean['joint'] - Y), axis=0))

# Save results 
RESULTS_FOLDER_DIR = os.path.join(CURRENT_DIR, 'Results')