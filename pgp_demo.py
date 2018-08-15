# Personalized Gaussian Process - DEMO 

import os 
import pathlib
import gpflow 
import numpy as np

from GP import *
from evaluation_metrics import *

if __name__ == '__main__':
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_CSV_DIR = os.path.join(CURRENT_DIR, 'data/ts_demo_data.csv')

    data = np.genfromtxt(DATA_CSV_DIR, delimiter=',')

    IDs = data[:, :1]
    X = data[:, 1:-8]
    Y = data[:, -8:-4]
    indicators = data[:, -4:]

    unique_IDs = np.unique(list(map(lambda x:int(x[0]), IDs)))

    # Loop for 10 folds: 
    mean = dict(zip(['sGP', 'pGP', 'tGP', 'joint'], [None, None, None, None]))
    variance = dict(zip(['sGP', 'pGP', 'tGP', 'joint'], [None, None, None, None]))
    for i in range(10):
        te_IDs = [unique_IDs[i]]
        tr_IDs = np.setdiff1d(unique_IDs, te_IDs)

        X_tr, Y_tr, X_te, Y_te, ind_te, ID_te = process_data(X, Y, indicators, IDs, tr_IDs, te_IDs)

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
        pGP = personalizedGP(X=X, Y=Y, kernel=k)
        tGP = targetGP(X=X, Y=Y, kernel=k)
        # Train and predict 
        m_ad, s_ad = None, None 
        m_t, s_t = None, None 
        for te in te_IDs: 
            te_rows = np.where(ID_te == te)[0] 

            X_ad_patient = X_te[te_rows][:-1, :]
            Y_ad_patient = Y_te[te_rows][:-1, :]
            X_te_patient = X_te[te_rows]

            sGP_patient_predictions = sGP.predict_y(X_te_patient)

            pGP.train(X_tr=X_tr, Y_tr=Y_tr, X_ad=X_ad_patient, Y_ad=Y_ad_patient, new_patient=True)
            tGP.train(X_tr=X_tr, Y_tr=Y_tr, X_t=X_ad_patient, Y_t=Y_ad_patient, new_patient=True)

            m_ad_patient, s_ad_patient = pGP.predict(X_te=X_te_patient, sGP_predictions=sGP_patient_predictions, v1=1)
            m_t_patient, s_t_patient = tGP.predict(X_te=X_te_patient, sGP_predictions=sGP_patient_predictions, v1=1)

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

        variance['sGP'] = s_s[:,0:1] if variance['sGP'] is None else np.vstack((variance['sGP'], s_s[:,0:1]))
        variance['pGP'] = s_ad if variance['pGP'] is None else np.vstack((variance['pGP'], s_ad))
        variance['tGP'] = s_t if variance['tGP'] is None else np.vstack((variance['tGP'], s_t))
        variance['joint'] = s_j if variance['joint'] is None else np.vstack((variance['joint'], s_j))

    # Save results 
    RESULTS_FOLDER_DIR = os.path.join(CURRENT_DIR, 'results')
    pathlib.Path(RESULTS_FOLDER_DIR).mkdir(parents=True, exist_ok=True)
    RESULTS_CSV_DIR = os.path.join(RESULTS_FOLDER_DIR, 'results.csv')
    MAE_METRICS_CSV_DIR = os.path.join(RESULTS_FOLDER_DIR, 'mae_metrics.csv')
    WES_METRICS_CSV_DIR = os.path.join(RESULTS_FOLDER_DIR, 'wes_metrics.csv')
    save_results(mean, variance, Y, IDs, indicators, RESULTS_CSV_DIR, MAE_METRICS_CSV_DIR, WES_METRICS_CSV_DIR)