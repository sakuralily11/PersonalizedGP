# Generates demo data 

from ts_formatter import *

import os 
import numpy as np 
np.random.seed(12345) 

patient_IDs = [11,13,14,20,24,30,37,42,45,77]

data_array = None
visits = 40 
ts = 4

for ID in patient_IDs: 
    # Generate sample data
    X1 = np.sort(5 * np.random.rand(visits, 1), axis=0)
    X2 = np.sort(5 * np.random.rand(visits, 1), axis=0)
    y = (np.sin(X1) + X2).ravel()
    y[::5] += 3 * (0.5 - np.random.rand(8))
    y = np.reshape(y, (visits, 1))
    true_inds = np.arange(visits)
    np.random.shuffle(true_inds)
    true_inds = true_inds[:35]
    indicators = np.zeros((visits, 1))
    indicators[true_inds, :] = 1

    # Assemble and save data 
    patient_data = np.hstack((np.full((visits, 1), ID), X1, X2, y, indicators))
    data_array = patient_data if data_array is None else np.vstack((data_array, patient_data))

# Save data 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV_DIR = os.path.join(CURRENT_DIR, 'demo_data.csv')
np.savetxt(DATA_CSV_DIR, data_array, delimiter=',')

# Format data and save new data 
TS_DATA_CSV_DIR = os.path.join(CURRENT_DIR, 'ts_demo_data.csv')
ts_formatter(DATA_CSV_DIR, TS_DATA_CSV_DIR, ts)