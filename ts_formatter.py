# Formats data to predict XX time steps ahead 

import numpy as np 

def ts_formatter(DATA_DIR, DATA_TS_DIR, t):
    """
    Formats data to predict t time steps ahead 

    PARAMETERS 
    DATA_DIR: string of data directory 
    DATA_TS_DIR: string of formatted data directory 
    t: int of time steps ahead to predict 
    """
    data = np.genfromtxt(DATA_DIR, delimiter=',')

    IDs = data[:, 0:1]
    unique_IDs = np.unique(IDs)
    X = data[:, 1:-2]
    Y = data[:, -2:-1]
    indicators = data[:, -1:]

    data_ts = None 
    for ID in unique_IDs:
        patient_rows = np.where(data[:, 0:1] == ID)[0]

        visits = len(patient_rows)

        patient_IDs = IDs[patient_rows, :]
        patient_X = X[patient_rows, :]
        patient_Y = Y[patient_rows, :]
        patient_indicators = indicators[patient_rows, :]

        # Repeat last value t-1 times 
        patient_Y_extended = np.vstack((patient_Y, np.full((t-1, 1), patient_Y[-1, 0])))
        patient_indicators_extended = np.vstack((patient_indicators, np.full((t-1, 1), patient_indicators[-1, 0])))

        patient_Y_ts = None 
        patient_indicators_ts = None 
        for ts in range(t):
            patient_Y_ts = patient_Y_extended[ts+1:visits+ts, :] if patient_Y_ts is None else np.hstack((patient_Y_ts, patient_Y_extended[ts+1:visits+ts, :]))
            patient_indicators_ts = patient_indicators_extended[ts+1:visits+ts, :] if patient_indicators_ts is None else np.hstack((patient_indicators_ts, patient_indicators_extended[ts+1:visits+ts, :]))
            
        patient_data_ts = np.hstack((patient_IDs[:-1, :], patient_X[:-1, :], patient_Y[:-1, :], patient_Y_ts, patient_indicators_ts))

        data_ts = patient_data_ts if data_ts is None else np.vstack((data_ts, patient_data_ts))

    np.savetxt(DATA_TS_DIR, data_ts, delimiter=',')