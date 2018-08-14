# Metric Calculation Helper Functions 

import numpy as np
import pandas as pd 
import math

def mae(y, y_hat):
	"""
	PARAMETERS
	y: array of ground truth values
	y_hat: array of predicted values

	RETURN
	mae_result: float of mean absolute error 
	"""
	y = np.asarray(y)
	y_hat = np.asarray(y_hat)
	diff = np.subtract(y, y_hat)
	abs_diff = np.fabs(diff)

	mae_result = sum(abs_diff)/len(y_hat)

	return mae_result 

def wes(y, y_hat, s_hat):
	"""
	PARAMETERS
	y: array of ground truth values
	y_hat: array of predicted values
	s_hat: array of predicted variances 

	RETURN 
	wes_result: float of weighted error score 
	"""

	# Define Functions
	root = lambda t: math.sqrt(t)
	root_func = np.vectorize(root)

	mult = lambda s: 0.674 * s
	mult_func = np.vectorize(mult)

	wes_result = []

	for i in range(y.shape[1]):
		y_sub = y[:, i]
		y_hat_sub = y_hat[:, i]
		s_hat_sub = s_hat[:, 0]

		# Compute lower and upper CI 
		lower_CI = y_hat_sub - mult_func(root_func(s_hat_sub))
		upper_CI = y_hat_sub + mult_func(root_func(s_hat_sub))

		num_entries = y_sub.shape[0]
		labels_all = -1 * np.ones(num_entries, float)
		lower_CI_all = -1 * np.ones(num_entries, float)  # lower margin
		upper_CI_all = -1 * np.ones(num_entries, float)  # upper margin

		for s in range(num_entries):
		        labels_all[s] = y_hat_sub[s]
		        lower_CI_all[s] = lower_CI[s]
		        upper_CI_all[s] = upper_CI[s]

		calc_coeffs = 1.0/(upper_CI_all - lower_CI_all)
		wes_results_sub = np.sum(calc_coeffs * np.abs(labels_all - y_sub))/np.sum(calc_coeffs)
		wes_result.append(wes_results_sub)

	return wes_result

def save_results(mean, variance, y, IDs, indicators, RESULTS_CSV_DIR, MAE_METRICS_CSV_DIR, WES_METRICS_CSV_DIR): 
	"""
	Saves results and MAE/WES calculations 

	PARAMETERS 
	mean: dict of predicted mean values 
	variance: dict of predicted variance values 
	y: array of ground truth values 
	IDs: array of ID column 
	indicators: array of indicators 
	RESULTS_CSV_DIR: string of results directory 
	MAE_METRICS_CSV_DIR: string of MAE metrics directory 
	WES_METRICS_CSV_DIR: string of WES metrics directory 
	"""
	keys = list(mean.keys())
	ts = indicators.shape[1]
	unique_IDs = np.unique(IDs)

	assert sorted(keys) == sorted(list(variance.keys()))
	assert ts == np.vstack(tuple(mean.values())).shape[1]

	results = IDs 
	for k in keys:
		results = np.hstack((results, mean[k], variance[k]))
	results = np.hstack((results, indicators))
	key_header = np.repeat(keys, 5)
	add_header = [' - t+1', ' - t+2', ' - t+3', ' - t+4', ' - variance']
	key_header = [x+add_header[i%len(add_header)] for i,x in enumerate(key_header)]
	header = list(np.concatenate((np.array(['ID']), key_header, np.array(['indicator - t+1', 'indicator - t+2', 'indicator - t+3', 'indicator- t+4']))))
	results_df = pd.DataFrame(results, columns=header)

	per_patient_mae = dict(zip(keys, [None]*len(keys)))
	per_patient_wes = dict(zip(keys, [None]*len(keys)))
	for t in range(ts):
		true_ts_indicators = np.where(indicators[:, t:t+1] == 1)[0]

		ts_mae = dict(zip(keys, [None]*len(keys)))
		ts_wes = dict(zip(keys, [None]*len(keys)))
		for i in unique_IDs: 
			patient_rows = np.where(IDs == i)[0]
			true_patient_rows = np.intersect1d(true_ts_indicators, patient_rows)
			patient_y = y[true_patient_rows, t:t+1]

			for k in keys: 
				k_patient_mean = mean[k][true_patient_rows, t:t+1]
				k_patient_var = variance[k][true_patient_rows, :]

				k_patient_mae = mae(patient_y, k_patient_mean)
				k_patient_wes = wes(patient_y, k_patient_mean, k_patient_var)

				ts_mae[k] = k_patient_mae if ts_mae[k] is None else np.vstack((ts_mae[k], k_patient_mae))
				ts_wes[k] = k_patient_wes if ts_wes[k] is None else np.vstack((ts_wes[k], k_patient_wes))

		for k in keys:
			per_patient_mae[k] = ts_mae[k] if per_patient_mae[k] is None else np.hstack((per_patient_mae[k], ts_mae[k]))
			per_patient_wes[k] = ts_wes[k] if per_patient_wes[k] is None else np.hstack((per_patient_wes[k], ts_wes[k]))

	mae_avg, wes_avg = None, None
	mae_std, wes_std = None, None
	for k in keys: 
		mae_avg = np.average(per_patient_mae[k], axis=0) if mae_avg is None else np.vstack((mae_avg, np.average(per_patient_mae[k], axis=0)))
		wes_avg = np.average(per_patient_wes[k], axis=0) if wes_avg is None else np.vstack((wes_avg, np.average(per_patient_wes[k], axis=0)))
		mae_std = np.std(per_patient_mae[k], axis=0) if mae_std is None else np.vstack((mae_std, np.std(per_patient_mae[k], axis=0)))
		wes_std = np.std(per_patient_wes[k], axis=0) if wes_std is None else np.vstack((wes_std, np.std(per_patient_wes[k], axis=0)))

	mae_metrics = np.full((len(keys), mae_avg.shape[1]*2), np.nan)
	wes_metrics = np.full((len(keys), wes_avg.shape[1]*2), np.nan)
	mae_metrics[:, ::2], wes_metrics[:, ::2] = mae_avg, wes_avg
	mae_metrics[:, 1::2], wes_metrics[:, 1::2] = mae_std, wes_std

	ts_header = ['t+1', '+/- std', 't+2', '+/- std', 't+3', '+/- std', 't+4', '+/- std']
	mae_metrics_df = pd.DataFrame(mae_metrics, index=keys, columns=ts_header)
	wes_metrics_df = pd.DataFrame(wes_metrics, index=keys, columns=ts_header)

	results_df.to_csv(RESULTS_CSV_DIR, index=False, header=True, sep=',')
	mae_metrics_df.to_csv(MAE_METRICS_CSV_DIR, index=True, header=True, sep=',')
	wes_metrics_df.to_csv(WES_METRICS_CSV_DIR, index=True, header=True, sep=',')
	
if __name__ == '__main__':
    pass