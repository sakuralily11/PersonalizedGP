

###########################################
## Metric Calculation - Helper Functions ##
###########################################

import numpy as np
import math

def calcMAE(trueLabels, predictionLabels):
	"""
	Parameters
	----------
	trueLabels - contains ground truth values.
	predictionLabels - contains predicted values.

	Returns
	----------
	maeResult - Mean Absolute Error

	"""
	trueLabels = np.asarray(trueLabels)
	predictionLabels = np.asarray(predictionLabels)
	diff = np.subtract(trueLabels, predictionLabels)
	abs_diff = np.fabs(diff)

	maeResult = sum(abs_diff)/len(predictionLabels)

	return maeResult 

def calcWES(trueLabels, predictionLabels, predictionVariance):
	"""
	Parameters
	----------
	trueLabels - contains ground truth values.
	predictionLabels - contains predicted values.
	predictionVariance - contains predicted variance values.

	Returns
	----------
	wesResult - Weighted Error Score

	"""

	# Define Functions
	root = lambda t: math.sqrt(t)
	rootFunc = np.vectorize(root)

	mult = lambda s: 0.674 * s
	multFunc = np.vectorize(mult)

	wesResult = []

	for i in range(trueLabels.shape[1]):
		trueLabelsSub = trueLabels[:, i]
		predictionLabelsSub = predictionLabels[:, i]
		predictionVarianceSub = predictionVariance[:, 0]

		# Compute lower and upper CI 
		predictLowerCI = predictionLabelsSub - multFunc(rootFunc(predictionVarianceSub))
		predictUpperCI = predictionLabelsSub + multFunc(rootFunc(predictionVarianceSub))

		numEntries = trueLabelsSub.shape[0]
		predictLabelsAll = -1 * np.ones(numEntries , float)
		predictLowerCIAll = -1 * np.ones(numEntries , float)  # lower margin
		predictUpperCIAll = -1 * np.ones(numEntries , float)  # upper margin

		#print ('\n\predictLabels type ', type(predictionLabels))
		#print ('\n\predictLabels size ', len(predictionLabels))
		#print ('\n\predictLabels 0,  = ', predictionLabels[0],predictionLabels[1] )

		for s in range(numEntries):
		        predictLabelsAll[s] = predictionLabelsSub[s]
		        predictLowerCIAll[s] = predictLowerCI[s]
		        predictUpperCIAll[s] = predictUpperCI[s]

		calcCoeffs = 1.0/(predictUpperCIAll - predictLowerCIAll)
		wesResultSub = np.sum(calcCoeffs * np.abs(predictLabelsAll - trueLabelsSub))/np.sum(calcCoeffs)
		wesResult.append(wesResultSub)

	return wesResult


