#--------------------------------------------------
# UCLA CS 260 - Machine Learning Algorithm
# Yao-Jen Chang, 2015 @UCLA
# Email: autekwing@ucla.edu
#
# The main function to evaluate result for different algorithm and parameter
#--------------------------------------------------

import sys
import copy
import matplotlib.pyplot as plt
from PlotData import *
from ReadFile import *
from KNN import *
from Perceptron import *
import MLPbatch

# ========= calculated different statistical features ==========
# Statical method: none, weight1, mean, addMean, miusMean, addMean_miusMean, rms, std, median, mean_median, mean_std, mean_rms

# none: don't do anything. 52 features
# weight1: add weights into 2 time series data. The weights are ([1,2,3... ,number of time series data])/sum of previous list. 
# mean:Mean of 2 time series data. Feature number: 2
# addMean:Sum of 2 means time series data. Feature number: 1
# miusMean:Absolute value of subtraction of 2 means time series data. Feature number: 1
# addMean_miusMean: method "addMean" and method "miusMean". Feature number: 2
# rms: Root mean square of time series data. Feature number: 2
# std: Standard deviation of time series data. Feature number: 2
# median: Median of time series data. Feature number: 2
# mean_median: Mean and median of time series data (Method 3 and method 9). Feature number: 4
# mean_std: Mean and standard deviation of data (Method 3 and method 8). Feature number: 4
# mean_rms: Mean and root mean square of data (Method 3 and method 7). Feature number: 4
def calStatic(trainFea, testFea, static):
	from numpy import mean
	from numpy import std
	from numpy import median
	from numpy import sqrt, square

	test = []
	train = []

	if static == 'none':
		for x in testFea:
			for y in x:
				test.append(y)
		for t in trainFea:
			tmp = []
			for x in t:
				for y in x:
					tmp.append(y)
			train.append(tmp)
	# weight
	elif static == 'weight1':
		weights = range(1, len(testFea[0]) + 1)
		tmpSum = sum(weights)
		weights = [x / float(tmpSum) for x in weights]

		test.append(weightedMovingAver(testFea[0], weights))
		test.append(weightedMovingAver(testFea[1], weights))

		for t in trainFea:
			tmp = []
			tmp.append(weightedMovingAver(t[0], weights))
			tmp.append(weightedMovingAver(t[1], weights))
			train.append(tmp)
	# mean
	elif static == 'mean':
		test.append(mean(testFea[0]))
		test.append(mean(testFea[1]))

		for t in trainFea:
			tmp = []
			tmp.append(mean(t[0]))
			tmp.append(mean(t[1]))
			train.append(tmp)

	# add the mean value of both blood pressure
	elif static == 'addMean':
		test.append( mean(testFea[0]) + mean(testFea[1]))
		for t in trainFea:
			train.append(mean(t[0]) + mean(t[1]))

	# substract the mean value of both blood pressure
	elif static == 'miusMean':
		test.append( mean(testFea[1]) - mean(testFea[0]))
		for t in trainFea:
			train.append(mean(t[1]) - mean(t[0]))

	# substract the mean value of both blood pressure
	elif static == 'addMean_miusMean':
		test.append( mean(testFea[0]) + mean(testFea[1]))
		test.append( mean(testFea[1]) - mean(testFea[0]))
		for t in trainFea:
			tmp = []
			tmp.append(mean(t[0]) + mean(t[1]))
			tmp.append(mean(t[1]) - mean(t[0]))
			train.append(tmp)

	# standard deviation 
	elif static == 'std':
		test.append(std(testFea[0]))
		test.append(std(testFea[1]))

		for t in trainFea:
			tmp = []
			tmp.append(std(t[0]))
			tmp.append(std(t[1]))
			train.append(tmp)
	
	# Root mean square
	elif static == 'rms':
		test.append(sqrt(mean(square(testFea[0]))))
		test.append(sqrt(mean(square(testFea[1]))))

		for t in trainFea:
			tmp = []
			tmp.append(sqrt(mean(square(t[0]))))
			tmp.append(sqrt(mean(square(t[1]))))
			train.append(tmp)

	# pick the median
	elif static == 'median':
		test.append(median(testFea[0]))
		test.append(median(testFea[1]))

		for t in trainFea:
			tmp = []
			tmp.append(median(t[0]))
			tmp.append(median(t[1]))
			train.append(tmp)

	# mean and median
	elif static == 'mean_median':
		test.append(mean(testFea[0]))
		test.append(mean(testFea[1]))
		test.append(median(testFea[0]))
		test.append(median(testFea[1]))

		for t in trainFea:
			tmp = []
			tmp.append(mean(t[0]))
			tmp.append(mean(t[1]))
			tmp.append(median(t[0]))
			tmp.append(median(t[1]))
			train.append(tmp)

	# mean and standard deviation
	elif static == 'mean_std':
		test.append(mean(testFea[0]))
		test.append(mean(testFea[1]))
		test.append(std(testFea[0]))
		test.append(std(testFea[1]))

		for t in trainFea:
			tmp = []
			tmp.append(mean(t[0]))
			tmp.append(mean(t[1]))
			tmp.append(std(t[0]))
			tmp.append(std(t[1]))
			train.append(tmp)

	# mean and Root mean square
	elif static == 'mean_rms':
		test.append(mean(testFea[0]))
		test.append(mean(testFea[1]))
		test.append(sqrt(mean(square(testFea[0]))))
		test.append(sqrt(mean(square(testFea[1]))))

		for t in trainFea:
			tmp = []
			tmp.append(mean(t[0]))
			tmp.append(mean(t[1]))
			tmp.append(sqrt(mean(square(t[0]))))
			tmp.append(sqrt(mean(square(t[1]))))
			train.append(tmp)

	else:
		print "No such static method"
	
	return test, train


# ========= calcuate the weighted moving average for function "calStatic" =========
def weightedMovingAver(data, weights):
	ans = 0
	size = len(data)
	for i in range(size):
		ans += (data[i] * weights[i])

	return ans


# ========= Leave-one-out cross-validation  =========== 
# Can run method "none" for model "knn", but not for model 'per' or 'mlp'
def  LOOCV(feature, label, model, method, parameter = []):
	ConfuMat = [] # confusion matrix
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	size = len(label)

	# Leave-one-out cross-validation 
	for i in range(size):
		trainFea = copy.deepcopy(feature) # copy from original list
		testFea = trainFea.pop(i)

		trainLab = copy.deepcopy(label)
		testLab = trainLab.pop(i)

		if method == 'none':
			test, train = calStatic(trainFea, testFea, 'none')
		elif method == 'weight1':
			test, train = calStatic(trainFea, testFea, 'weight1')
		elif method == 'mean':
			test, train = calStatic(trainFea, testFea, 'mean')
		elif method == 'addMean':
			test, train = calStatic(trainFea, testFea, 'addMean')
		elif method == 'miusMean':
			test, train = calStatic(trainFea, testFea, 'miusMean')
		elif method == 'addMean_miusMean':
			test, train = calStatic(trainFea, testFea, 'addMean_miusMean')
		elif method == 'rms':
			test, train = calStatic(trainFea, testFea, 'rms')			
		elif method == 'std':
			test, train = calStatic(trainFea, testFea, 'std')
		elif method == 'median':
			test, train = calStatic(trainFea, testFea, 'median')
		elif method == 'mean_median':
			test, train = calStatic(trainFea, testFea, 'mean_median')			
		elif method == 'mean_std':
			test, train = calStatic(trainFea, testFea, 'mean_std')
		elif method == 'mean_rms':
			test, train = calStatic(trainFea, testFea, 'mean_rms')

		if model == 'knn': # parameter is k value
			pred = knn(train, trainLab, test, parameter)
		elif model == 'per': # parameter is weight based-random number
			normTrain = normalize(train)
			normTest = normalize(test)
			# print train
			# print test
			pred = perceptron(normTrain, trainLab, normTest, parameter)
		elif model == 'MLP':
			normTrain = normalize(train)
			normTest = normalize(test)
			# print train
			# print test
			pred = MLP(normTrain, trainLab, normTest, parameter)
		elif model == 'MLPbatch':
			nIter      = parameter[0]
			learnRate  = parameter[1]
			activation = parameter[2]
			hiddenNode = parameter[3]
			outputtype = parameter[4]

			normTrain = normalize(train)
			normTest = normalize(test)

			trainLabArr = np.asarray(trainLab).reshape(len(trainLab),1)
			normTrainArr = np.asarray(normTrain).reshape(len(normTrain), len(normTrain[0]))
			normTestArr = np.asarray(normTest).reshape(1, len(normTest))

			p = MLPbatch.MLPbatch(normTrainArr, trainLabArr, hiddenNode, outtype = outputtype)
			p.mlptrain(normTrainArr, trainLabArr, learnRate, nIter)
			pred = p.predict(normTestArr, activation)		
			pred = pred[0][0] # extract from array
		else:
			print "illegal learning method. Please try knn, per, MLP, MLPbatch\n"
			return

		# print pred

		if testLab == 1 and pred == 1:
			TP += 1
		elif testLab == 1 and pred == 0:
			FP += 1
		elif testLab == 0 and pred == 0:
			TN += 1
		elif testLab == 0 and pred == 1:
			FN += 1

		# print testLab, pred

	ConfuMat.append(TP) # correctly classified
	ConfuMat.append(FN)
	ConfuMat.append(FP)
	ConfuMat.append(TN) # correctly classified

	accuracy = (TP + TN) / float(TP + FP + TN + FN)
	accuMatrix, Fmeasure, TPR, FPR = accuracyMatrix(TP, TN, FP, FN)

	tmpstr = ""
	for a in accuMatrix:
		tmpstr += ("%.3f" % a + "\t")

	print str(accuracy) +"  "+ str(ConfuMat) + "  " + str(tmpstr) + "%.3f"  % Fmeasure 
	return accuMatrix, ConfuMat, accuracy, Fmeasure, TPR, FPR 

# ========= Calculate accuracy Matrix ==========
# Parameter: TP, TN, FP, FN
# Retuen: accuracy Matrix(precission, recall, sensitivity, specificity), Fmeasure, TPR, FPR
def accuracyMatrix(TP, TN, FP, FN):
	try:
		precission  = TP / float(TP + FP)
	except ZeroDivisionError:
		precission = 0

	try:
		recall      = TP / float(TP + FN)
	except ZeroDivisionError:
		recall = 0

	try:
		sensitivity = TP / float(TP + FN)
	except ZeroDivisionError:
		sensitivity = 0

	try:
		specificity = TN / float(TN + FP)
	except ZeroDivisionError:
		specificity = 0

	mat = []
	mat.append(precission)
	mat.append(recall)
	mat.append(sensitivity)
	mat.append(specificity)
	
	try:
		Fmeasure = 2.0 * (precission * recall) / float(precission + recall)
	except ZeroDivisionError:
		Fmeasure = 0
	try:
		FPR = FP / float(FP + TN) #false positive rate 
	except ZeroDivisionError:
		FPR = 0

	TPR = sensitivity         #true positive rate 

	return mat, Fmeasure, TPR, FPR

def fmeasure(mat):
	precission = mat[0]
	recall= mat[1]
	sensitivity= mat[2]
	specificity= mat[3]
	try:
		Fmeasure = 2.0 * (precission * recall) / float(precission + recall)
	except ZeroDivisionError:
		Fmeasure = 0
	tmpstr = ""
	for a in mat:
		tmpstr += ("%.3f" % a + "\t")

	print str(tmpstr) + "%.3f"  % Fmeasure 

	return Fmeasure	

# ========= Print accuracy matrix ==========
def printComfusionMatrix(matrix):
	TP = matrix[0] # correctly classified
	FN = matrix[1]
	FP = matrix[2]
	TN = matrix[3] # correctly classified

	print "              Output"
	print "          class 1, class 0"
	print " class 1    " + str(TP) + "        " + str(FP)
	print " class 0    " + str(FN) + "        " + str(TN)


# ========= Use training data to train model and test data =========== 
def testModel(trainFea, trainLab, AlltestFea, AlltestLab, model, method, parameter = []):
	ConfuMat = [] # confusion matrix
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	size = len(AlltestLab)

	# Leave-one-out cross-validation 
	for i in range(size):
		testFea = AlltestFea[i]
		testLab = AlltestLab[i]

		if method == 'none':
			test, train = calStatic(trainFea, testFea, 'none')
		elif method == 'weight1':
			test, train = calStatic(trainFea, testFea, 'weight1')
		elif method == 'mean':
			test, train = calStatic(trainFea, testFea, 'mean')
		elif method == 'addMean':
			test, train = calStatic(trainFea, testFea, 'addMean')
		elif method == 'miusMean':
			test, train = calStatic(trainFea, testFea, 'miusMean')
		elif method == 'addMean_miusMean':
			test, train = calStatic(trainFea, testFea, 'addMean_miusMean')
		elif method == 'rms':
			test, train = calStatic(trainFea, testFea, 'rms')			
		elif method == 'std':
			test, train = calStatic(trainFea, testFea, 'std')
		elif method == 'median':
			test, train = calStatic(trainFea, testFea, 'median')
		elif method == 'mean_median':
			test, train = calStatic(trainFea, testFea, 'mean_median')			
		elif method == 'mean_std':
			test, train = calStatic(trainFea, testFea, 'mean_std')
		elif method == 'mean_rms':
			test, train = calStatic(trainFea, testFea, 'mean_rms')

		if model == 'knn': # parameter is k value
			pred = knn(train, trainLab, test, parameter)
		elif model == 'per': # parameter is weight based-random number
			normTrain = normalize(train)
			normTest = normalize(test)
			# print train
			# print test
			pred = perceptron(normTrain, trainLab, normTest, parameter)
		elif model == 'MLP':
			normTrain = normalize(train)
			normTest = normalize(test)
			# print train
			# print test
			pred = MLP(normTrain, trainLab, normTest, parameter)
		elif model == 'MLPbatch':
			nIter      = parameter[0]
			learnRate  = parameter[1]
			activation = parameter[2]
			hiddenNode = parameter[3]

			normTrain = normalize(train)
			normTest = normalize(test)
			# normTrain = train
			# normTest = test
			trainLabArr = np.asarray(trainLab).reshape(len(trainLab),1)
			normTrainArr = np.asarray(normTrain).reshape(len(normTrain), len(normTrain[0]))
			normTestArr = np.asarray(normTest).reshape(1, len(normTest))

			p = MLPbatch.MLPbatch(normTrainArr, trainLabArr, hiddenNode, outtype = 'logistic')
			p.mlptrain(normTrainArr, trainLabArr, learnRate, nIter)
			pred = p.predict(normTestArr, activation)		
			pred = pred[0][0] # extract from array

		# print pred

		if testLab == 1 and pred == 1:
			TP += 1
		elif testLab == 1 and pred == 0:
			FP += 1
		elif testLab == 0 and pred == 0:
			TN += 1
		elif testLab == 0 and pred == 1:
			FN += 1

		# print testLab, pred

	ConfuMat.append(TP) # correctly classified
	ConfuMat.append(FN)
	ConfuMat.append(FP)
	ConfuMat.append(TN) # correctly classified

	accuracy = (TP + TN) / float(TP + FP + TN + FN)
	accuMatrix, Fmeasure, TPR, FPR = accuracyMatrix(TP, TN, FP, FN)

	tmpstr = ""
	for a in accuMatrix:
		tmpstr += ("%.3f" % a + "\t")

	print str(accuracy) +"  "+ str(ConfuMat) + "  " + str(tmpstr) + "%.3f"  % Fmeasure 
	return accuMatrix, ConfuMat, accuracy, Fmeasure, TPR, FPR 


# ========= Output result into files ===========
def ouputToFile(filepath, method, accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR):
	f = open(filepath, 'a') # append the result to existing file
	f.write(method + ':  Accuracy: ' + str(accu) + '\n\n')
	f.write('Precission:\trecall:\t\tsensitivity:\tspecificity:\n')
	
	for a in accuMatrix:
		f.write("%.4f" % a + '\t\t')
	f.write('\n\nF - measure: ' + str(Fmeasure) +'\n\n')

	TP = ConfuMat[0] # correctly classified
	FN = ConfuMat[1]
	FP = ConfuMat[2]
	TN = ConfuMat[3] # correctly classified

	f.write('              Output'+'\n')
	f.write('          class 1, class 0'+'\n')
	f.write(' class 1    ' + str(TP) + '        ' + str(FP)+'\n')
	f.write(' class 0    ' + str(FN) + '        ' + str(TN)+'\n')
	f.write('-----------------------------------------------------------------------------------\n\n')
	f.close()


def main():
	f1 = open('Output_first_dataset.txt', 'w') # create a new file
	f1.close()
	f2 = open('Output_second_dataset.txt', 'w') # create a new file
	f2.close()

	# first dataset folder, you can change the folder's name into new dataset
	trainFea, trainLab = readData('outDataClass')   
	testFea, testLab = readData('outDataClassLDL') # second dataset folder

	# =============== test first dataset, LOOV of KNN for different K values (weighted points) ===============
	# accuracy = []
	# accuuracyMatrix = []
	# kValue = []
	# TPRList = []
	# FPRList = []
	# FmeasureList = []
	# for i in range(19):
	# 	k = 1 + i *2
	# 	# change the 'weight1' to any parameter described above in "calStatic" function calStatic will work
	# 	# you can change 'weight1' into 
	# 	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(trainFea, trainLab, 'knn', 'weight1', k)
	# 	kValue.append(k)
	# 	accuracy.append(accu)
	# 	accuuracyMatrix.append(accuMatrix)
	# 	FmeasureList.append(Fmeasure)
	# 	TPRList.append(TPR)
	# 	FPRList.append(FPR)

	# following 2 function are defined in PlotData.py
	## plotGraph(kValue, accuracy, 'K Value', 'Accuracy', 'bo-','KNN for weighted points from each user', 'AccuKnnWeight1.jpg')
	## plotROC(FPRList,TPRList, 'False Positive Rate', 'True Positive Rate', 'bo', 'ROC curve for KNN with different K value (weighted points)', 'ROCKnnWeight.jpg' )	

	#  Experiment for first dataset
	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(trainFea, trainLab, 'knn', 'mean', 5)
	ouputToFile('Output_first_dataset.txt', 'Exp 1.  5-NN LOOCV (feature: means of data)', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)
	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(trainFea, trainLab, 'knn', 'addMean', 5)
	ouputToFile('Output_first_dataset.txt', 'Exp 2.  5-NN LOOCV (feature: addMean)', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)
	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(trainFea, trainLab, 'knn', 'addMean_miusMean', 5)
	ouputToFile('Output_first_dataset.txt', 'Exp 3.  5-NN LOOCV (feature: addMean_miusMean)', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)
	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(trainFea, trainLab, 'knn', 'mean_median', 7)
	ouputToFile('Output_first_dataset.txt', 'Exp 4.  7-NN LOOCV (feature: mean_median)', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)
	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(trainFea, trainLab, 'knn', 'weight1', 5)
	ouputToFile('Output_first_dataset.txt', 'Exp 5.  7-NN LOOCV (feature: weight1)', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)
	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(trainFea, trainLab, 'knn', 'median', 21)
	ouputToFile('Output_first_dataset.txt', 'Exp 6.  7-NN LOOCV (feature: median)', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)
	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(trainFea, trainLab, 'knn', 'rms', 5)
	ouputToFile('Output_first_dataset.txt', 'Exp 7.  7-NN LOOCV (feature: rms)', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)
	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(trainFea, trainLab, 'knn', 'mean_rms', 5)
	ouputToFile('Output_first_dataset.txt', 'Exp 8.  7-NN LOOCV (feature: mean_rms)', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)


	# =============== test second dataset LOOV of KNN for different K values (weighted points) ===============
	# accuracy = []
	# accuuracyMatrix = []
	# kValue = []
	# TPRList = []
	# FPRList = []
	# FmeasureList = []
	# for i in range(19):
	# 	k = 1 + i *2
	# 	# change the 'weight1' to any parameter described above in "calStatic" function calStatic will work
	# 	# you can change 'weight1' into 
	# 	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(testFea, testLab, 'knn', 'mean_rms', k)
	# 	kValue.append(k)
	# 	accuracy.append(accu)
	# 	accuuracyMatrix.append(accuMatrix)
	# 	FmeasureList.append(Fmeasure)
	# 	TPRList.append(TPR)
	# 	FPRList.append(FPR)

	# following 2 function are defined in PlotData.py
	## plotGraph(kValue, accuracy, 'K Value', 'Accuracy', 'bo-','KNN for each subjects', 'SecAccumean_rms.jpg')
	## plotROC(FPRList,TPRList, 'False Positive Rate', 'True Positive Rate', 'bo', 'ROC curve for KNN with different K value', 'SecROCKnnmean_rms.jpg')


	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(testFea, testLab, 'knn', 'std', 3)
	ouputToFile('Output_second_dataset.txt', 'Exp 1.  3-NN LOOCV (feature: std of data)', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)
	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(testFea, testLab, 'knn', 'addMean', 7)
	ouputToFile('Output_second_dataset.txt', 'Exp 2.  7-NN LOOCV (feature: "addMean")', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)
	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(testFea, testLab, 'knn', 'none', 7)
	ouputToFile('Output_second_dataset.txt', 'Exp 3.  3-NN LOOCV (feature: "none")', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)
	# ==============================================================================================


	# ============================= test first dataset, LOOV for perceptron =============================
	# accuracy = []
	# accuuracyMatrix = []
	# TPRList = []
	# FPRList = []
	# FmeasureList = []	
	# thresh = []
	# for j in range(11):
	# 	threshold = 0.1 * j
	# 	# accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR  = LOOCV(trainFea, trainLab, 'per', 'weight1', [1000, 0.25, threshold, False, True])
	# 	accuracy.append(accu)
	# 	accuuracyMatrix.append(accuMatrix)
	# 	FmeasureList.append(Fmeasure)
	# 	TPRList.append(TPR)
	# 	FPRList.append(FPR)
	# 	thresh.append(threshold)

	## plotGraph(thresh, accuracy, 'Threshold', 'Accuracy', 'bo-','Single-layer Perceptron (1000 iteration)', 'AccuPerNobiasYeskernelweight1.jpg')
	## plotROC(FPRList,TPRList, 'False Positive Rate', 'True Positive Rate', 'bo', 'ROC curve for Single-layer Perceptron', 'ROCperNobiasYeskernelweight1.jpg')

	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(trainFea, trainLab, 'per', 'mean_rms', [1000, 0.25, 0.2, False, True])
	ouputToFile('Output_first_dataset.txt', 'Exp 9.  perceptron LOOCV (feature: "mean_rms") no Bias, has Kernel', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)
	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(trainFea, trainLab, 'per', 'weight1', [1000, 0.25, 0.1, False, True])
	ouputToFile('Output_first_dataset.txt', 'Exp 10.  perceptron LOOCV (feature: "weight1") no Bias, has Kernel', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)
	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(trainFea, trainLab, 'per', 'mean', [1000, 0.25, 0.1, False, False])
	ouputToFile('Output_first_dataset.txt', 'Exp 11.  perceptron LOOCV (feature: "mean") no Bias, no Kernel', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)


	# ============================= test second dataset, LOOV for perceptron =============================
	# accuracy = []
	# accuuracyMatrix = []
	# TPRList = []
	# FPRList = []
	# FmeasureList = []	
	# thresh = []
	# for j in range(11):
	# 	threshold = 0.1 * j
	# 	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR  = LOOCV(testFea, testLab, 'per', 'weight1', [1000, 0.25, threshold, True, True])
	# 	accuracy.append(accu)
	# 	accuuracyMatrix.append(accuMatrix)
	# 	FmeasureList.append(Fmeasure)
	# 	TPRList.append(TPR)
	# 	FPRList.append(FPR)
	# 	thresh.append(threshold)

	## plotGraph(thresh, accuracy, 'Threshold', 'Accuracy', 'bo-','Single-layer Perceptron (1000 iteration)', 'SecAccuPerYesbiasYeskernelweight1.jpg')
	## plotROC(FPRList,TPRList, 'False Positive Rate', 'True Positive Rate', 'bo', 'ROC curve for Single-layer Perceptron', 'SecROCperYesbiasYeskernelweight1.jpg')

	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(testFea, testLab, 'per', 'mean_median', [1000, 0.25, 0.2, False, True])
	ouputToFile('Output_second_dataset.txt', 'Exp 4.  perceptron LOOCV (feature: "mean_median") no Bias, has Kernel', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)
	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(testFea, testLab, 'per', 'rms', [1000, 0.25, 0.3, False, True])
	ouputToFile('Output_second_dataset.txt', 'Exp 5.  perceptron LOOCV (feature: "rms") no Bias, has Kernel', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)

	# ============================= test first dataset, LOOV for MLP =============================
	# accuracy = []
	# accuuracyMatrix = []
	# TPRList = []
	# FPRList = []
	# FmeasureList = []	
	# thresh = []
	# for j in range(5):
	# 	threshold = 0.2 * j + 0.1
	# 	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(trainFea, trainLab, 'MLP', 'mean', [1000, 0.25, threshold, True, False, 1, 'logistic', 3])
	# 	accuracy.append(accu)
	# 	accuuracyMatrix.append(accuMatrix)
	# 	FmeasureList.append(Fmeasure)
	# 	TPRList.append(TPR)
	# 	FPRList.append(FPR)
	# 	thresh.append(threshold)

	## plotGraph(thresh, accuracy, 'Threshold', 'Accuracy', 'bo-','MLP (1000 iteration)', 'AccuMLPhide3.jpg')
	## plotROC(FPRList,TPRList, 'False Positive Rate', 'True Positive Rate', 'bo', 'ROC curve for MLP', 'ROCMLPhide3.jpg')

	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(trainFea, trainLab, 'MLP', 'mean', [1000, 0.25, 0.5, True, False, 1, 'logistic', 2])
	ouputToFile('Output_first_dataset.txt', 'Exp 12.  MLP LOOCV with 2 hidden layer nodes (feature: "mean") has Bias, no Kernel', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)
	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(trainFea, trainLab, 'MLP', 'mean', [1000, 0.25, 0.5, True, False, 1, 'logistic', 3])
	ouputToFile('Output_first_dataset.txt', 'Exp 13.  MLP LOOCV with 3 hidden layer nodes (feature: "mean") has Bias, no Kernel', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)

	# ============================= test second dataset, LOOV for MLP =============================
	# accuracy = []
	# accuuracyMatrix = []
	# TPRList = []
	# FPRList = []
	# FmeasureList = []	
	# thresh = []
	# for j in range(10):
	# 	threshold = 0.1 * j + 0.1
	# 	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(testFea, testLab, 'MLP', 'mean', [1000, 0.25, threshold, True, False, 1, 'logistic', 2])
	# 	accuracy.append(accu)
	# 	accuuracyMatrix.append(accuMatrix)
	# 	FmeasureList.append(Fmeasure)
	# 	TPRList.append(TPR)
	# 	FPRList.append(FPR)
	# 	thresh.append(threshold)

	# # plotGraph(thresh, accuracy, 'Threshold', 'Accuracy', 'bo-','MLP (1000 iteration)', 'SecAccuMLPhide2.jpg')
	# # plotROC(FPRList,TPRList, 'False Positive Rate', 'True Positive Rate', 'bo', 'ROC curve for MLP', 'SecROCMLPhide2.jpg')


	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(testFea, testLab, 'MLP', 'mean', [1000, 0.25, 0.5, True, False, 1, 'logistic', 2])
	ouputToFile('Output_second_dataset.txt', 'Exp 6.  MLP LOOCV with 2 hidden layer nodes (feature: "mean") has Bias, no Kernel', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)
	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR = LOOCV(testFea, testLab, 'MLP', 'mean', [1000, 0.25, 0.5, True, False, 1, 'logistic', 3])
	ouputToFile('Output_second_dataset.txt', 'Exp 7.  MLP LOOCV with 3 hidden layer nodes (feature: "mean") has Bias, no Kernel', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)

	# ============================= test first dataset, LOOV for MLP batch =============================
	# accuracy = []
	# accuuracyMatrix = []
	# TPRList = []
	# FPRList = []
	# FmeasureList = []	
	# thresh = []
	# for j in range(11):
	# 	threshold = 0.1 * j
	# 	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR  = LOOCV(trainFea, trainLab, 'MLPbatch', 'mean', [1000, 0.3, threshold, 2])
	# 	accuracy.append(accu)
	# 	accuuracyMatrix.append(accuMatrix)
	# 	FmeasureList.append(Fmeasure)
	# 	TPRList.append(TPR)
	# 	FPRList.append(FPR)
	# 	thresh.append(threshold)

	## plotGraph(thresh, accuracy, 'Threshold', 'Accuracy', 'bo-','MLP batch (10000 iteration)', 'AccuMLPlinearbatchHide2.jpg')
	## plotROC(FPRList,TPRList, 'False Positive Rate', 'True Positive Rate', 'bo', 'ROC curve for MLP', 'ROCMLPbatchlinearHide2.jpg')

	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR  = LOOCV(trainFea, trainLab, 'MLPbatch', 'mean', [1000, 0.3, 0.5, 2, 'linear'])
	ouputToFile('Output_first_dataset.txt', 'Exp 14.  MLP(batch) (linear) LOOCV with 2 hidden layer nodes (feature: "mean")', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)
	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR  = LOOCV(trainFea, trainLab, 'MLPbatch', 'mean', [1000, 0.3, 0.5, 3, 'logistic'])
	ouputToFile('Output_first_dataset.txt', 'Exp 15.  MLP(batch) (logistic) LOOCV with 3 hidden layer nodes (feature: "mean")', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)
	# ============================= test second dataset, LOOV for MLP batch=============================
	# accuracy = []
	# accuuracyMatrix = []
	# TPRList = []
	# FPRList = []
	# FmeasureList = []	
	# thresh = []
	# for j in range(11):
	# 	threshold = 0.1 * j
	# 	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR  = LOOCV(testFea, testLab, 'MLPbatch', 'mean', [1000, 0.3, threshold, 3, 'linear'])
	# 	accuracy.append(accu)
	# 	accuuracyMatrix.append(accuMatrix)
	# 	FmeasureList.append(Fmeasure)
	# 	TPRList.append(TPR)
	# 	FPRList.append(FPR)
	# 	thresh.append(threshold)

	# plotGraph(thresh, accuracy, 'Threshold', 'Accuracy', 'bo-','MLP batch (10000 iteration)', 'SecAccuMLPlinearbatchHide3.jpg')
	# plotROC(FPRList,TPRList, 'False Positive Rate', 'True Positive Rate', 'bo', 'ROC curve for MLP', 'SecROCMLPbatchlinearHide3.jpg')

	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR  = LOOCV(testFea, testLab, 'MLPbatch', 'mean', [1000, 0.3, 0.5, 2, 'linear'])
	ouputToFile('Output_second_dataset.txt', 'Exp 8.  MLP(batch) LOOCV with 2 hidden layer nodes (feature: "mean")', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)
	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR  = LOOCV(testFea, testLab, 'MLPbatch', 'mean', [1000, 0.3, 0.5, 3, 'linear'])
	ouputToFile('Output_second_dataset.txt', 'Exp 9.  MLP(batch) LOOCV with 3 hidden layer nodes (feature: "mean")', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)
	accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR  = LOOCV(testFea, testLab, 'MLPbatch', 'mean', [1000, 0.3, 0.5, 4, 'linear'])
	ouputToFile('Output_second_dataset.txt', 'Exp 10.  MLP(batch) LOOCV with 4 hidden layer nodes (feature: "mean")', accuMatrix, ConfuMat, accu, Fmeasure, TPR, FPR)


if __name__ == "__main__":
	main()