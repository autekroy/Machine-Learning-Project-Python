#--------------------------------------------------
# UCLA CS 260 - Machine Learning Algorithm
# Yao-Jen Chang, 2015 @UCLA
# Email: autekwing@ucla.edu
#
# Function about perceptron algorithm and MLP(sequential)
# normalization method for data points
#--------------------------------------------------

import numpy as np

#---------------- single layer perceptron ----------------------------
# Three node N0(bias node), N1, N2
# the bias node input is always -1

# the parameter is a list represent as follow:
# parameter[0]: number of iteration
# parameter[1]: learning rate
# parameter[2]: value for activation function
# parameter[3]: if there's a bias node
# parameter[4]: if use kernel function to increase dimension
def perceptron(train, trainLab, test, parameter):
	import random
	# no bias, activatio: -1
	# has bias, activation: 0
	
	# parameter = [10000, 0.3, 0] # number of iteration, learning rate, activation function
	nIter      = parameter[0]
	learnRate  = parameter[1]
	activation = parameter[2]
	bias       = parameter[3]
	useKernel  = parameter[4]

	# make 2 dimension feature into 6 dimentsion, and hope they will be linear seperable
	if len(train[0]) == 2 and useKernel:
		train = kernel(train)
		test = kernel(test)	

	numInput = len(train)
	numFea = len(train[0])
	numNode = numFea # number of node is the number of features
	if bias:
		numNode += 1 # add one more bias node

	weights = []
	randomWight = 0.05
	for i in range(numNode):
		weights.append(random.random() * randomWight - (randomWight/2)) # -0.05 ~ 0.05
	# print weights

	for n in range(nIter):
		# if n % 20 == 0:
			# print weights
		for j in range(numInput): # every input of training data
			tra = train[j]
			result = 0
			for i in range(numFea): # from [0] to [numFea -1]
				result = result + tra[i] * weights[i]
			if bias: # bias node with always -1 input
				result = result -1 * weights[numFea] # the last weight
			# if the neuron fired
			# print str(result) + '  ' + str(tra) + '  ' + str(trainLab[j])
			if result < activation:
				predClass = 0
			else:
				predClass = 1
			# the result is correct
			if (predClass == 0 and trainLab[j] == 0) or (predClass == 1 and trainLab[j] == 1):
				continue
			else: # update weights
				for i in range(numFea):
					weights[i] = weights[i] - learnRate * (predClass - trainLab[j]) * tra[i]
				if bias:
					weights[numFea] = weights[numFea] - learnRate * (predClass - trainLab[j]) * (-1)

	# for test data
	result = 0
	for i in range(numFea): # from [0] to [numFea -1]
		result = result + test[i] * weights[i]
	if bias: # bias node with always -1 input
		result = result -1 * weights[numFea] # the last weight
	# if the neuron fired
	# print str(result) + '  ' + str(tra) + '  ' + str(trainLab[j])
	if result < activation:
		ans = 0
	else:
		ans = 1

	# print weights

	return ans



#------------------------- multi-layer perceptron -------------------------
# the parameter is a list represent as follow:
# parameter[0]: number of iteration
# parameter[1]: learning rate
# parameter[2]: value for activation function
# parameter[3]: if there's a bias node
# parameter[4]: if use kernel function to increase dimension
# parameter[5]: beta value
# parameter[6]: Different types of output neurons, which has 'linear', 'logistic', and 'softmax'
# parameter[7]: number of node in hidden layer
def MLP(train, trainLab, test, parameter):
	import random
	from numpy import sqrt
	# no bias, activatio: -1
	# has bias, activation: 0
	
	# parameter = [10000, 0.3, 0] # number of iteration, learning rate, activation function
	nIter      = parameter[0]
	learnRate  = parameter[1]
	activation = parameter[2]
	bias       = parameter[3]
	useKernel  = parameter[4]
	beta       = parameter[5]
	outtype    = parameter[6] # Different types of output neurons
	numHid     = parameter[7] # number of node in hidden layer
	# make 2 dimension feature into 6 dimentsion, and hope they will be linear seperable
	if len(train[0]) == 2 and useKernel:
		train = kernel(train)
		test = kernel(test)	

	numInput = len(train)
	numFea = len(train[0])
	numNode = numFea # number of node is the number of features
	

	weights1 = [] # weights for input layer
	weights2 = [] # weights for hidden layer
	
	randomWight = 1/sqrt(numHid) * 2
	if bias:
		randomWight = 1/sqrt(numHid + 1) * 2

	for i in range(numHid):
		weights2.append(random.random() * randomWight - (randomWight/2)) 
	if bias:
		numNode += 1 # add one more bias node
		weights2.append(random.random() * randomWight - (randomWight/2)) # bias node in hidden layer

	# every node(include bias) in input layer will have a weight for every node in hidden layer
	randomWight = 1/sqrt(numNode) * 2
	for i in range(numNode):
		tmp = []
		for j in range(numHid):
			tmp.append(random.random() * randomWight - (randomWight/2)) # -0.05 ~ 0.05
		weights1.append(tmp)

	# print weights1, weights2
	change = range(numInput)

	# ==================== Training =====================
	for n in range(nIter): # number of iteration
		np.random.shuffle(change) # random the order of training data set
		# print weights1
		for k in change:# every input vector of training data
			tra = train[k]
			# ------- Forwards phase in input layer -------
			hiddenInput = []
			for j in range(numHid): # number of node j in hidden layer
				result = 0
				for i in range(numFea): # from [0] to [numFea -1]
					result = result + tra[i] * weights1[i][j]
				if bias: # input layer bias node with always -1 input
					result = result -1.0 * weights1[numFea][j] # the last weight for input bias node

				result = 1.0/(1.0 + np.exp(-beta * result)) # the input result will be input of hidden layer
				hiddenInput.append(result)

			# ------- Forwards phase in hidden layer -------
			OutputResult = 0
			for i in range(numHid): # from [0] to [numFea -1]
				OutputResult = OutputResult + hiddenInput[i] * weights2[i]
			if bias: # hidden layer bias node with always -1 input
				OutputResult = OutputResult -1.0 * weights2[numHid] # the last weight

			if outtype == 'linear':
				OutputResult = OutputResult
			elif outtype == 'logistic':
				OutputResult = 1.0/(1.0 + np.exp(-beta * OutputResult))
			else:
				print 'error type'
			# print hiddenInput, OutputResult

			ans = 1 # it remain be 1 if OutputResult >= activation
			if OutputResult < activation:
				ans = 0
			# print ans, trainLab[k]

			# ------- Backward phase in computing error at output -------
			if outtype == 'linear':
				outputError = (OutputResult - trainLab[k])/numInput
			elif outtype == 'logistic':
				outputError = (OutputResult - trainLab[k]) * OutputResult * (1 - OutputResult) # only one output node
			else:
				print 'error type'

			# ------- Backward phase in computing error in hidden layer -------
			hiddenError = []
			for i in range(numHid):
				tmp = hiddenInput[i] * (1 - hiddenInput[i]) * weights2[i] * outputError
				hiddenError.append(tmp)

			# ------- Backward phase in updating output layer weights-------
			for i in range(numHid):
				weights2[i] = weights2[i] - learnRate * outputError * hiddenInput[i]
			if bias:
				weights2[numHid] =  weights2[i] - learnRate * outputError * (-1)

			# ------- Backward phase in updating hidden layer weights-------
			for j in range(numHid):
				for i in range(numFea):
					weights1[i][j] = weights1[i][j] - learnRate * hiddenError[j] * tra[i]
				if bias:
					weights1[numFea][j] = weights1[numFea][j] - learnRate * hiddenError[j] * (-1)
			
		# print outputError, hiddenError



	# ==================== run the MLP for the test data ===================

	# ------- Forwards phase in input layer -------
	hiddenInput = []
	for j in range(numHid): # number of node j in hidden layer
		result = 0
		for i in range(numFea): # from [0] to [numFea -1]
			result = result + test[i] * weights1[i][j]
		if bias: # bias node with always -1 input
			result = result -1.0 * weights1[numFea][j] # the last weight for input bias node

		result = 1.0/(1.0 + np.exp(-beta * result)) # the input result will be input of hidden layer
		hiddenInput.append(result)

	# ------- Forwards phase in hidden layer -------
	OutputResult = 0
	for i in range(numHid): # from [0] to [numFea -1]
		OutputResult = OutputResult + hiddenInput[i] * weights2[i]
	if bias: # bias node with always -1 input
		OutputResult = OutputResult -1.0 * weights2[numHid] # the last weight

	if outtype == 'linear':
		OutputResult = OutputResult
	elif outtype == 'logistic':
		OutputResult = 1.0/(1.0 + np.exp(-beta * OutputResult)) 
	else:
		print 'error type'
	# print hiddenInput, OutputResult

	ans = 1 # it remain be 1 if OutputResult >= activation
	if OutputResult < activation:
		ans = 0

	return  ans # it remain be 1 if OutputResult >= activation

# nomalize data into 0~1 range
# Max = [110, 170]
# Min = [50, 80]
# 50~110,   80~170
def normalize(data, interval = 1, dim = 0):
	import copy
	normalData = copy.deepcopy(data)

	if type(normalData[0]) != type([]):
		normalData = [normalData]

	dimension = len(normalData[0]) # number of features
	if dim != 0:
		dimension = dim

	size = len(normalData)

	Max = [102, 163, 102, 163, 102, 163]
	Min = [60, 92, 60, 92, 60, 92]
	# Max = [102, 163, 9, 16]
	# Min = [60, 92, 0, 0]

	# This part can run to determine the Max and Min list when training different data set
	# for i in range(dimension): # for different features 
	# 	tmpMax = 0
	# 	tmpMin = 1000
	# 	for j in range(size): # how many element in list (normalData) 
	# 		tmpMax = max(normalData[j][i], tmpMax)
	# 		tmpMin = min(normalData[j][i], tmpMin)
	# 	tmpMax += 1 # to make new normalized data won't be 1
	# 	tmpMin -= 1 # to make new normalized data won't be 0
	# 	Max.append(tmpMax)
	# 	Min.append(tmpMin)

	for point in normalData: # how many element in list (normalData) 
		for i in range(dimension): # how many feature for each data element
			if interval == 1:
				point[i] = (point[i] - Min[i])/(Max[i] - Min[i])
			else:
				point[i] = ( 2 *(point[i] - Min[i])/(Max[i] - Min[i])) - 1

	if len(normalData) == 1: #only one list of element
		return normalData[0]

	return normalData


# only use for 2 dimenstion data
def kernel(data):
	import copy
	from math import sqrt

	extraData = []

	if type(data[0]) != type([]):
		return [1, sqrt(2) * data[0], sqrt(2) * data[1], data[0] ** 2, data[1] ** 2, sqrt(2)*data[0]*data[1]]

	for point in data:
		tmp = [1, sqrt(2) * point[0], sqrt(2) * point[1], point[0] ** 2, point[1] ** 2, sqrt(2)*point[0]*point[1]]
		extraData.append(tmp)

	return extraData
