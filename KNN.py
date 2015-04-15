#--------------------------------------------------
# UCLA CS 260 - Machine Learning Algorithm
# Yao-Jen Chang, 2015 @UCLA
# Email: autekwing@ucla.edu
#
# KNN classifier
#--------------------------------------------------

def knn(train, trainLab, test, k):
	import numpy as np

	distances = []
	for tra in train:
		if type(tra) == type([]): # more than 1 dimension features
			size = len(test)
			dis = 0
			for j in range(size):
				dis += abs(tra[j] - test[j]) ** 2
		else: # only one feature
			dis = abs(tra - test[0])
		distances.append(dis)

	indices = np.argsort(distances) # return a indices list of sorted distances ascendingly

	neigh = [0, 0]
	for i in range(0, k): # from 1 to k
		neighLabel = trainLab[indices[i]] # 0 or 1
		neigh[neighLabel] += 1

	# print neigh
	if neigh[0] > neigh[1]:
		return 0
	return 1

