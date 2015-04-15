#--------------------------------------------------
# UCLA CS 260 - Machine Learning Algorithm
# Yao-Jen Chang, 2015 @UCLA
# Email: autekwing@ucla.edu
#
# Functions about loading training data
#--------------------------------------------------


# Read the trainning data
# Have 39 users with 20 label 0, 19 label 1
# Each user data have time series data from 26 week

# userData[0] has two list, which mean Diastolic and Systolic
# each list has 26 data point
def readData(path):
	import os
	userData = []
	label = []

	files = os.listdir('./' + path)
	for filename in files:
		if filename[-4:] != '.csv': # if file is not csv type, jump
			continue

		tmpUserData, tmpLabel = readcsvData(path + '/' + filename)
		userData.append(tmpUserData)
		label.append(tmpLabel)

	return userData, label


def readcsvData(filePath):
	import csv
	with open(filePath) as csvfile:
		reader = csv.DictReader(csvfile)
		bloodPres = [[], []]
		
		ifHealthy = filePath[-5]# [-5] will be the labe of each user
		if ifHealthy == '0' or ifHealthy == '1':
			ifHealthy = int(ifHealthy)
		else:
			ifHealthy = -1
		
		for row in reader:
			# print(row['Diastolic'], row['Systolic'])
			bloodPres[0].append(int(row['Diastolic']))
			bloodPres[1].append(int(row['Systolic']))

		return bloodPres, ifHealthy 


def listUserdate(list):
	# list will have 2 list with same size
	size = len(list[0]) # same as len(list[1])
	for i in range(size):
		print str(list[0][i]) + ',  ' + str(list[1][i])


