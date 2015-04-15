#--------------------------------------------------
# UCLA CS 260 - Machine Learning Algorithm
# Yao-Jen Chang, 2015 @UCLA
# Email: autekwing@ucla.edu
#
# Functions about plotting data
#--------------------------------------------------

def plotMeanData(userData, label, timeFlag):
	from numpy import mean
	import matplotlib.pyplot as plt
	for i in range(39):
		if label[i] == 0:
			plt.plot(mean(userData[i][0][timeFlag:]), mean(userData[i][1][timeFlag:]), 'ro')
		elif label[i] == 1:
			plt.plot(mean(userData[i][0][timeFlag:]), mean(userData[i][1][timeFlag:]), 'bo')
	
	plt.xlabel('Diastolic Pressure')
	plt.ylabel('Systolic Pressure')
	plt.title('Average data from 39 subjects')
	plt.grid(True)
	plt.savefig("averPoint.jpg")


def plotAllData(userData, label, timeFlag):
	import matplotlib.pyplot as plt
	for i in range(39):
		timeFlag = 0
		if label[i] == 0:
			plt.plot(userData[i][0][timeFlag:], userData[i][1][timeFlag:], 'ro')
		elif label[i] == 1:
			plt.plot(userData[i][0][timeFlag:], userData[i][1][timeFlag:], 'bo')

	plt.xlabel('Diastolic Pressure')
	plt.ylabel('Systolic Pressure')
	plt.title('All data point from 39 subjects')
	plt.grid(True)
	plt.savefig("allPoint.jpg")


def plotStd(userData, label, timeFlag):
	from numpy import std
	import matplotlib.pyplot as plt

	for i in range(39):
		if label[i] == 0:
			plt.plot(std(userData[i][0][timeFlag:]), std(userData[i][1][timeFlag:]), 'ro')
		elif label[i] == 1:
			plt.plot(std(userData[i][0][timeFlag:]), std(userData[i][1][timeFlag:]), 'bo')
	
	plt.xlabel('Diastolic Pressure standard deviation')
	plt.ylabel('Systolic Pressure standard deviation')
	plt.title('All standard deviation from 39 subjects')
	plt.grid(True)
	plt.savefig("allStd.jpg")


def plotMedianData(userData, label, timeFlag):
	from numpy import median
	import matplotlib.pyplot as plt
	for i in range(39):
		if label[i] == 0:
			plt.plot(median(userData[i][0][timeFlag:]), median(userData[i][1][timeFlag:]), 'ro')
		elif label[i] == 1:
			plt.plot(median(userData[i][0][timeFlag:]), median(userData[i][1][timeFlag:]), 'bo')
	
	plt.xlabel('Diastolic Pressure')
	plt.ylabel('Systolic Pressure')
	plt.title('Median data of each subject')
	plt.grid(True)
	plt.savefig("medianPoint.jpg")

def plotROC(xEle, yEle, xLab, yLab, color = 'bo-', TitleName = 'accuracy', fileName = 'tmpAccuracy.jpg'):
	import matplotlib.pyplot as plt
	plt.plot(xEle, yEle, color)
	
	plt.xlabel(xLab)
	plt.ylabel(yLab)

	# plt.xlim([0,1])
	# plt.ylim([0,1])

	plt.title(TitleName)
	plt.grid(True)
	plt.savefig(fileName)


def plotGraph(xEle, yEle, xLab, yLab, color = 'bo-', TitleName = 'accuracy', fileName = 'tmpAccuracy.jpg'):
	import matplotlib.pyplot as plt
	plt.plot(xEle, yEle, color)
	
	plt.xlabel(xLab)
	plt.ylabel(yLab)
	plt.title(TitleName)
	plt.grid(True)
	plt.savefig(fileName)


