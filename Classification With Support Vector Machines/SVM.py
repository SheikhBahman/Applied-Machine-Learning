# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 10:36:23 2019

@author: Bahman
"""
import csv
import math
import numpy as np
import random
from matplotlib import pyplot as plt

def readMyCSVData(fileName):
    with open(fileName, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = []
        label = []
        for row in reader: 
            data.append([float(row[0]), float(row[2]), float(row[4]), float(row[10]), float(row[11]), float(row[12])])
            
            if len(row) == 15:
                if row[14] == ' <=50K':
                    label.append(-1)
                elif row[14] == ' >50K':
                    label.append(+1)
                else:
                    print("Data Error!!")

                
                
        csvfile.close()
        return data, label

def average(listNumbers):
	return sum(listNumbers)/float(len(listNumbers))

def standarDeviation(listNumbers):
	avgerage = average(listNumbers)	
	return math.sqrt(sum([pow(x-avgerage,2) for x in listNumbers])/float(len(listNumbers)-1))

def dataStandardization(data):
    print("Scaling the variables:", end="")
    normalParameters = [(average(feature), standarDeviation(feature)) for feature in zip(*data)]
    for row in data:
        for i in range(len(row)):
                row[i] = (row[i] - normalParameters[i][0]) / normalParameters[i][1]
    print("...OK")           

def splitDataTrainTest(dataX, dataY, percentage):
    dataLen = len(dataX)
    testLen = round(percentage * dataLen)
    trainX = dataX.copy()
    trainY = dataY.copy()
    testX = []
    testY = []
    for k in range(testLen):
        i =  random.randrange(len(trainX))
        testX.append(trainX[i])
        testY.append(trainY[i])
        trainX.pop(i)    
        trainY.pop(i)
    return trainX, trainY, testX, testY

def predictBySVM(a, b, data):
    results = []
    for xV in data:
        value = np.dot(xV, a) + b
        if value > 0.0:
            results.append(+1)
        else:
            results.append(-1)
    return results

def accuracy(predictedData, testData):
    correct = 0
    for i in range(len(testData)):
        if testData[i] ==  predictedData[i]:
            correct += 1
    return correct/float(len(testData))

def vectorMagnitude(data):
    return math.sqrt(sum([i ** 2 for i in data]))



#//////Main
      
originalTrainX, originalTrainY = readMyCSVData('train.txt')
originalTestX, originalTestY = readMyCSVData('test.txt')
print("Training data read: ", len(originalTrainX))
print("Testing data read: ", len(originalTestX))

dataStandardization(originalTrainX)
dataStandardization(originalTestX)

regularizations = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 1e-1, 1]
seasons = 1000
kStep = 30
steps = 4000
random.uniform(0, 1)
a = [random.uniform(0, 1) for _ in range(len(originalTrainX[0]))]
b = random.uniform(0, 1)
 
trainX, trainY, testX, testY = splitDataTrainTest(originalTrainX, originalTrainY, 0.1)
dicAccuracylanda = {}
dicCofALanda = {}
dicCofBLanda = {}
dicCofAllLanda = {}
for landa in regularizations:
    accuracySeason = {}
    coefficientASeason = {}
    coefficientBSeason = {}
    coefficientMagnitudeSeason = {}
    for season in range(seasons):
        stepLength = 1.0 / (0.1 * season + 100) #etaa        
        seasonTrainX, seasonTrainY, heldOutvalidationX, heldOutvalidationY = splitDataTrainTest(trainX, trainY, 0.1)        
        for step in range(steps): 
            k =  random.randrange(len(trainX)) #Nb = 1   #number of batch items                        
            if trainY[k]*(np.dot(trainX[k], a) + b) >= 1:
                for feature in range(len(trainX[k])):
                    a[feature] = a[feature] - stepLength * landa * a[feature]
            else:
                for feature in range(len(trainX[k])):
                    a[feature] = a[feature] - stepLength * (landa * a[feature] - trainY[k] * trainX[k][feature])
                b = b + stepLength * trainY[k]
            if step % kStep == 0:
                accuracyS = accuracy(predictBySVM(a, b, heldOutvalidationX), heldOutvalidationY)
                accuracySeason[step] = accuracyS
                magnitudeA = vectorMagnitude(a)
                coefficientASeason[step] = magnitudeA
                coefficientBSeason[step] = b
                coefficientMagnitudeSeason[step] = math.sqrt(magnitudeA*magnitudeA + b*b)
    dicAccuracylanda[landa] = accuracySeason
    dicCofALanda[landa] = coefficientASeason
    dicCofBLanda[landa] = coefficientBSeason
    dicCofAllLanda[landa] = coefficientMagnitudeSeason
#select the best landa    
bestLanda = -0.1
maxAccuracy = 0.0
for landa in dicAccuracylanda:    
    items = (sorted(dicAccuracylanda[landa]))    
    accuracy = dicAccuracylanda[landa][items[-1]]    
    if accuracy > maxAccuracy:
        maxAccuracy = accuracy
        bestLanda = landa
#Cof a and b with the best landa
for season in range(seasons):
    stepLength = 1.0 / (0.1 * season + 100) #etaa
    for step in range(steps): 
        k =  random.randrange(len(originalTrainX)) #Nb = 1   #number of batch items                        
        if originalTrainY[k]*(np.dot(originalTrainX[k], a) + b) >= 1:
            for feature in range(len(originalTrainX[k])):
                a[feature] = a[feature] - stepLength * bestLanda * a[feature]
        else:
            for feature in range(len(originalTrainX[k])):
                a[feature] = a[feature] - stepLength * (bestLanda * a[feature] - originalTrainY[k] * originalTrainX[k][feature])
            b = b + stepLength * originalTrainY[k]

print("Cof. a = ", a)
print("Cof. b = ", b)

for item in sorted(dicAccuracylanda):
    lists = sorted(dicAccuracylanda[item].items()) 
    x, y = zip(*lists)
    plt.plot(x, y, label = "landa = " + str(item))
plt.legend(loc='upper left')
plt.xlabel('Season Step')
plt.ylabel('Accuracy')
plt.show()

for item in sorted(dicCofAllLanda):
    lists = sorted(dicCofAllLanda[item].items()) 
    x, y = zip(*lists)
    plt.plot(x, y, label = "landa = " + str(item))
plt.legend(loc='upper left')
plt.xlabel('Season Step')
plt.ylabel('Magnitude of Cof. Vector')
plt.show()

for item in sorted(dicCofALanda):
    lists = sorted(dicCofALanda[item].items()) 
    x, y = zip(*lists)
    plt.plot(x, y, label = "landa = " + str(item))
plt.legend(loc='upper left')
plt.xlabel('Season Step')
plt.ylabel('Magnitude of Cof. "a" vector')
plt.show()

for item in sorted(dicCofBLanda):
    lists = sorted(dicCofBLanda[item].items()) 
    x, y = zip(*lists)
    plt.plot(x, y, label = "landa = " + str(item))
plt.legend(loc='upper left')
axes = plt.gca()
axes.set_ylim([-2.0,0.0])
plt.xlabel('Season Step')
plt.ylabel('Cof. "b"')
plt.show()
    
predictedLabels = predictBySVM(a, b, originalTestX)      
with open("submission.txt", "w") as text_file:
    for item in predictedLabels:
        if item == -1:
            print('<=50K', file=text_file) 
        elif item == 1:
            print('>50K', file=text_file)
        else:
            print("Data Error2!")
    
    text_file.close()
        
    

