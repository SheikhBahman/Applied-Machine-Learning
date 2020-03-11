# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 11:52:18 2019

@author: Bahman
"""


import csv
import random   
import math 
#read the csv file    
def readData():
    with open('pima-indians-diabetes.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(list(rec) for rec in csv.reader(f, delimiter=',')) 
        for i in range(len(data)):
            data[i] = [float(x) for x in data[i]]
    return data
#split data to train and test with percentage ratio
def splitDataTrainTest(data, percentage):
    dataLen = len(data)
    testLen = round(percentage * dataLen)
    train = data.copy()
    test = []
    for k in range(testLen):
        i =  random.randrange(len(train))
        test.append(train[i])
        train.pop(i)        
    return train, test

#Split data for each class        
def splitByClass(tData):
    conData = {}    
    conData[0] = []
    conData[1] = []    
    groupProb  = {}
    count0 = 0
    count1 = 0
    for item in tData:  
        if item[-1] == 0:
            conData[0].append(item)
            count0 += 1
        elif item[-1] == 1:  
            conData[1].append(item)
            count1 += 1
        else:
            print("Data Error!")
    
    groupProb[0] = count0 / len(tData)
    groupProb[1] = count1 / len(tData)
    return conData, groupProb
#average of a list
def average(listNumbers):
	return sum(listNumbers)/float(len(listNumbers))
#variance of a list 
def variance(listNumbers):
	avgerage = average(listNumbers)
	variance = sum([pow(x-avgerage,2) for x in listNumbers])/float(len(listNumbers)-1)
	return variance
#calculate normal dist. parameters (mean and variance) for each feature for partA
def normalDisParameters_PartA(data):  
    normalParameters = [(average(feature), variance(feature)) for feature in zip(*data)]
    del normalParameters[-1]   
    return normalParameters    

#calculate normal dist. parameters (mean and variance) for each feature for partB ignore
#if index is equal to 2 or 3 or 5 or 7 and feature value is zero
def normalDisParameters_PartB(data):   
    normalParameters = []
    dataArrangedByFeature = list(zip(*data))
    for i in range(len(dataArrangedByFeature) - 1):  
        featureData = dataArrangedByFeature[i]      
        if i == 2 or i == 3 or i == 5 or i == 7:
            featureData = [x for x in featureData if x > 0.0]
        if len(featureData) > 0:
            normalParameters.append((average(featureData), variance(featureData)))
    return normalParameters  

def calculateNormalDisParm_PartA(data):
    probData = {}
    for key in data:
        probData[key] = []
        probData[key].append(normalDisParameters_PartA(data[key]))
    return probData

def calculateNormalDisParm_PartB(data):
    probData = {}
    for key in data:
        probData[key] = []
        probData[key].append(normalDisParameters_PartB(data[key]))
    return probData

#calculate the Guassian probability of x
def normalProbability(x, average, variance):
    first = 1.0 / (math.sqrt(2.0 * math.pi * variance))
    p = first * math.exp(-(x-average)*(x-average)/(2.0*variance))
    return p

#calculate the proabability of each class fro the given test data for partA
def calculateProbabilityAllFeatures(probData, groupProb, testD):
    probabilityGroups = {}
    for group, value in tData.items():      
        probabilityGroups[group] = math.log(groupProb[group])
        for i in range(len(probData[group][0])):         
            if  probData[group][0][i][1] > 0: #variance > 0
                probabilityGroups[group] += math.log(0.0000001 + normalProbability(testD[i], probData[group][0][i][0], probData[group][0][i][1]))
            
    maxP = -10000.0; groupPredicted = None
    for group in probabilityGroups:
        if groupPredicted == None or probabilityGroups[group] > maxP:
            maxP = probabilityGroups[group]
            groupPredicted = group
        
    return groupPredicted

#calculate the proabability of each class fro the given test data for partB ignore data
#if index is equal to 2 or 3 or 5 or 7 and test feature value is zero
def calculateProbabilityAllFeatures_PartB(probData, groupProb, testD):
    probabilityGroups = {}
    for group, value in tData.items(): 
        probabilityGroups[group] = math.log(groupProb[group])
        for i in range(len(probData[group][0])):            
            if i == 2 or i == 3 or i == 5 or i == 7:
                if testD[i] > 0.0:
                    probabilityGroups[group] += math.log(0.0000001 + normalProbability(testD[i], probData[group][0][i][0], probData[group][0][i][1]))
            else:
                probabilityGroups[group] += math.log(0.0000001 + normalProbability(testD[i], probData[group][0][i][0], probData[group][0][i][1]))
            
    maxP = -10000.0    
    groupPredicted = None
    for group in probabilityGroups:
        if groupPredicted == None or probabilityGroups[group] > maxP:
            maxP = probabilityGroups[group]
            groupPredicted = group
        
    return groupPredicted
    
def predictions(probData, groupProb, testData):
    predictions = []
    for testD in testData:
        predictions.append(calculateProbabilityAllFeatures(probData, groupProb, testD))
    return predictions

def predictions_PartB(probData, groupProb, testData):
    predictions = []
    for testD in testData:
        predictions.append(calculateProbabilityAllFeatures_PartB(probData, groupProb, testD))
    return predictions

#calculate the accuracy
def accuracy(predictedData, testData):
    correct = 0
    for i in range(len(testData)):
        if testData[i][-1] ==  predictedData[i]:
            correct += 1
    return correct/float(len(predictedData))
        
#////////////////////////////////////////////////////////    
#////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////
#Main
data = readData()
accuracyData = []

for i in range(10):
    train, test = splitDataTrainTest(data, 0.2)
    tData, groupProb = splitByClass(train)
    probData = calculateNormalDisParm_PartA(tData)
    predictData = predictions(probData, groupProb, test)
    accuracyData.append(accuracy(predictData, test))

print(average(accuracyData))


for i in range(10):
    train, test = splitDataTrainTest(data, 0.2)
    tData, groupProb = splitByClass(train)
    probData = calculateNormalDisParm_PartB(tData)
    predictData = predictions_PartB(probData, groupProb, test)
    accuracyData.append(accuracy(predictData, test))

print(average(accuracyData))

