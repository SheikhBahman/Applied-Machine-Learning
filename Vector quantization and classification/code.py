# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 10:32:54 2019

@author: Bahman
"""
import os
import random
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def readData():
    allData = {}
    for path, subdirs, files in os.walk("Data"):
        for name in files:
            folderName = path.split("\\")[1]           
            with open(os.path.join(path, name), 'r') as file:   
                lines = file.readlines()
                newFileData = []
                for line in lines:
                    newFileData.append([float(x) for x in line.strip().split()])
                if folderName not in allData:
                    allData[folderName] = []
                    allData[folderName].append(newFileData)
                else:
                    allData[folderName].append(newFileData)
    return allData

def chunkList(listData, chunkSize, overLapSize):
    #without overlapping
    '''m, n = divmod(len(listData), chunkSize)    
    if n != 0:        
        chunked = [listData[i:i+chunkSize] for i in range(0, len(listData), chunkSize)][:-1]
    else:
        chunked = [listData[i:i+chunkSize] for i in range(0, len(listData), chunkSize)]'''
    chunked = []
    counter = 0
    newchunked = []
    i = 0
    while i < len(listData):
        if i == len(listData) - 1:
            chunked.append(listData[-chunkSize:])
            break
        else:
            counter = counter + 1
            newchunked.append(listData[i])               
            if counter == chunkSize:
                counter = 0
                chunked.append(newchunked)
                newchunked = []
                i = i - overLapSize
            i = i + 1      
    flattenChuncked = []
    for chunk in chunked:
        newflatten = []
        for item in chunk:
            newflatten += item
        flattenChuncked.append(newflatten)
    return flattenChuncked

def makeChunks(data, chunkSize, overLapSize):
    chunkDataAll = []    
    chunkDataCategoryFiles = {}    
    for category in data:
        chunkDataCategoryFiles[category] =[]
        for file in data[category]:         
            chunked = chunkList(file, chunkSize, overLapSize) 
            chunkDataCategoryFiles[category].append(chunked)
            for item in chunked:
                chunkDataAll.append(item)            
    return chunkDataAll, chunkDataCategoryFiles
            
def makeHistogramFeatureVector(kmeans, categoriesChunked):
    featureVector = []
    labelVector = []
    categoriesHistogram = {}
    for category in categoriesChunked:
        categoriesHistogram[category] = []
        for file in categoriesChunked[category]: 
            labelVector.append(category)
            kmeanPrediction = kmeans.predict(file)
            fileHis = [0] * kmeans.n_clusters
            for item in kmeanPrediction:
                fileHis[item] += 1
            categoriesHistogram[category].append(fileHis)
            featureVector.append(fileHis)
    return featureVector, labelVector, categoriesHistogram

def splitDataTrainTest(dataX, dataY, percentage):
    dataLen = len(dataX)
    testLen = round(percentage * dataLen)
    trainX = dataX.copy()
    trainY = dataY.copy()
    splitedX = []
    splitedY = []    
    for cross in range(int(1.0/percentage)-1):
        testX = []
        testY = []
        for k in range(testLen):
            i =  random.randrange(len(trainX))
            testX.append(trainX[i])
            testY.append(trainY[i])
            trainX.pop(i)    
            trainY.pop(i)
        splitedX.append(testX)
        splitedY.append(testY)  
    splitedX.append(trainX)
    splitedY.append(trainY)
    return splitedX, splitedY
                
def accuracy(predictedData, testData):
    correct = 0
    for i in range(len(testData)):
        if testData[i] ==  predictedData[i]:
            correct += 1
    return correct/float(len(testData))

def computeMean(data):
    meanData = {}
    for category in data:
        meanData[category] = sum(np.array(data[category])) / float(len(data[category]))
    return meanData

def drawBarChart(data, label):
    plt.bar(range(len(data)), list(data), align='center')    
    #plt.xticks(range(len(data)), list(data.keys()), rotation=90)    
    plt.xlabel('Cluster Centers')
    plt.ylabel('Frequency')   
    plt.title(label)  
    plt.show()    
    
    
#Main    
chunkSize = 16
ClusterSize = 500
overLapSize = int(0.75 * chunkSize)

allData = readData()
allChunksTogether, categoriesFilesChunked = makeChunks(allData, chunkSize, overLapSize)
kmeans = KMeans(n_clusters=ClusterSize, random_state=True).fit(allChunksTogether)
featureVector, labelVector, categoriesHistogram = makeHistogramFeatureVector(kmeans, categoriesFilesChunked)
crossValidationDataX, crossValidationDataY = splitDataTrainTest(featureVector, labelVector, 0.33)
clf=RandomForestClassifier(n_estimators=200)
#cross validation
accuracyData = []
for i in range(3):
    X_test = crossValidationDataX[i]
    Y_test = crossValidationDataY[i]
    X_train = []
    Y_train = []   
    for k in range(3):
        if k != i:
            X_train += crossValidationDataX[k] 
            Y_train += crossValidationDataY[k]
    clf.fit(X_train,Y_train)
    y_pred=clf.predict(X_test)
    accuracyData.append((accuracy(y_pred, Y_test)))

print("Accuracy = ", sum(accuracyData) / len(accuracyData))


meanCategoriesHistogram = computeMean(categoriesHistogram)
#with best parameters
#clf.fit(featureVector, labelVector)
Labels = list(meanCategoriesHistogram.keys())
cf = confusion_matrix(Y_test, y_pred, Labels)


