# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:53:49 2019

@author: Bahman
"""

import numpy as np
import sklearn.decomposition
import matplotlib.pyplot as plt


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def readData():
    data = []
    dataLabel = []
    batchLabels = []
    for i in range(1,6):
        newData = unpickle('cifar-10-batches-py/data_batch_' + str(i))
        X = newData[b"data"]
        Y = newData[b"labels"]        
        for i in range(len(X)):
            data.append(X[i])
            dataLabel.append(int(Y[i]))
            
    newData = unpickle('cifar-10-batches-py/test_batch')
    X = newData[b"data"]
    Y = newData[b"labels"] 
    for i in range(len(X)):
        data.append(X[i])
        dataLabel.append(int(Y[i]))
    
    batchData = unpickle("cifar-10-batches-py/batches.meta")
    for item in batchData[b'label_names']:
        batchLabels.append(str(item)[2:-1])
        
    return data, dataLabel, batchLabels
        

def viewImage(imageData,imageLabel):
	img = imageData.reshape(3,32,32).transpose([1, 2, 0])
	plt.imshow(img)
	plt.title(imageLabel)
    
def seperateCategories(data, dataLabel, batchLabels):
    seperatedData = {}
    for i in range(len(data)):
        if batchLabels[dataLabel[i]] not in seperatedData:
            seperatedData[batchLabels[dataLabel[i]]] = []
            seperatedData[batchLabels[dataLabel[i]]].append(data[i])
        else:
            seperatedData[batchLabels[dataLabel[i]]].append(data[i])
    return seperatedData
'''
def mean(data):
    meanData = {}
    for key in data:
         sumation = [sum(i) for i in zip(*data[key])]
         meanData[key] = [i / len(data[key]) for i in sumation]
    return meanData'''
    
def reConstructImage(classImages, nP):    
    mean = np.mean(classImages, axis=0)
    pca = sklearn.decomposition.PCA()
    pca.fit(classImages)
    reConsImage = np.dot(pca.transform(classImages)[:,:nP], pca.components_[:nP,:])
    reConsImage += mean   
    #eigenVals = pca.explained_variance_
    eigenVecs = pca.components_
    #eigenValues, eigenVectors =  np.linalg.eig(np.cov(classImages-mean))
    #idx = eigenValues.argsort()[::-1]   
    #eigenVals = eigenValues[idx]
    #eigenVecs = eigenVectors[:,idx]
    return reConsImage, mean, eigenVecs

    
def reConstructImagesOfClasses(data, nPrincipal):
    reConsImageClasses = {}
    meanData = {}
    principalComp = {}
    for classImage in data:
        reConsImageClasses[classImage], meanData[classImage], principalComp[classImage] = reConstructImage(seperatedData[classImage], nPrincipal)
        
    return reConsImageClasses, meanData, principalComp
        
def error(data, reConsData):
    errorClass = {}
    for classImage in data:
        errorClass[classImage] = np.sum(np.power(data[classImage]-reConsData[classImage],2)) / len(data[classImage])
    return errorClass

def writeResults(filename, data):
    for i in range(len(data)):
        print(data[i], end="", file=filename) 
        if i != len(data) - 1:
            print(",", end="", file=filename) 
        else:
            print("\n", end="", file=filename)

def euclideanDistancePartB(data):
    
    with open("partb_distances.csv", "w") as csvFile:
        labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        euclideanDistance = []
        for label_i in labels:
            newRow = []
            for label_j in labels:
                newRow.append(np.sum(np.power(data[label_i]-data[label_j],2)))                
            euclideanDistance.append(newRow)
            writeResults(csvFile, newRow)
        csvFile.close()   
    return euclideanDistance
    
    
def drawBarChart(data):
    plt.bar(range(len(data)), list(data.values()), align='center')    
    plt.xticks(range(len(data)), list(data.keys()), rotation=90)    
    plt.show()
    

def myMDS(distanceSq):    
    N = len(distanceSq)
    A = np.identity(N) - 1/N * np.ones((N,N))
    W = -0.5 * np.matmul(np.matmul(A, distanceSq), A.T)   
    eig_vals, eig_vecs = np.linalg.eig(W)
    eig_pairs = [((eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    for i in range(len(eig_pairs)):
        if eig_pairs[i][0] < 0:
            eig_pairs[i] = (eig_pairs[i][0]*-1.0, eig_pairs[i][1]*-1.0)             
    eig_pairs.sort()
    eig_pairs.reverse()      
    A, U = zip(*eig_pairs)   
    As = A[:2]
    As = np.diag(As)
    As12 = np.sqrt(abs(As))    
    #As12 = np.sqrt(As) 
    Us = np.array(U)[:2,:]    
    Y = np.matmul(Us.T, As12)    
    return Y

def myPCA(dataA, dataB, meanA, nP): #for part C       
    
    '''
    PCA = sklearn.decomposition.PCA(n_components=nP)
    pcaB = PCA.fit(dataB)
    pcaA = PCA.fit(dataA)
    aa = pcaA.mean_
    pcaB.mean_ = pcaA.mean_
    reConsImage = aa + np.dot(pcaB.transform(dataA), pcaB.components_)'''    
       
    
    PCA = sklearn.decomposition.PCA(n_components=nP)
    pcaB = PCA.fit(dataB)
    pcaB.mean_ = np.mean(dataA, axis=0)
    A = pcaB.transform(dataA)
    reConsImage = pcaB.inverse_transform(A)    
    
    return reConsImage

       

def E_AtoB(data, meanData, eigenVecs):
    
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    EofAtoB = []
    for label_i in labels:
        newRow = []
        for label_j in labels:
            reConsImage = myPCA(data[label_i], data[label_j], meanData[label_i], 20)                
            newRow.append(np.sum(np.power(data[label_i]-reConsImage,2)) / len(data[label_i])) 
              
        EofAtoB.append(newRow)
    return EofAtoB

def EAB(EofAtoB):
    with open("partc_distances.csv", "w") as csvFile:
        EofAtoB_F = []    
        for A in range(len(EofAtoB)):
            newRow = []
            for B in range(len(EofAtoB[A])):
                newRow.append((EofAtoB[A][B] + EofAtoB[B][A])/2.0)
            writeResults(csvFile, newRow)
            EofAtoB_F.append(newRow)
            
        csvFile.close()
        return EofAtoB_F
        
            
def scatterPlotLabel(data, labels):
    x,y = zip(*data)
    fig, ax = plt.subplots()    
    plt.title("MDS graph")
    ax.scatter(x, y, alpha=1)    
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], y[i]))
    plt.show()

#//////////////////////////////////////////////////
#Main
data, dataLabel, batchLabels = readData()
#Example of data
#viewImage(data[500],batchLabels[dataLabel[500]])
print("Total data size = ", len(data))
print("Total data size = ", len(dataLabel))
seperatedData = seperateCategories(data, dataLabel, batchLabels)
#viewImage(seperatedData['deer'][100],'deer')
reConsImageClasses, meanData, eigenVecs = reConstructImagesOfClasses(seperatedData, 20)
#viewImage(np.array([int(i) for i in meanData['deer']]),"")
#viewImage(np.array([int(i) for i in reConsImageClasses['deer'][5]]),"")
errorClass = error(seperatedData, reConsImageClasses)
drawBarChart(errorClass)
distSq = euclideanDistancePartB(meanData)
MSD = myMDS(distSq)
labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
scatterPlotLabel(MSD, labels)
EAtoB = E_AtoB(seperatedData, meanData, eigenVecs)
D_EAB = EAB(EAtoB)
MSD = myMDS(D_EAB)
scatterPlotLabel(MSD, labels)














