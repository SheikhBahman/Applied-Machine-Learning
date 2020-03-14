# -*- coding: utf-8 -*-
"""
Created on Sun Feb 8 11:36:06 2019

@author: Bahman
"""

import numpy as np
import csv

def readMyCSVData(fileName):
    with open(fileName, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        data = []
        for row in reader:
            newRow = []
            for item in row:
                newRow.append(float(item))
            data.append(newRow)
                                
        csvfile.close()
        return data
def myPCA(data, meanMat, covMat, nP):
    eig_vals, eig_vecs = np.linalg.eig(covMat)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs.sort()
    eig_pairs.reverse()    
    X_std = data - meanMat
    scoreV = np.matmul(X_std, eig_vecs)
    reConsData = meanMat + np.dot(scoreV[:,:nP], eig_vecs.T[:nP,:])
    
    return reConsData
    
    
def writeResults(filename, data):
    for i in range(len(data)):
        print(data[i], end="", file=filename) 
        if i != len(data) - 1:
            print(",", end="", file=filename) 
        else:
            print("\n", end="", file=filename)
            
def writeResults2(filename, data):    
    for i in range(len(data)):
        writeResults(filename, data[i])
        
def meanSquaredError(data, estimatedData):
    errorV = 0.0
    for i in range(len(data)):
        featureV = data[i] - estimatedData[i]
        errorV = errorV + sum([k ** 2 for k in featureV])
    return errorV / len(data)


#//Main

with open("bahmans2-numbers.csv", "w") as numbersFile:
    print("0N,1N,2N,3N,4N,0c,1c,2c,3c,4c", file=numbersFile) 
    
    with open("bahmans2-recon.csv", "w") as reconFile: 
        print("\"Sepal.Length\",\"Sepal.Width\",\"Petal.Length\",\"Petal.Width\"", file=reconFile) 
        
        iris    = np.array(readMyCSVData("iris.csv"))
        dataI   = np.array(readMyCSVData("dataI.csv"))
        dataII  = np.array(readMyCSVData("dataII.csv"))
        dataIII = np.array(readMyCSVData("dataIII.csv"))
        dataIV  = np.array(readMyCSVData("dataIV.csv"))
        dataV   = np.array(readMyCSVData("dataV.csv"))
        
        mean_mat_iris    = np.mean(iris, axis=0)
        mean_mat_dataI   = np.mean(dataI, axis=0)
        mean_mat_dataII  = np.mean(dataII, axis=0)
        mean_mat_dataIII = np.mean(dataIII, axis=0)
        mean_mat_dataIV  = np.mean(dataIV, axis=0)
        mean_mat_dataV   = np.mean(dataV, axis=0)
        
        cov_mat_iris    = (iris    - mean_mat_iris).T.dot((iris       - mean_mat_iris)) / (iris.shape[0])
        cov_mat_dataI   = (dataI   - mean_mat_dataI).T.dot((dataI     - mean_mat_dataI)) / (dataI.shape[0])
        cov_mat_dataII  = (dataII  - mean_mat_dataII).T.dot((dataII   - mean_mat_dataII)) / (dataII.shape[0])
        cov_mat_dataIII = (dataIII - mean_mat_dataIII).T.dot((dataIII - mean_mat_dataIII)) / (dataIII.shape[0])
        cov_mat_dataIV  = (dataIV  - mean_mat_dataIV).T.dot((dataIV   - mean_mat_dataIV)) / (dataIV.shape[0])
        cov_mat_dataV   = (dataV   - mean_mat_dataV).T.dot((dataV     - mean_mat_dataV)) / (dataV.shape[0])
        
        
        result = []
        for landa in range(5):
            xi_ = myPCA(dataI, mean_mat_iris, cov_mat_iris, landa)
            result.append(meanSquaredError(iris, xi_))             
        for landa in range(5):
            xi_ = myPCA(dataI, mean_mat_dataI, cov_mat_dataI, landa)
            result.append(meanSquaredError(iris, xi_)) 
        writeResults(numbersFile, result)
        
        result = []
        for landa in range(5):
            xi_ = myPCA(dataII, mean_mat_iris, cov_mat_iris, landa)
            result.append(meanSquaredError(iris, xi_))             
        for landa in range(5):
            xi_ = myPCA(dataII, mean_mat_dataII, cov_mat_dataII, landa)
            result.append(meanSquaredError(iris, xi_)) 
        writeResults(numbersFile, result)
        
        result = []
        for landa in range(5):
            xi_ = myPCA(dataIII, mean_mat_iris, cov_mat_iris, landa)
            result.append(meanSquaredError(iris, xi_))             
        for landa in range(5):
            xi_ = myPCA(dataIII, mean_mat_dataIII, cov_mat_dataIII, landa)
            result.append(meanSquaredError(iris, xi_)) 
        writeResults(numbersFile, result)
        
        result = []
        for landa in range(5):
            xi_ = myPCA(dataIV, mean_mat_iris, cov_mat_iris, landa)
            result.append(meanSquaredError(iris, xi_))             
        for landa in range(5):
            xi_ = myPCA(dataIV, mean_mat_dataIV, cov_mat_dataIV, landa)
            result.append(meanSquaredError(iris, xi_)) 
        writeResults(numbersFile, result)
        
        result = []
        for landa in range(5):
            xi_ = myPCA(dataV, mean_mat_iris, cov_mat_iris, landa)
            result.append(meanSquaredError(iris, xi_))             
        for landa in range(5):
            xi_ = myPCA(dataV, mean_mat_dataV, cov_mat_dataV, landa)
            result.append(meanSquaredError(iris, xi_)) 
        writeResults(numbersFile, result)        
        
        
        xi_ = myPCA(dataI, mean_mat_dataI, cov_mat_dataI, 2)
        writeResults2(reconFile, xi_)
        numbersFile.close()
        reconFile.close()


      