# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 19:10:11 2019

@author: Bahman
"""

from struct import unpack
import gzip
from numpy import zeros, uint8, float32
from pylab import imshow, show, cm
import math 
from sklearn.ensemble import RandomForestClassifier
import cv2
#From Martin Thoma
#From https://martin-thoma.com/classify-mnist-with-pybrain/
def get_labeled_data(imagefile, labelfile):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    # Open the images with gzip in read binary mode
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')

    # Read the binary data

    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    x = zeros((N, rows, cols), dtype=float32)  # Initialize numpy array
    y = zeros((N, 1), dtype=uint8)  # Initialize numpy array
    for i in range(N):
        if i % 1000 == 0:
            print("i: %i" % i)
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  # Just a single byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return (x, y)

def view_image(image, label=""):
    """View a single image."""
    print("Label: %s" % label)
    imshow(image, cmap=cm.gray)
    show()
 
def filterImageThreshold(allImages, threshold):
    
    for image in allImages:
        for i in range(len(image)):
            for j in range(len(image[0])):
                if( image[i][j] > threshold):
                    image[i][j] = 1.0
                else:
                    image[i][j] = 0.0
    return allImages
    
# array of labels -> dictionary of unique labels and positions(undex)    
def countLabels(labels):
    labelsAndPositions = {}
    for i in range(len(labels)):
        if labels[i][0] in labelsAndPositions:
            labelsAndPositions[labels[i][0]].append(i)
        else:
            labelsAndPositions[labels[i][0]] = [i]
    return labelsAndPositions

#image data, list of images index -> dictionary of 0-9 and Normal parameter
# calculate the mean and variance of each pixel of each image
def calculateNormalParametersOfImagePix(imageData, labelsAndPositions):
    imageNoramlProp = {}
    for imageCategory, allImageIndex in labelsAndPositions.items():
        meanMatrix = [[0.0 for x in range(len(imageData[0]))] for y in range(len(imageData[0][0]))] 
        varianceMatrix = [[0.0 for x in range(len(imageData[0]))] for y in range(len(imageData[0][0]))]         
        
        #mean matrix over all similar images
        for index in allImageIndex:
            for i in range(len(imageData[index])):
                for j in range(len(imageData[index][0])):
                     meanMatrix[i][j] += imageData[index][i][j]                     
        meanMatrix = [[float(j)/float(len(allImageIndex)) for j in i] for i in meanMatrix]
        
        #variance matrix over all similar images     
        for index in allImageIndex:
            for i in range(len(imageData[index])):
                for j in range(len(imageData[index][0])):
                    varianceMatrix[i][j] += pow(imageData[index][i][j]-meanMatrix[i][j],2)
        varianceMatrix = [[float(j)/(len(allImageIndex)-1) for j in i] for i in varianceMatrix]
        
        imageNoramlProp[imageCategory] = (meanMatrix, varianceMatrix)
        
    return imageNoramlProp

#image data, list of images index -> dictionary of 0-9 and Bernoulli matrix for each
# calculate the Bernoulli metrix of each image
def calculateBernoulliParametersOfImagePix(imageData, labelsAndPositions):
    imageBernoulliProp = {}
    for imageCategory, allImageIndex in labelsAndPositions.items():
        BernoulliMatrix = [[0.0 for x in range(len(imageData[0]))] for y in range(len(imageData[0][0]))] 
        for index in allImageIndex:
            for i in range(len(imageData[index])):
                for j in range(len(imageData[index][0])):
                    BernoulliMatrix[i][j] += imageData[index][i][j]
        BernoulliMatrix = [[float(j)/float(len(allImageIndex)) for j in i] for i in BernoulliMatrix]            
        imageBernoulliProp[imageCategory] = BernoulliMatrix
    return imageBernoulliProp
 
#calculate image category(0-9) probability
def imageProbability(labelsAndPositions, totalImageNumber):
    groupProb = {}
    for imageLabel in labelsAndPositions:
        groupProb[imageLabel] = len(labelsAndPositions[imageLabel]) / float(totalImageNumber)
    return groupProb    
    

#calculate probability from normal distribution
def normalProbability(x, average, variance):
    p = 0.00000001
    first = 1.0 / (math.sqrt(2.0 * math.pi * variance))
    p += first * math.exp(-(x-average)*(x-average)/(2.0*variance))     
    return p
#predicting the testImage label by normal dist., groupProb is the probabilities of labels
def predict(imageNoramlProp, groupProb, testImage):
    allImageProbability = {} 
    for imageCat in imageNoramlProp:
        allImageProbability[imageCat] = math.log(groupProb[imageCat]) #log(p(y))
        
        for i in range(len(testImage)):
            for j in range(len(testImage[0])):
                if imageNoramlProp[imageCat][1][i][j] > 0: # variance > 0
                    allImageProbability[imageCat] += math.log(normalProbability(testImage[i][j], imageNoramlProp[imageCat][0][i][j], imageNoramlProp[imageCat][1][i][j]))
                
    maxP = -10000.0    
    groupPredicted = None
    for imageCat in allImageProbability:
        if groupPredicted == None or allImageProbability[imageCat] > maxP:
            maxP = allImageProbability[imageCat]
            groupPredicted = imageCat
        
    return groupPredicted

#predicting the testImage label by Bernoulli., groupProb is the probabilities of labels
def predictByBernoulli(imageBernoulliProp, groupProb, testImage):
    allImageProbability = {} 
    for imageCat in imageNoramlProp:
        allImageProbability[imageCat] = math.log(groupProb[imageCat]) #log(p(y))
        
        for i in range(len(testImage)):
            for j in range(len(testImage[0])):
                if testImage[i][j] > 0:
                    allImageProbability[imageCat] += math.log(0.00000001 + imageBernoulliProp[imageCat][i][j])
                else:
                    allImageProbability[imageCat] += math.log(1 - imageBernoulliProp[imageCat][i][j])
                
    maxP = -10000.0    
    groupPredicted = None
    for imageCat in allImageProbability:
        if groupPredicted == None or allImageProbability[imageCat] > maxP:
            maxP = allImageProbability[imageCat]
            groupPredicted = imageCat
        
    return groupPredicted
            
        
def predictions(imageNoramlProp, groupProb, testData):
    predictions = []
    for testImage in testData:
        predictions.append(predict(imageNoramlProp, groupProb, testImage))
    return predictions  

def predictionsByBernoulli(imageBernoulliProp, groupProb, testData):
    predictions = []
    for testImage in testData:
        predictions.append(predictByBernoulli(imageBernoulliProp, groupProb, testImage))
    return predictions   

#calculate the accuracy
def accuracy(predictedData, testDataLabels):
    correct = 0
    for i in range(len(testDataLabels)):
        if testDataLabels[i][0] ==  predictedData[i]:
            correct += 1
    return correct/float(len(predictedData))

#change data format suitable for random forest
def changeDataFormat(x, y):
    xData = []
    for image in range(len(x)):
        newImage = []
        for i in range(len(x[image])):
            for j in range(len(x[image][i])):
                newImage.append(x[image][i][j])
        xData.append(newImage)
        
    yData = []
    for imageL in range(len(y)):
        yData.append(y[imageL][0])
    
    return xData, yData

#crop and resize an image    
def imageStretched(image):
    minI = 50
    minJ = 50
    maxI = 0
    maxJ = 0
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] > 0.0:
                if i < minI:
                    minI = i
                if j < minJ:
                    minJ = j
                     
                if i > maxI:
                    maxI = i
                if j > maxJ:
                    maxJ = j
     
    newImage = image[minI:maxI, minJ:maxJ]
    return cv2.resize(newImage, (20, 20)) 

#crop and resize all images      
def allImageStretched(images):
    newImages = []
    for image in images:
        newImages.append(imageStretched(image))
    
    return newImages


#////////////////////////////////////////////////////////    
#////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////
#Main    
imagefile = "train-images-idx3-ubyte.gz"    
labelfile = "train-labels-idx1-ubyte.gz"
imagefileTest = "t10k-images-idx3-ubyte.gz"    
labelfileTest = "t10k-labels-idx1-ubyte.gz"
x, y = get_labeled_data(imagefile, labelfile)
testData, testDataLabels = get_labeled_data(imagefileTest, labelfileTest)

x = filterImageThreshold(x, 127)
testData = filterImageThreshold(testData, 127)

labelsAndPositions = countLabels(y)
groupProb = imageProbability(labelsAndPositions, len(x))
imageNoramlProp = calculateNormalParametersOfImagePix(x, labelsAndPositions)
imageBernoulliProp = calculateBernoulliParametersOfImagePix(x, labelsAndPositions)

predictedData = predictions(imageNoramlProp, groupProb, testData)
predictedDataBernoulli = predictionsByBernoulli(imageBernoulliProp, groupProb, testData)
print("NormalDis accuracy: ", accuracy(predictedData, testDataLabels))
print("Bernoulli accuracy: ", accuracy(predictedDataBernoulli, testDataLabels))

predictedData = predictions(imageNoramlProp, groupProb, x)
predictedDataBernoulli = predictionsByBernoulli(imageBernoulliProp, groupProb, x)
print("NormalDis accuracy training data: ", accuracy(predictedData, y))
print("Bernoulli accuracy training data: ", accuracy(predictedDataBernoulli, y))

x = allImageStretched(x)
testData = allImageStretched(testData)

labelsAndPositions = countLabels(y)
groupProb = imageProbability(labelsAndPositions, len(x))
imageNoramlProp = calculateNormalParametersOfImagePix(x, labelsAndPositions)
imageBernoulliProp = calculateBernoulliParametersOfImagePix(x, labelsAndPositions)

predictedData = predictions(imageNoramlProp, groupProb, testData)
predictedDataBernoulli = predictionsByBernoulli(imageBernoulliProp, groupProb, testData)
print("NormalDis accuracy stretched: ", accuracy(predictedData, testDataLabels))
print("Bernoulli accuracy stretched: ", accuracy(predictedDataBernoulli, testDataLabels))

predictedData = predictions(imageNoramlProp, groupProb, x)
predictedDataBernoulli = predictionsByBernoulli(imageBernoulliProp, groupProb, x)
print("NormalDis accuracy stretched training data: ", accuracy(predictedData, y))
print("Bernoulli accuracy stretched training data: ", accuracy(predictedDataBernoulli, y))


print("Random Forest: ")
print("Untouched data: ")

imagefile = "train-images-idx3-ubyte.gz"    
labelfile = "train-labels-idx1-ubyte.gz"
imagefileTest = "t10k-images-idx3-ubyte.gz"    
labelfileTest = "t10k-labels-idx1-ubyte.gz"
x, y = get_labeled_data(imagefile, labelfile)
testData, testDataLabels = get_labeled_data(imagefileTest, labelfileTest)

x = filterImageThreshold(x, 127)
testData = filterImageThreshold(testData, 127)

xData, yData = changeDataFormat(x, y)
xData_t, yData_t = changeDataFormat(testData, testDataLabels)


clf=RandomForestClassifier(n_estimators=10, max_depth=4)
clf.fit(xData,yData)
pred=clf.predict(xData_t)
print("tree 10 depth4 test", accuracy(pred, testDataLabels))
pred=clf.predict(xData)
print("tree 10 depth4 train", accuracy(pred, y))


clf=RandomForestClassifier(n_estimators=10, max_depth=16)
clf.fit(xData,yData)
pred=clf.predict(xData_t)
print("tree 10 depth16 test", accuracy(pred, testDataLabels))
pred=clf.predict(xData)
print("tree 10 depth16 train", accuracy(pred, y))


clf=RandomForestClassifier(n_estimators=30, max_depth=4)
clf.fit(xData,yData)
pred=clf.predict(xData_t)
print("tree 30 depth4 test", accuracy(pred, testDataLabels))
pred=clf.predict(xData)
print("tree 30 depth4 train", accuracy(pred, y))


clf=RandomForestClassifier(n_estimators=30, max_depth=16)
clf.fit(xData,yData)
pred=clf.predict(xData_t)
print("tree 30 depth16 test", accuracy(pred, testDataLabels))
pred=clf.predict(xData)
print("tree 30 depth16 train", accuracy(pred, y))


print("Random Forest: ")
print("Stretched data: ")

imagefile = "train-images-idx3-ubyte.gz"    
labelfile = "train-labels-idx1-ubyte.gz"
imagefileTest = "t10k-images-idx3-ubyte.gz"    
labelfileTest = "t10k-labels-idx1-ubyte.gz"
x, y = get_labeled_data(imagefile, labelfile)
testData, testDataLabels = get_labeled_data(imagefileTest, labelfileTest)

x = filterImageThreshold(x, 127)
testData = filterImageThreshold(testData, 127)

x = allImageStretched(x)
testData = allImageStretched(testData)

xData, yData = changeDataFormat(x, y)
xData_t, yData_t = changeDataFormat(testData, testDataLabels)

print("Random Forest: ")
print("Untouched data: ")
clf=RandomForestClassifier(n_estimators=10, max_depth=4)
clf.fit(xData,yData)
pred=clf.predict(xData_t)
print("tree 10 depth4 test", accuracy(pred, testDataLabels))
pred=clf.predict(xData)
print("tree 10 depth4 train", accuracy(pred, y))


clf=RandomForestClassifier(n_estimators=10, max_depth=16)
clf.fit(xData,yData)
pred=clf.predict(xData_t)
print("tree 10 depth16 test", accuracy(pred, testDataLabels))
pred=clf.predict(xData)
print("tree 10 depth16 train", accuracy(pred, y))


clf=RandomForestClassifier(n_estimators=30, max_depth=4)
clf.fit(xData,yData)
pred=clf.predict(xData_t)
print("tree 30 depth4 test", accuracy(pred, testDataLabels))
pred=clf.predict(xData)
print("tree 30 depth4 train", accuracy(pred, y))


clf=RandomForestClassifier(n_estimators=30, max_depth=16)
clf.fit(xData,yData)
pred=clf.predict(xData_t)
print("tree 30 depth16 test", accuracy(pred, testDataLabels))
pred=clf.predict(xData)
print("tree 30 depth16 train", accuracy(pred, y))



