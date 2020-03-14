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

def accuracy(predictedData, testDataLabels):
    correct = 0
    for i in range(len(testDataLabels)):
        if testDataLabels[i][0] ==  predictedData[i]:
            correct += 1
    return correct/float(len(predictedData))

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

def filterImageThreshold(allImages, threshold):
    
    for image in allImages:
        for i in range(len(image)):
            for j in range(len(image[0])):
                if( image[i][j] > threshold):
                    image[i][j] = 1.0
                else:
                    image[i][j] = 0.0
    return allImages

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
       
def allImageStretched(images):
    newImages = []
    for image in images:
        newImages.append(imageStretched(image))
    
    return newImages

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

print("Stretched bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
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
