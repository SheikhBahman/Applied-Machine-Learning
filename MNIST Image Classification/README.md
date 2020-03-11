# Description
Without using libraries write the naive Bayes prediction code.
The distribution parameters can be calculated manually or via libraries.  Additionally, a libraries are used to load the MNIST data (e.g. python-mnist or scikit-learn) and to rescale the images (e.g. openCV).

Model each class of the dataset using a Normal distribution and (separately) a Bernoulli distribution for both untouched images v. stretched bounding boxes, using 20 x 20 for your bounding box dimension.  This should result in 4 total models.  Use the training set to calculate the distribution parameters.

# Author
Bahman Sheikh

# Programming Language
Python

# Data
The MNIST dataset is a dataset of 60,000 training and 10,000 test examples of handwritten digits, originally constructed by Yann Lecun, Corinna Cortes, and Christopher J.C. Burges. It is very widely used to check simple methods. There are 10 classes in total ("0" to "9"). 

The dataset consists of 28 x 28 images. These were originally binary images, but appear to be grey level images as a result of some anti-aliasing. I will ignore mid-grey pixels (there aren't many of them) and call dark pixels "ink pixels", and light pixels "paper pixels"; The digit has been centered in the image by centering the center of gravity of the image pixels, but as mentioned on the original site, this is probably not ideal. Here are some options for re-centering the digits that I will refer to in the exercises.
	• Untouched: Do not re-center the digits, but use the images as is.
	• Bounding box: Construct a 20 x 20 bounding box so that the horizontal (resp. vertical) range of ink pixels is centered in the box.
	• Stretched bounding box: Construct a 20 x 20 bounding box so that the horizontal (resp. vertical) range of ink pixels runs the full horizontal (resp. vertical) range of the box. Obtaining this representation will involve rescaling image pixels: you find the horizontal and vertical ink range, cut that out of the original image, then resize the result to 20 x 20. Once the image has been re-centered, you can compute features.

# Objectives
- Develop classifiers from scratch (without using libraries).
- Classify MNIST using a decision forest.
- Compute the accuracy values for the four combinations of Normal v. Bernoulli distributions for both untouched images v. stretched bounding boxes.  Both the training and test set accuracy will be reported.
- For each digit, plot the mean pixel values calculated for the Normal distribution of the untouched images.  In Python, a library such as matplotlib should prove useful.

# Results
![GitHub Logo](/MNIST%20Image%20Classification/IMG/1.png)

![GitHub Logo](/MNIST%20Image%20Classification/IMG/2.png)
