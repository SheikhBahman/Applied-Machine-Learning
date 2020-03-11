# Description
Without using libraries write the naive Bayes prediction code.
The distribution parameters can be calculated manually or via libraries.  Additionally, a libraries are used to load the MNIST data (e.g. python-mnist or scikit-learn) and to rescale the images (e.g. openCV).

Model each class of the dataset using a Normal distribution and (separately) a Bernoulli distribution for both untouched images v. stretched bounding boxes, using 20 x 20 for your bounding box dimension.  This should result in 4 total models.  Use the training set to calculate the distribution parameters.

# Author
Bahman Sheikh

# Programming Language
Python

# Objectives
- Develop classifiers from scratch (without using libraries).
- Classify MNIST using a decision forest.
- Compute the accuracy values for the four combinations of Normal v. Bernoulli distributions for both untouched images v. stretched bounding boxes.  Both the training and test set accuracy will be reported.
- For each digit, plot the mean pixel values calculated for the Normal distribution of the untouched images.  In Python, a library such as matplotlib should prove useful.

# Results
![GitHub Logo](/MNIST%20Image%20Classification/IMG/1.png)

![GitHub Logo](/MNIST%20Image%20Classification/IMG/2.png)
