# Description

This homework focuses on vector quantization and classification. More specifically, 1) data slicing, 2) vector clustering, 3) making histograms, and 4) building a multi-class classifer. 

# Author
Bahman Sheikh

# Programming Language
Python

# Data
Obtain the actitivities of daily life dataset from the UC Irvine machine learning website (https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer, data provided by Barbara Bruno, Fulvio Mastrogiovanni and Antonio Sgorbissa).

# Objectives

## Part A 
Build a classifier that classifies sequences into one of the 14 activities provided and evaluate its performance using average accuracy over 3 fold cross validation. 
To do the cross validation, divide the data for each class into 3 folds separately. Then, for a given run select 2 folds from each class for training and use the remaining fold from each class for test. To make features, vector quantize, then use a histogram of cluster center. 
(i) the average error rate over 3 fold cross validation and (ii) the class confusion matrix of classifier for the fold with the lowest error.

## Part A 
See if we can improve your classifier by (i) modifying the number of cluster centers in the hierarchical k-means and (ii) modifying the size of the fixed length samples.

# Results
![GitHub Logo](/Vector%20quantization%20and%20classification/IMG/1.png)
![GitHub Logo](/Vector%20quantization%20and%20classification/IMG/2.png)
![GitHub Logo](/Vector%20quantization%20and%20classification/IMG/3.png)


