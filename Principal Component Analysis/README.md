# Description

The goal of this homework is to use PCA to smooth the noise in the provided data. 

We have five noisy versions of the Iris dataset, and a noiseless version. For each of the 5 noisy data sets, we should compute the principal components in two ways. In the first, use the mean and covariance matrix of the noiseless dataset. In the second, use the mean and covariance of the respective noisy datasets. Based on these components, I computed the mean squared error between the noiseless version of the dataset and each of a PCA representation using 0 (i.e. every data item is represented by the mean), 1, 2, 3, and 4 principal components. The mean squared error here should compute the sum of the squared errors over the features and compute the mean of this over the rows.

# Author
Bahman Sheikh

# Programming Language
Python

# Output
- A csv file showing the numbers filled in a table set out as below, where "N" columns represents the components calculated via the noiseless dataset and the "c" columns of the noisy datasets.
Example: The entry corresponding to Dataset I and 2N should contain the mean squared error between the noiseless version of the dataset and the PCA representation of Dataset I, using 2 principal components computed from the mean and covariance matrix of the noiseless dataset.		
The first part, with "N" columns asks to reconstruct the noisy datasets using the PCs of the noiseless dataset. 
The second part, with "c" columns asks to reconstruct the noisy datasets using the PCs of the noisy dataset.
- A csv file containing reconstruction of Dataset I ("dataI.csv"), expanded onto 2 principal components, where mean and principal components are computed from Dataset I.



