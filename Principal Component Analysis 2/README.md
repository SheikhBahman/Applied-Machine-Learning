# Description

This homework focuses on familiarizing with low-rank approximations and multi-dimensional scaling. 

# Author
Bahman Sheikh

# Programming Language
Python

# Data
CIFAR-10 is a dataset of 32x32 images in 10 categories, collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. It is often used to evaluate machine learning algorithms. You can download this dataset from https://www.cs.toronto.edu/~kriz/cifar.html. I combine the test and train sets (all the images) and separate them by category.

# Objectives

## Part A
	1. For each class, find the mean image, and compute the first 20 principal components.
	2. Use the mean as well as the principle components to compute a low-dimensional reconstruction of each image in the class.
	3. For each image, compute the squared difference between the original and reconstructed version, and sum this over all pixels over all channels. 
	4. Plot the above value in the bar graph against its category/class label. 
## Part B
	1. Compute a 10 x 10 distance matrix D such that D[i,j] is the Euclidean distance between the mean images of class i and class j. Square the elements of this matrix and write it out to a CSV file named partb_distances.csv. 
	2. Perform multi-dimensional scaling with the squared distance matrix you have. Refer to the MDS section for details on how to do that.
	3. Plot the first component along the x-axis and component 2 along the y-axis of a scatter plot. You will submit this plot.
## Part C
	1. Like in Part B, first compute a 10 x 10 distance matrix. However, here, D[i,j] will contain E(i → j). Let's define E(A → B).
		○ E(A → B) = (E(A| B) + E(B|A))/2
		○ To compute E(A|B), use the mean image of class A and the first 20 principal components of class B to reconstruct the images of class A
		○ Use the procedure described in steps 3 and 4 of Part A to compute the mean of the sum of pixel-wise squared difference between the reconstructed and original images.
		○ Similarly compute E(B|A).
	2. Once we have computed D, write it out to a CSV file named partc_distances.csv.
	3. Perform MDS with this distance matrix, plot the first component along the x-axis and component 2 along the y-axis of a scatter plot. 
## Principal Coordinate Analysis (MDS)


# Results
![GitHub Logo](/Principal%20Component%20Analysis%202/IMG/1.png)
![GitHub Logo](/Principal%20Component%20Analysis%202/IMG/2.png)


