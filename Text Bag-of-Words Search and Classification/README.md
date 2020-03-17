# Description

This homework focuses on implementing a bag-of-word based pipeline to retrieve similar documents and classify reviews. Specifically, text retrieval by simple nearest neighbor queries and classification of using Logistic regression.

# Author
Bahman Sheikh

# Programming Language
Python

# Data
At http://courses.engr.illinois.edu/cs498aml/sp2019/homeworks/yelp_2k.csvyou will find a dataset of Yelp reviews. Theoriginal dataset contains 5,261,668 reviews and we select 2000 from them, where half of them for reviews with 1 and 5 stars respectively.

# Objectives

## Part A 
	1. Download and import the dataset. And then extract text and stars columns as X (data) and y (label).
	2. Convert the text into lower case then into bag-of-words representation. Without using a pre-existing list of stop-words.
	3. Bag-of-words Analysis and Repreprocessing.
		○ Graph the distribution of words counts vs word rank.
		○ Identify the set of common stop words by looking at the words. 
		○ Choose a max document frequency theshold for word occurances and a minimum word occurance to cull the less useful words.
		○ Reprocess the data using the stop-words list determined, the max document frequency and the minimum word occurance.
		○ Graph the updated words counts vs word rank.
After removing stop-words, convert all the data into bag-of-words vectors for use in the next part.

## Part B: Text-Retrieval
	1. Using nearest neighbor with a cos-distance metric, find 5 reviews matching Horrible customer service.
	2. Print the original text from the review along with the associated distance score. 
  
## Part C: Classification with Logistic Regression
	1. Separate the data in train and test sets. Use 10% of the data for test.
	2. Create a classifier based on Logistic Regression.
	3. The accuracy on the training set and the test set of the classifier?
	4. 
		○ Plot a histogram of the scores on the training data.		
		○ Choose a new threshold based on the plot and report the accuracy on the training set and the test set. 
		○ Plot the ROC curve for your classifier.
		○ At what false positive rate would the classifier minimize false positives while maximizing true positives?
# Results
![GitHub Logo](/Text%20Bag-of-Words%20Search%20and%20Classification/IMG/1.png)
![GitHub Logo](/Text%20Bag-of-Words%20Search%20and%20Classification/IMG/2.png)
![GitHub Logo](/Text%20Bag-of-Words%20Search%20and%20Classification/IMG/3.png)
![GitHub Logo](/Text%20Bag-of-Words%20Search%20and%20Classification/IMG/4.png)
![GitHub Logo](/Text%20Bag-of-Words%20Search%20and%20Classification/IMG/5.png)
![GitHub Logo](/Text%20Bag-of-Words%20Search%20and%20Classification/IMG/6.png)


