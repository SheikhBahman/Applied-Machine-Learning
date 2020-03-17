# Description

This homework focuses on implementing linear regression and using Cook's distance, leverage and standardized residuals to dectect outliers by R.

# Author
Bahman Sheikh

# Programming Language
R

# Data
At https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data, you will find the famous Boston Housing dataset. This consists of 506 data items. You will find the explanation of the dataset at https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names. Each is 13 measurements, and a house price. The data was collected by Harrison, D. and Rubinfeld, D.L in the 1970s (a date which explains the very low house prices). The dataset has been widely used in regression exercises, but seems to be waning in popularity. At least one of the independent variables measures the fraction of population nearby that is ''Black'' (their word, not mine). This variable appears to have had a significant effect on house prices then (and, sadly, may still now).


# Objectives

	1. Regress house price (variable 14) against all others, and use leverage, Cook's distance, and standardized residuals to find possible outliers. 
	2. Remove all points suspect as outliers, and compute a new regression. Reproduce a diagnostic plot.
	3. Apply a Box-Cox transformation (use boxcox command) to the dependent variable. Now transform the dependent variable, build a linear regression, and check the standardized residuals. If they look acceptable, produce a plot of fitted house price against true house price.

# Results
![GitHub Logo](/Outlier%20Detection/IMG/1.png)
![GitHub Logo](/Outlier%20Detection/IMG/2.png)
![GitHub Logo](/Outlier%20Detection/IMG/3.png)
![GitHub Logo](/Outlier%20Detection/IMG/4.png)


