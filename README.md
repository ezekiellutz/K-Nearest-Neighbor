# K Nearest Neighbors Algorithm
A supervised machine learning algorithm used for classification labeling.

# Installation Instructions
Download both the .py and .csv files and place them in the same directory on your PC. Run the code from your IDE.

It is recommended that this script be run in Spyder 4.2.5 [conda (Python 3.8.8)]. 

# How the Algorithm Works
The K-nearest neighbors algorithm is a supervised machine learning algorithm. It is easy to implement but has the ability to perform rather challenging classification tasks. The K-nearest neighbors algorithm is what is often referred to as a "lazy learning algorithm" since it is ideally used when the data set is continuously updated with new entries. For this reason, it is extremely common for the K nearest neighbors algorithm to be used with online recommendation systems, like Amazon (people who purchased this item also bought...) or Netflix (people who watched this show also liked...), in which the dataset is always increasing. Because of this, there is little to no training phase with the K-nearest neighbors algorithm, since the training data set would quickly become obsolete. Finally, the K-nearest neighbors algorithm is non-parametric, meaning that it makes no assumptions about the underlying data. This is an extremely important feature, since most real world data does not tend to follow any theoretical assumption such as linear-separability, uniform distribution, etc.

The versatility of the K-nearest neighbor algorithm mentioned above does come at a cost. As already mentioned, the lack of a training phase in the algorithm tends to result in a lower accuracy for the overall algorithm. This is to be expected, as the algorithm is designed to process large datasets (that are continually increasing in size) and therefore have little ability to filter out the data prior to making assumptions with it. It is for this reason that the K-nearest neighbors algorithm should only be used in applications where accuracy of the data is not essential. The algorithm is more suited for predicting a person's spending habits, rather than predicting an incoming threat type on a military aircraft. 

# Description
This GitHub repository contains a Python script that utilizes the K-nearest neighbors algorithm in an interesting way. The algorithm used here is a combination of "lazy learning" and "eager learning" algorithms. Because the fictional dataset used here is not ever expanding (since it is pulling information from a .csv file rather than a database) a small amount of training and testing of the dataset could be implemented. The script here trains on 80% of the data and tests on the remaining 20%. 

The .csv file contains a plethora of information about a fictional customer database, such as: income, marital status, geographical region, education, age, and gender. This information is then coupled with the service plan these customers select out of four different plans. The script takes this information and uses it to train itself on how to best predict the service plan a new customer might pick based on the relevant information about the customer mentioend previously. The script will test different values of K (from 1 to 20) to see how many of the K-nearest neighbors should be used to accurately predict the service plan this new customer will select. The script produces a graph plotting the accuracy of the model versus the value of K and prints the most accurate result to the console. See below:



   ![Figure 2021-06-25 093238](https://user-images.githubusercontent.com/83550613/123440868-df1e9200-d598-11eb-80ce-e54dc759d26a.png)



    runfile('C:/example', wdir='C:/example')
    The algorithm had its highest accuracy of 0.36 with a k value of 16
