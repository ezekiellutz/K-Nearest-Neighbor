'''
Author: Ezekiel Lutz
Contact Info: xxZeke77xx@gmail.com
Time: 12:58 UTC
Date: 06-21-2021
Data Source: https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv
Recommended IDE: Spyder 4.2.5 [conda (Python 3.8.8)]
NOTE: Compatability issues with this code are known to exist, please use the IDE listed above.
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

#opens the .csv file
with open ('teleCust1000t.csv','r') as csv_file:
    df = pd.read_csv(csv_file)

#selects the feature set to be used for KNN
features = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values.astype(float)

#defines the dependent variable
y = df['custcat'].values

#defines the independent variables and normalizes them (necessary for KNN)
x = preprocessing.StandardScaler().fit(features).transform(features.astype(float))

#defines the training and testing set for the dependent and independent variables
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=4)

#defines variables to be used when plotting the accuracy of different k values
Ks = 20
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

#calculates the accuracy of the KNN algorithm for different values of k
for n in range(1,Ks):
     
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(x_train,y_train)
    predict = knn.predict(x_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, predict) 
    std_acc[n-1]=np.std(predict==y_test)/np.sqrt(predict.shape[0])

#plots the calculated accuracy for each value of k and announces the value with the highest accuracy
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()
print('The algorithm had its highest accuracy of', mean_acc.max(), 'with a k value of', mean_acc.argmax()+1) 
print('\n', confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))
