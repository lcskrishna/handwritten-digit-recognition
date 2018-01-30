
# coding: utf-8

# In[2]:

#### Assignment2
#### This script includes the implementation of the Neural Network for the Digit Recognition.
#### It takes the following libararies - CSV, Numpy, Sklearn.Neural_Network, sklearn and Math.
### @Author: Chaitanya Sri Krishna Lolla, Student ID: 800960353

## Libraries Used.
import csv
import numpy as np
from sklearn.neural_network import MLPClassifier
import math
from sklearn import metrics

## Form the Training Input data into the variable X.
## Form the Training Target Value into the variable Y.
## Takes the Training CSV file using the CSV libarary and forms the training input and output data X.
with open('optdigits_raining.csv') as trainingFile:
    reader = csv.reader(trainingFile)
    
    X=[]
    Y=[]
    
    for row in reader:
        X.append(row[:64])
        Y.append(row[64])

print("Done with loading the training data.")

## Cross validation is being done on two folds of the data 70-30% Split. 
## The first fold of the data is being given as X_train, Y_train.
## The Second fold called as Validation dataset is given as X_validation, Y_validation.
length_TrainingSet = len(X)

percentage_training = 0.7
    
len_train = math.floor(length_TrainingSet * percentage_training);
    
X_train = X[:len_train]
Y_train = Y[:len_train]

## This step is being done to avoid type casting errors while giving to the MLPClassifier.
for i in range(0,len(X_train)):
    lst = X_train[i]
    for j in range(0,len(lst)):
        lst[j] = float(int(lst[j]))
    X_train[i] = lst
for i in range(0,len(Y_train)):
    Y_train[i] = float(int(Y_train[i]))

print("Done with forming the Training Data.")

X_validation = X[len_train:len(X)]
Y_validation = Y[len_train:len(Y)]


for i in range(0,len(X_validation)):
    lst = X_validation[i]
    for j in range(0,len(lst)):
        lst[j] = float(int(lst[j]))
    X_validation[i] = lst
for i in range(0,len(Y_validation)):
    Y_validation[i] = float(int(Y_validation[i]))


print("Done with forming the Validation dataset.")



# In[3]:

## Classification of the above training dataset using Neural Networks (MLPClassifier())
## Default Parameters have been used here.
## Iterations = 200, Hidden Layers = 600, learning rate = 0.001, solver = 'SGD- Stochastic Gradient Descent', Activation Layer = 'RELU'.
clf = MLPClassifier(hidden_layer_sizes = (600,), activation='relu', solver='sgd', alpha = 0.0001, verbose=True, learning_rate = 'constant',learning_rate_init= 0.001,max_iter=200)
clf = clf.fit(X_train,Y_train)
print("Classification is Done.")


# In[4]:

## Training data prediction for verifying the training.
output_Predicted = clf.predict(X_train);
accuracy_training = metrics.accuracy_score(output_Predicted,Y_train)
print("Accuracy on the Training Data set:")
print(accuracy_training* 100)


# In[5]:

## Calculating the accuracy on the validation dataset.
output_predicted_validation = clf.predict(X_validation)
accuracy_validation = metrics.accuracy_score(output_predicted_validation,Y_validation)
print("Accuracy on the Validation Data set is : ")
print(accuracy_validation * 100)


# In[11]:

### Formation of the given Test Dataset and verifying it using the above classifier trained.
## Formation of Testing Data:
with open('optdigits_test.csv') as testingFile:
    reader = csv.reader(testingFile)
    
    X_test=[]
    Y_test=[]
    
    for row in reader:
        X_test.append(row[:64])
        Y_test.append(row[64])

## This step is used to avoid any type casting errors that have been occured.
for i in range(0,len(X_test)):
    lst = X_test[i]
    for j in range(0,len(lst)):
        lst[j] = float(int(lst[j]))
    X_test[i] = lst
for j in range(0,len(Y_test)):
    Y_test[j] = float(int(Y_test[j]))

print("Done forming the Testing Dataset.")

## Calculation of the accuracy on the Testing dataset.
output_predicted_testing = clf.predict(X_test)
accuracy_testing = metrics.accuracy_score(output_predicted_testing, Y_test)
print("Accuracy on the Testing Dataset is : ")
print(accuracy_testing*100)


# In[ ]:



