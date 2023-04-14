#-------------------------------------------------------------------------
# AUTHOR: Kyle Just
# FILENAME: preceptron.py
# SPECIFICATION: A demonstration of a basic preceptron to learn a linearly separable dataset
# FOR: CS 4210- Assignment #4
# TIME SPENT: 20 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

highest_accuracy_preceptron = 0
highest_accuracy_MLP = 0

for rate_option in n: #iterates over n
    for shuffle_option in r: #iterates over r
        for algo in ["Preceptron", "MLP"]: #iterates over the algorithms

            #Create a Neural Network classifier
            if algo == "Perceptron":
               clf = Perceptron(eta0=rate_option, shuffle=shuffle_option, max_iter=1000)    #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
            else:
               clf = MLPClassifier(activation='logistic', learning_rate_init=rate_option, shuffle=shuffle_option, max_iter=1000) #use those hyperparameters: activation='logistic', learning_rate_init = learning rate, hidden_layer_sizes = number of neurons in the ith hidden layer,
            #                          shuffle = shuffle the training data, max_iter=1000
            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:

       
            accuracy_counter = 0
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                if clf.predict([x_testSample]) == y_testSample:
                    accuracy_counter += 1
            accuracy = accuracy_counter / len(X_test)

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            if accuracy > highest_accuracy_preceptron and algo == "Perceptron":
                highest_accuracy_preceptron = accuracy
                print("Highest Perceptron accuracy so far: " + str(highest_accuracy_preceptron) + ", Parameters: learning rate=" + str(rate_option) + ", shuffle=" + str(shuffle_option))
            elif accuracy > highest_accuracy_MLP and algo == "MLP":
                highest_accuracy_MLP = accuracy
                print("Highest MLP accuracy so far: " + str(highest_accuracy_MLP) + ", Parameters: learning rate=" + str(rate_option) + ", shuffle=" + str(shuffle_option))

            










