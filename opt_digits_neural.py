#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 22:00:02 2017

@author: gowthamkommineni
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


train_data = pd.read_csv('optdigits_raining.csv',header=None)
test_data=pd.read_csv('optdigits_test.csv',header=None)

#train_input=df[df.columns[2]]


train_input=train_data.iloc[:,:64].as_matrix()
train_output=train_data.iloc[:,64].as_matrix()

print("size of train input",train_input.shape)
print("size of train output",train_output.shape)

test_input=test_data.iloc[:,:64].as_matrix()
test_output=test_data.iloc[:,64].as_matrix()

print("size of test input",test_input.shape)
print("size of test output",test_output.shape)


# different solver
solver_params = [{'solver': 'lbfgs'},{'solver': 'sgd'},{'solver': 'adam'}]

#different learning parameters
learn_params=[{'learning_rate': 'constant'},
              {'learning_rate': 'invscaling'},
              {'learning_rate':'adaptive'}]


#different learning rates for sgd solver
learning_rate_params=[{'learning_rate_init': 0.001,'solver': 'sgd'},
                      {'learning_rate_init': 0.01,'solver': 'sgd'},
                      {'learning_rate_init': 0.04,'solver': 'sgd'},
                      {'learning_rate_init': 0.08,'solver': 'sgd'},
                      {'learning_rate_init': 0.12,'solver': 'sgd'},
                      {'learning_rate_init': 0.16,'solver': 'sgd'},
                      {'learning_rate_init': 0.20,'solver': 'sgd'}
        ]


#different maximum number of iterations
max_itr=[{'max_iter':200,'solver':'sgd'},
         {'max_iter':300,'solver':'sgd'},
         {'max_iter':500,'solver':'sgd'},
         {'max_iter':700,'solver':'sgd'}]


solver=[]
for param in solver_params:
    clf = MLPClassifier( alpha=1e-5, activation='logistic', 
                        hidden_layer_sizes=(20,20), random_state=1, max_iter=800, **param)
    
    clf.fit(train_input,train_output)                         
    
    predicted_output=clf.predict(test_input)
    
    print("Different solvers")
    solver.append(accuracy_score(test_output, predicted_output))
    print(accuracy_score(test_output, predicted_output))

lparam=[]
for param in learn_params:
    clf = MLPClassifier( alpha=1e-5, activation='logistic', 
                        hidden_layer_sizes=(20,20), random_state=1, max_iter=800, **param)
    
    clf.fit(train_input,train_output)                         
    
    predicted_output=clf.predict(test_input)
    
    print("Different learn_param")
    lparam.append(accuracy_score(test_output, predicted_output))
    print(accuracy_score(test_output, predicted_output))
    
lrate=[]
for param in learning_rate_params:
    clf = MLPClassifier( alpha=1e-5, activation='logistic', 
                        hidden_layer_sizes=(20,20), random_state=1, max_iter=800, **param)
    
    clf.fit(train_input,train_output)                         
    
    predicted_output=clf.predict(test_input)
    
    
    print("Different learning rate")
    lrate.append(accuracy_score(test_output, predicted_output))
    print(accuracy_score(test_output, predicted_output))

max_iterations=[]
for param in max_itr:
    clf = MLPClassifier( alpha=1e-5, activation='logistic', 
                        hidden_layer_sizes=(20,20), random_state=1, **param)
    
    clf.fit(train_input,train_output)                         
    
    predicted_output=clf.predict(test_input)
    
    print("Different maximum iterations")
    max_iterations.append(accuracy_score(test_output, predicted_output))
    print(accuracy_score(test_output, predicted_output))


#Plotting graphs
plt.figure(1)
plt.subplot(211)
plt.plot(solver, color='red')
plt.title("different solver")
  

plt.figure(2)
plt.subplot(212)
plt.plot(lparam, color='blue')
plt.title("different learning parameters")


plt.figure(3)
plt.subplot(221)
plt.plot(lrate, color='green')
plt.title("different learning rates")

plt.figure(4)
plt.subplot(222)
plt.plot(max_iterations, color='yellow')
plt.title("different maximum iterations")

plt.show()


