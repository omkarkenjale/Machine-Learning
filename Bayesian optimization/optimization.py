#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from bayes_opt import BayesianOptimization
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt

bounds = {
    'max_depth':(1,50),
    'n_estimators': (1,150)
}

def bagging(max_depth, 
                n_estimators):
    params = {
        'max_depth': int(max_depth),
        'n_estimators': int(n_estimators)
    }
    tree_model = DecisionTreeClassifier()
    model = BaggingClassifier(base_estimator = tree_model, n_estimators=int(n_estimators))
    model.fit(trainx,trainy)
    y_pred=model.predict(valx)
    acc =accuracy_score(y_pred,valy)*100
    return acc

def normalize(x, m, s): return (x-m)/s

def boosting(max_depth, 
                n_estimators):
    params = {
        'max_depth': int(max_depth),
        'n_estimators': int(n_estimators)
    }
    tree_model = DecisionTreeClassifier(max_depth=int(max_depth))
    model = AdaBoostClassifier(tree_model, n_estimators=int(n_estimators))
    model.fit(trainx,trainy)
    y_pred=model.predict(valx)
    acc =accuracy_score(y_pred,valy)*100
    return acc

def read():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    
    train_x = train.iloc[:, 1:785]
    train_y = train[train.keys()[0]]
    test_x = test.iloc[:, 1:785]
    test_y = test[test.keys()[0]]
    
    #splitting data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.1, random_state=1)
    x_train1, x_train2, y_train1, y_train2 = train_test_split(train_x, train_y, test_size=0.6, random_state=1)
    
    #Final data
    '''
    X_val, y_val => validation data
    x_train1, y_train1 => training data
    '''
   
    ### Normalization
    #x_train, x_test = train_x.float(), test_x.float()
    train_mean,train_std = x_train1.mean(),x_train1.std()
    train_mean,train_std
    
    x_train = normalize(x_train1, train_mean, train_std)
    x_val = normalize(X_val, train_mean, train_std)
    
    return x_train, y_train1, x_val, y_val

if __name__ == "__main__":
    #Loading data
    trainx,trainy,valx,valy = read()
    #Initiating optimization for bagging
    optimizer = BayesianOptimization(
    f=bagging,
    pbounds=bounds,
    random_state=1,
    )
    optimizer.maximize(n_iter=50)
    #Appending Accuracy
    bagging_accuracy = []
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
        bagging_accuracy.append(optimizer.res[i]['target'])
    #Plotting iteration Vs performance
    plt.plot(np.arange(0,55),bagging_accuracy, label = 'Bagging Accuracy')
    plt.xlabel("BO iterations")
    plt.ylabel("Performance")
    plt.title("Bayesian Optimization Performance over Bagging")
    plt.legend()
    plt.show()
    plt.savefig('plot2ia.png')
    #Initiating optimization for boosting
    optimizer1 = BayesianOptimization(
    f=boosting,
    pbounds=bounds,
    random_state=1,
    )
    optimizer1.maximize(n_iter=50)
    #Initiating optimization for boosting
    boosting_accuracy = []
    for i, res in enumerate(optimizer1.res):
        print("Iteration {}: \n\t{}".format(i, res))
        boosting_accuracy.append(optimizer1.res[i]['target'])
    #Plotting
    plt.plot(np.arange(0,55),boosting_accuracy, label = 'Boosting Accuracy')
    plt.xlabel("BO iterations")
    plt.ylabel("Performance")
    plt.title("Bayesian Optimization Performance over Boosting")
    plt.legend()
    plt.show()
    plt.savefig('plot2b.png')

