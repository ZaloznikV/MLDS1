# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:09:08 2023

@author: Hmeljaro
"""

import pandas as pd
import numpy as np
import random

#data preparation
data = pd.read_csv("tki-resistance.csv", index_col=False)
Class = data.Class #original names of classes
y = pd.get_dummies(data.Class, drop_first=True) #binnary encoding

X = data.drop(columns="Class")
X_train = X[:130].to_numpy()
X_test = X[130:].to_numpy()
y_train = y[:130].to_numpy()
y_test = y[130:].to_numpy()



def all_columns(X, rand):
    return range(X.shape[1])

class Node():
    def __init__(self, feature=None, threshold=None, gini=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.gini = gini
        self.left = left
        self.right = right
        self.value = value #if leaf node
        
    def predict_helper(self, X):
        if self.value != None:
            return self.value()
        elif X[self.feature] < self.threshold:
            return self.left.predict_helper(X)
        else:
            return self.right.predict_helper(X)
        
    def predict(self, X):
        results = []
        for x in X: #for row in matrix
            results.append(self.predict_helper(x))
        return np.array(results)
            
            

class Tree():
    def __init__(self, rand=None, get_candidate_columns=all_columns, min_samples=2):
        self.rand = random.Random(rand)
        self.get_candidate_columns=get_candidate_columns
        self.min_samples = min_samples
        
        self.root = None
        
        def find_best_split(self, X,y):
            for feature in self.get_candidate_columns:
                for threshold in (X[:,feature]): #i for y, treshold for X
                    
            return -1
        
        def build(self, X, y):
            
            if np.size(y) < self.min_samples: #return Node with output - stopping criteria
                return Node(value=np.bincount(y).argmax()) #most occurance value
            
            elif len(set(y)) == 1: # all classes are equal
                return y[0]
            
            else: #perform split
                index_left, index_right, threshold, feature = self.find_best_split(X,y)
                
        
        
    
        

