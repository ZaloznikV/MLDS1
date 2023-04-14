# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:09:08 2023

@author: Hmeljaro
"""

import pandas as pd
import numpy as np
import random
import time




def all_columns(X, rand):
    return list(range(X.shape[1]))

def random_sqrt_columns(X, rand):
    c = random.sample(list(range(X.shape[1])), round(np.sqrt(X.shape[1])) )
    return c


import pickle


class RandomForest:
    def __init__(self, rand=None, n=100):
        self.rand = random.Random(rand)
        self.n=n
    
    def build(self, X,y):
        trees = []
        n_rows = X.shape[0] #number of rows
        
        for i in range(self.n):
            #try:
                bootstrap_index = random.choices(list(range(n_rows)), k=n_rows)
                X_bootstrap = X[bootstrap_index]
                y_bootstrap = y[bootstrap_index]
                # print(X_bootstrap)
                # print(y_bootstrap)
                # print("#########################")
                # try:
                tree = Tree(self.rand, get_candidate_columns=random_sqrt_columns, min_samples=2)
                trees.append(tree.build(X_bootstrap, y_bootstrap))
                # except: 
                #     print("X: ", X_bootstrap, type(X_bootstrap), np.shape(X_bootstrap))
                #     print("y: ", y_bootstrap, type(y_bootstrap), np.shape(y_bootstrap))
                #     tree = Tree(self.rand, get_candidate_columns=random_sqrt_columns, min_samples=2)
                #     bla = tree.build(X_bootstrap, y_bootstrap)
                
            # except:
            #     print("x_bootstrap: ", X_bootstrap)
            #     print("y_bootstrap: ", y_bootstrap)
            #     print("index: ", bootstrap_index)
            #     print("###############################")
            #     #tree.build(X_bootstrap, y_bootstrap)
   
            
        return RFModel(trees)
        
        


class RFModel:

    def __init__(self, ls):
        self.ls = ls
        
    def predict(self, X):
        x_results = []

        for tree in self.ls:
            x_results.append(tree.predict(X))
        
            # results.append(max(set(x_results), key=x_results.count ))
        
        partial = np.transpose(np.array(x_results))
        results = [np.bincount(x).argmax() for x in partial]
        return np.array(results)




# [[0 0]
#  [1 1]
#  [0 1]
#  [1 0]]
# [0 1 0 1]



# x_bootstrap = np.array([[0, 1],
#  [0, 1],
#  [0, 1],
#  [1, 1]])
# y_bootstrap = np.array( [0, 0, 0, 1])





# """TEST"""

# def find_best(X_test, y_test):
#     print("split")
#     x_columns = all_columns(X_test,1)
#     gini_index = 2
#     for feature in x_columns:
#         for threshold in X_test[:,feature]:
#             idx_left = []
#             idx_right = []
#             for i, value in enumerate(X_test[:,feature]):
#                 #print(i, threshold,value)
#                 #time.sleep(5)
    
#                 if value < threshold:
#                     idx_left.append(i)
#                 else:
#                     idx_right.append(i)
#             w_left = len(idx_left)/len(y_test)
#             w_right = len(idx_right)/len(y_test)
#             # idx_left = np.array(idx_left)
#             # idx_right = np.array(idx_right)
#             y_left = y_test[[idx_left]]
#             y_right = y_test[[idx_right]]
            
#             if (np.size(y_left) != 0) and (np.size(y_right) !=0): #valid split
            
#                 g_left = 1 - np.square(np.sum(y_left)/np.size(y_left)) - np.square((np.size(y_left) - np.sum(y_left))/np.size(y_left))
#                 g_right = 1 - np.square(np.sum(y_right)/np.size(y_right)) - np.square((np.size(y_right) - np.sum(y_right))/np.size(y_right))
                
#                 gini_split = w_left * g_left + w_right * g_right
                
#                 if gini_split < gini_index: #update current best split
#                     gini_index = gini_split
#                     final_left = idx_left
#                     final_right = idx_right
#                     final_treshold = threshold
#                     final_feature = feature
#     return final_left, final_right, gini_index, final_treshold, final_feature



# blax = np.array([[1, 0],
#   [1, 0],
#   [0, 0],
#   [0, 0]])
# blay = np.array([1, 1, 0, 0])

test_X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
test_y = np.array([0, 0, 1, 1])

model = RandomForest(rand=random.Random(0), n=10)
t = model.build(test_X, test_y)
for i, tr in enumerate(t.ls):
    print(i)
    pr = tr.predict(test_X)
    print(pr)
print("___________")
print(t.predict(test_X))

# bla = t.predict(test_X)



# test_tree = Tree(1, all_columns(test_X, 1))
# model = test_tree.build(test_X, test_y)
# print(model.predict(test_X))


# bla = find_best(test_X, test_y)

# split = find_best(test_X, test_y)

# Lx = test_X[split[0]]
# Ly = test_y[split[0]]

# Rx = test_X[split[1]]
# Ry = test_y[split[1]]



# tree = Tree(2, all_columns)
# model = tree.build( test_X, test_y)







def hw_tree_full():
    #data preparation
    data = pd.read_csv("tki-resistance.csv", index_col=False)
    Class = data.Class #original names of classes
    y = pd.get_dummies(data.Class, drop_first=True) #binnary encoding

    X = data.drop(columns="Class")
    X_train = X[:130].to_numpy()
    X_test = X[130:].to_numpy()
    y_train = y[:130].to_numpy().flatten()
    y_test = y[130:].to_numpy().flatten()
    
    n=-1
    a = time.time()
    tree = Tree(2, all_columns)
    model = tree.build(X_train[:n], y_train[:n])
    print("time used: ", time.time() - a)
    acc_train = (y_train == model.predict(X_train)).sum() / np.size(y_train)
    acc_test = (y_test == model.predict(X_test)).sum() / np.size(y_test)
    
    #bernoulli distribution
    #SE = sqrt(p(1-p)/n)
    
    SE_train = np.sqrt(acc_train * (1- acc_train) / len(y_train))
    SE_test = np.sqrt(acc_test * (1- acc_test) / len(y_test))
    
    
    print("acc train: ", acc_train)
    print("acc test: ", acc_test)
    print("SE train: ", SE_train)
    print("SE test: ", SE_test)
    
    
    #return acc_train, acc_test, SE_train, SE_test
   
def hw_forest_full():    
    data = pd.read_csv("tki-resistance.csv", index_col=False)
    Class = data.Class #original names of classes
    y = pd.get_dummies(data.Class, drop_first=True) #binnary encoding
    
    X = data.drop(columns="Class")
    X_train = X[:130].to_numpy()
    X_test = X[130:].to_numpy()
    y_train = y[:130].to_numpy().flatten()
    y_test = y[130:].to_numpy().flatten()
    model = RandomForest(rand=random.Random(0), n=100)
    t = model.build(X_train, y_train)
    
    bla = t.predict(X_test)
    
    print(np.sum(np.array(bla) == y_test)/len(y_test))
        

    
    
from itertools import combinations_with_replacement

for comb in combinations_with_replacement(list(range(4)), 4):
    # print(comb)
    X = test_X[list(comb)]
    y = test_y[list(comb)]
    # print(X)
    # print(y)
    tree = Tree(all_columns)
    m = tree.build(X,y)
    print(m.predict(test_X))
    
   
# data = pd.read_csv("tki-resistance.csv", index_col=False)
# Class = data.Class #original names of classes
# y = pd.get_dummies(data.Class, drop_first=True) #binnary encoding

# X = data.drop(columns="Class")
# X_train = X[:130].to_numpy()
# X_test = X[130:].to_numpy()
# y_train = y[:130].to_numpy().flatten()
# y_test = y[130:].to_numpy().flatten()
# model = RandomForest(rand=random.Random(0), n=100)
# t = model.build(X_train, y_train)

# bla = t.predict(X_test)

# print(np.sum(np.array(bla) == y_test)/len(y_test))   










        
            
        

                
                
        
        
    
        

