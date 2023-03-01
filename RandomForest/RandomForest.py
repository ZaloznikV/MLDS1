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

class Node:
    def __init__(self, feature=None, threshold=None, gini=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.gini = gini
        self.left = left
        self.right = right
        self.value = value #if leaf node
        
    def predict_helper(self, X):
        if self.value != None:
            return self.value
        elif X[self.feature] < self.threshold:
            return self.left.predict_helper(X)
        else:
            return self.right.predict_helper(X)
        
    def predict(self, X):
        results = []
        for x in X: #for row in matrix
            results.append(self.predict_helper(x))
        return np.array(results)
            
            

class Tree:
    def __init__(self, rand=None, get_candidate_columns=all_columns, min_samples=2):
        self.rand = random.Random(rand) #mogoce treba popravit v argument konstruktorja random(stevilka)
        self.get_candidate_columns=get_candidate_columns
        self.min_samples = min_samples
        
        #self.root = None
        
    def find_best_split(self, X,y):
        gini_index = 2 #more than max possible, looking for the smallest one
        final_left = None
        final_right = None
        final_treshold = None
        final_feature = None
        # print(self.get_candidate_columns)
        # time.sleep(10)
        columns = self.get_candidate_columns(X, 1)
        #for feature in self.get_candidate_columns: #iterate over all features
            #print(feature)
        for feature in columns: #iterate over all features

            for threshold in (X[:,feature]): #iterate over all tresholds
                idx_left = []
                idx_right = []
                for i, value in enumerate(list(X[:,feature])):
                    #print(value)
                    try:
                        if value < threshold:
                            idx_left.append(i)
                        else:
                            idx_right.append(i)
                    except:
                        #print("v: ", value, "t: ", threshold, "f: ", feature, "i: ",i)
                        break
                #weights
                w_left = len(idx_left)/len(y)
                w_right = len(idx_right)/len(y)
                
                y_left = y[[idx_left]]
                y_right = y[[idx_right]]
                
                if (np.size(y_left) != 0) and (np.size(y_right) !=0): #valid split
                    
                    g_left = 1 - np.square(np.sum(y_left)/np.size(y_left)) - np.square((np.size(y_left) - np.sum(y_left))/np.size(y_left))
                    g_right = 1 - np.square(np.sum(y_right)/np.size(y_right)) - np.square((np.size(y_right) - np.sum(y_right))/np.size(y_right))
                    # print("g_left: ", g_left, y_left)
                    # print("g_right: ", g_right, y_right)
                    # print()
                    # print("###################")
                    gini_split = w_left * g_left + w_right * g_right
                    
                    if gini_split < gini_index: #update current best split
                        gini_index = gini_split
                        final_left = idx_left
                        final_right = idx_right
                        final_treshold = threshold
                        final_feature = feature

                        
                
        return final_left, final_right, gini_index, final_treshold, final_feature
        
    def build(self, X, y):
        #print("split")
        # print(self.get_candidate_columns)
        # print("###")
        #print(np.shape(y))
        if np.size(y) < self.min_samples: #return Node with output - stopping criteria
            return Node(value=np.bincount(y).argmax()) #most occurance value
        
        elif len(set(y.flatten())) == 1: # all classes are equal
            return Node(value=y.flatten()[0])
        
        else: #perform split

            index_left, index_right, gini_index, treshold, feature = self.find_best_split(X,y)
            X_left = X[index_left]
            y_left = y[index_left]
            X_right = X[index_right]
            y_right = y[index_right]
            
            left_child = self.build(X_left, y_left)
            right_child = self.build(X_right, y_right)
            return Node(feature, treshold, gini_index, left_child, right_child, None ) #popravi!!!
            



# class RandomForest:
#     def __init__(self, rand=None, n=100):
#         self.rand = random.Random(rand)
#         self.n=n
    
#     def build(self, X,y):
        














"""TEST"""

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


test_X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
test_y = np.array([0, 0, 1, 1])

# test_tree = Tree(1, all_columns(test_X, 1))
# model = test_tree.build(test_X, test_y)
# print(model.predict(test_X))


# bla = find_best(test_X, test_y)

# split = find_best(X_train, y_train)

# Lx = X_test[split[0]]
# Ly = y_test[split[0]]

# Rx = X_test[split[1]]
# Ry = y_test[split[1]]










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
    

    
    
    
    
    
    










        
            
        

                
                
        
        
    
        

