# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:44:21 2023

@author: zaloz
"""
from scipy.optimize import fmin_l_bfgs_b
import pandas as pd
import numpy as np
data = pd.read_csv("dataset.csv", sep=";")
from sklearn import preprocessing
# print(data.columns)
# print(data.dtypes)


#one-hot encoding categorical variables 
#we did not use closed form solution so there is no need to drop first column
#when performing one-hot encoding

def softmax(u):
    exp = np.exp(u)
    return exp/np.sum(exp, axis=1, keepdims=True)

def inverse_logit(u):
    return 1/(1+np.exp(-u))

one_hot_encoded_data = pd.get_dummies(data, columns =['Competition', 'PlayerType', 'Movement'])
print(one_hot_encoded_data)
X = one_hot_encoded_data.drop(columns=["ShotType","Angle", "Distance"])
#normaliziraj kot in razdaljo, najbolj redek shot daj na konec kot reference value
y = one_hot_encoded_data.ShotType
label_encoder = preprocessing.LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


class MultinomialLogReg():
    def __init__(self, W=None, intercept=True):
        self.W = W
        self.intercept = intercept
        

        
    def build(self, X, y):
        self.X = X
        self.y = y
        self.n_classes =len(np.unique(self.y)) #all possible classes
        if self.intercept: #using intercept
            self.X = np.concatenate((self.X, np.ones([self.X.shape[0], 1])), axis=1) #add intercept
 
            
        #weights
        W = np.zeros((self.X.shape[1], self.n_classes-1))
        arg_min, _, _ = fmin_l_bfgs_b(self.loss, W.flatten(), approx_grad=True) 
        W = arg_min.reshape(self.X.shape[1], self.n_classes-1) #back to original dimensions
        self.W = W
        return self #needed for tests to pass and that we dont need additional class for trained model.
        
        
        
        
    def loss(self, w):
        #print(w)
        w = w.reshape(self.X.shape[1], self.n_classes-1)
        Loss = np.dot(self.X, w)
        added_reference = np.concatenate((Loss, np.zeros((Loss.shape[0], 1))), axis=1) #added value for last class with value 0
        
        proba = softmax(added_reference) #calculate softmax value
        # print(proba)
        # print("###############################")
        # print(proba)
        log_likelihood = 0
        for i in range(len(self.y)): #looking for max, so min is - log_loss
            log_likelihood -= np.log(proba[i, self.y[i]])#log of probability that predicted value is ground truth value
        
        return log_likelihood
    
    
    def predict(self, X):
        if self.intercept:
            X = np.concatenate((X, np.ones([X.shape[0], 1])), axis=1)
        
        y_pred = np.dot(X, self.W)
        y_added_reference = np.concatenate((y_pred, np.zeros((y_pred.shape[0], 1))), axis=1)
        y_proba = softmax(y_added_reference)
        return y_proba
    
    
    
    
    
    
            
        
class OrdinalLogReg():
    def __init__(self, W=None) :
        #we will not use intercept in this case since it do not add any value to the model 
        #this also mean that first threshold is not set to 0
 
        self.W = W
        
    
    def build(self, X, y):
        self.X = X
        self.y = y
        self.n_classes = len(np.unique(self.y)) #all possible classes
        W = np.ones(self.X.shape[1]) #only vector of weights (not matrix)
        diff = np.ones(self.n_classes-1) #vector of thresholds
        bounds = len(W) * [(None, None)] + len(diff) * [(0, None)] # bounds for weights and differences between thresholds
        bounds[len(W)]  = (None, None) #first difference is actual first threshold value and can be negative
        
        matrix = np.concatenate((W,diff)) #first weights, then differences
        
        arg_min, _, _ = fmin_l_bfgs_b(self.loss, matrix.flatten(), approx_grad=True, bounds=bounds)
        
        self.W = arg_min[:len(W)]
        self.diff = arg_min[len(W):]
        
        T = []
        s = 0
        for e in self.diff:
            s = s + e
            T.append(s)
        
        self.T = T #thresholds    
        
        return self
        
        
        
    
    def loss(self, matrix):
        W = matrix[:self.X.shape[1]]
        #print(W.shape)
        
        diff = matrix[self.X.shape[1]:]
        #print(diff.shape)
        T = []
        s = 0
        for e in diff:
            s = s + e
            T.append(s) #thresholds
        
        U = np.dot(self.X, W)
        U = np.array([[x] for x in U])
        
        #print(U.shape)
        
        proba = inverse_logit(T - U)
        #print(proba.shape)

        proba_added_1 = np.concatenate((proba, np.ones([self.X.shape[0], 1])), axis=1) #add last value (prob  of last class for values up to inf)
        #print(proba_added_1.shape)
        C  = proba_added_1[:,1:] - proba
        #print(C.shape)
        proba_added_1[:,1:]  = proba_added_1[:,1:] - proba 
        proba = proba_added_1
        log_likelihood = 0
        for i in range(len(self.y)): #looking for max, so min is - log_loss
            log_likelihood -= np.log(proba[i, self.y[i]])#log of probability that predicted value is ground truth value
        
        return log_likelihood
        
        
    def predict(self, X):
        U = np.dot(X, self.W)
        U = np.array([[x] for x in U])
        proba = inverse_logit(self.T - U)
        proba_added_1 = np.concatenate((proba, np.ones([X.shape[0],1])), axis=1)
        proba_added_1[:,1:]  = proba_added_1[:,1:] - proba
        
        return proba_added_1
        


test_X = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                    [1, 1],
                  [0, 0],
                                      [0, 1],
                                      [1, 0],
                                      [1, 1],
                                      [1, 1], [0, 1],
                                      [1, 0],
                                      [1, 1],
                                      [1, 1]])

test_y = np.array([0, 0, 1, 1, 2, 0, 0, 1, 1, 2, 0, 1, 1, 2])


# train_x, train_y = test_X[::2], test_y[::2] 
# test_x, test_y = test_X[1::2], test_y[1::2]

# L = MultinomialLogReg()
# L.build(train_x, train_y)
# proba = L.predict(test_x)
# print(proba)
# print(proba.shape)
# print(np.sum(proba, axis=1))

# l = MultinomialLogReg()
# c = l.build(train_x, train_y)
# prob = l.predict(test_x)



# M = MultinomialLogReg()
# M.build(X[:10],y_encoded[:10]) 

#testing
import time
a = time.time()       
L = MultinomialLogReg()
M = L.build(X, y_encoded)
print(time.time() - a)
#print(np.argmax(L.predict(test_X), axis=1))
print("test set accuracy: ", np.sum(y_encoded  == np.argmax(M.predict(X), axis=1))/len(y_encoded))
        
L = OrdinalLogReg()
L.build(test_X, test_y)
print(np.sum(np.argmax(L.predict(test_X), axis=1) == test_y)/len(test_y))

 
 