# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:44:21 2023

@author: zaloz
"""
from scipy.optimize import fmin_l_bfgs_b
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import random
from scipy.stats import sem
from tqdm import tqdm


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
        diff = np.ones(self.n_classes-1) #vector of differences between thresholds
        bounds = len(W) * [(None, None)] + len(diff) * [(0, None)] # bounds for weights and differences between thresholds
        bounds[len(W)]  = (None, None) #first difference is actual first threshold value and can be negative since we dont have intercept
        
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
        


# test_X = np.array([[0, 0],
#                     [0, 1],
#                     [1, 0],
#                     [1, 1],
#                     [1, 1],
#                   [0, 0],
#                                       [0, 1],
#                                       [1, 0],
#                                       [1, 1],
#                                       [1, 1], [0, 1],
#                                       [1, 0],
#                                       [1, 1],
#                                       [1, 1]])

# test_y = np.array([0, 0, 1, 1, 2, 0, 0, 1, 1, 2, 0, 1, 1, 2])


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

# #testing
import time
     
# L = MultinomialLogReg()
# M = L.build(X, y_encoded)
# print(time.time() - a)
# #print(np.argmax(L.predict(test_X), axis=1))
# print("test set accuracy: ", np.sum(y_encoded  == np.argmax(M.predict(X), axis=1))/len(y_encoded))
        
# L = OrdinalLogReg()
# L.build(test_X, test_y)
# print(np.sum(np.argmax(L.predict(test_X), axis=1) == test_y)/len(test_y))




#evaluate models with cross validation

# L = MultinomialLogReg()
# M = L.build(X, y_encoded)
# #print(time.time() - a)
# #print(np.argmax(L.predict(test_X), axis=1))
# print("test set accuracy: ", np.sum(y_encoded  == np.argmax(M.predict(X), axis=1))/len(y_encoded))
        
data = pd.read_csv("dataset.csv", sep=";")
one_hot_encoded_data = pd.get_dummies(data, columns =['Competition', 'PlayerType', 'Movement'])
y = one_hot_encoded_data.ShotType
print("majority classifier: ", 3055/len(y))


        
data = pd.read_csv("dataset.csv", sep=";")
one_hot_encoded_data = pd.get_dummies(data, columns =['Competition', 'PlayerType', 'Movement'])
X = one_hot_encoded_data.drop(columns=["ShotType"]) #"Angle", "Distance"   

labels = {'dunk': 0, 'tip-in': 1, 'layup': 2, 'hook shot': 3, 'above head': 4, 'other': 5}



y_encoded = y.replace(labels)

# model = MultinomialLogReg()
# model.build(X, y_encoded)


idx = (random.choices( list(range(3) ), k=5) )  

X_joined = X.copy()
X_joined["y_encoded"] = y_encoded    


df = X_joined.sample(len(X_joined), replace=True)
df_y = df.y_encoded
df_x = df.drop(columns="y_encoded")
scaler = preprocessing.MinMaxScaler()
# df_x = scaler.fit_transform(df_x)

a = time.time()

# model = MultinomialLogReg()
# model.build(X.to_numpy()[:1000], y_encoded.to_numpy()[:1000])


def bootstrap_weights(n):
    
    weights_dunk = []
    weights_tipin = []
    weights_layup = []
    weights_hookshot = []
    weights_above_head = []

    for i in range(n):
        print(i)
        scaler = preprocessing.MinMaxScaler()
        data = pd.read_csv("dataset.csv", sep=";")
        one_hot_encoded_data = pd.get_dummies(data, columns =['Competition', 'PlayerType', 'Movement'])
        y = one_hot_encoded_data.ShotType
        X = one_hot_encoded_data.drop(columns=["ShotType"]) #"Angle", "Distance"   
        
        labels = {'dunk': 0, 'tip-in': 1, 'layup': 2, 'hook shot': 3, 'above head': 4, 'other': 5}
        
        
        
        y_encoded = y.replace(labels)
    
        X_np = X.to_numpy()
        y_encoded_np = y_encoded.to_numpy()
        
        idx = random.choices(list(range(X_np.shape[0])), k =X_np.shape[0])
        
        x_boot = X_np[idx]
        x_boot = scaler.fit_transform(x_boot) #only angle and distance
        y_boot = y_encoded_np[idx]
        
        model = MultinomialLogReg()
        model.build(x_boot, y_boot)
        
        weights_dunk.append(model.W[:,0])
        weights_tipin.append(model.W[:,1])
        weights_layup.append(model.W[:,2])
        weights_hookshot.append(model.W[:,3]) 
        weights_above_head.append(model.W[:,4]) 
    
    return [weights_dunk, weights_tipin, weights_layup, weights_hookshot, weights_above_head] #zadnji intercept, prvi transition


import pickle

def summary(calculate = False):
    if calculate:
        weights_list = bootstrap_weights(30)
        
        with open('weights_list.pickle', 'wb') as handle:
            pickle.dump(weights_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    with open('weights_list.pickle', 'rb') as handle:
        weights_list = pickle.load(handle)
    
    dunk = np.column_stack([row for row in weights_list[0]])
    tipin = np.column_stack([row for row in weights_list[1]])
    layup = np.column_stack([row for row in weights_list[2]])
    hookshot = np.column_stack([row for row in weights_list[3]])
    overhead = np.column_stack([row for row in weights_list[4]])
    
    print("dunk_mean: ", np.mean(dunk, axis=1), "dunk_SE: ", sem(dunk, axis=1))
    print("")
    print("tipin_mean: ", np.mean(tipin, axis=1), "tipin_SE: ", sem(tipin, axis=1))
    print("")
    print("layup_mean: ", np.mean(layup, axis=1), "layup_SE: ", sem(layup, axis=1))
    print("")
    
    print("hookshot_mean: ", np.mean(hookshot, axis=1), "hookshot_SE: ", sem(hookshot, axis=1))
    print("")
    
    print("overhead_mean: ", np.mean(overhead, axis=1), "overhead_SE: ", sem(overhead, axis=1))




def multinomial_bad_ordinal_good(size, rand):
    classes = np.random.choice([0, 1, 2], size=size, p=[0.05, 0.3, 0.65]) #not equally distributed between classes
    gender = np.random.choice([0, 1], size=size)
    income = np.random.normal(1, 1, size=size)
    age = np.random.normal(0, 1, size=size)
    data = pd.DataFrame({'Classes': classes, 'Age': age, 'Gender': gender, 'Income': income})
    y = data.Classes.to_numpy()
    
    data = data.drop(columns=["Classes"])
    X = data.to_numpy()
    
    return X,y

MBOG_TRAIN = 150

def log_loss(proba, true_y):
    s = 0
    for i in (range(len(true_y))):
        s -= np.log(proba[i, true_y[i]])
    
    return s


def log_loss_uncertaints(n):
    multi_loss = []
    ordinal_loss = []
    for i in tqdm(range(n)):
        X_train, y_train = multinomial_bad_ordinal_good(MBOG_TRAIN, random.Random(1))
        
        X_test, y_test = multinomial_bad_ordinal_good(1000, random.Random(1))
        
        M = MultinomialLogReg()
        O = OrdinalLogReg()
        
        M.build(X_train, y_train)
        O.build(X_train, y_train)
        
        multi_proba = M.predict(X_test)
        ordinal_proba = O.predict(X_test)
        
        # print(log_loss(multi_proba, y_test))
        # print(log_loss(ordinal_proba, y_test))
        
        multi_loss.append(log_loss(multi_proba, y_test))
        ordinal_loss.append(log_loss(ordinal_proba, y_test))
        
    print(np.mean(multi_loss), np.mean(ordinal_loss))    
    print(sem(multi_loss), sem(ordinal_loss))    

    return multi_loss, ordinal_loss

        
        
    
    
    
    
    

        
        
        
        
    
        
























    
    
    
    
    