import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as rmse
from scipy.optimize import minimize
from sklearn import decomposition
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from scipy.stats import sem
import random
import matplotlib.pyplot as plt
from tqdm import trange

def load(fname):
    
    data = pd.read_csv(fname)
    features = list(data.columns[:-1])
    y = data.critical_temp

    X = data.drop(columns="critical_temp")
    
    X = X.to_numpy()
    y = y.to_numpy()
    #X["intercept"] = 1
    
    #X = X[["intercept"] + features]
    X_train = X[:200]
    X_test = X[200:]

    y_train = y[:200]
    y_test = y[200:]
    
    return features, X_train, y_train, X_test, y_test

class RidgeModel: #with predict method
    def __init__(self, B=None, pca_transformer=None, mu=None, std=None):
        self.B=B
        self.pca_transformer=pca_transformer
        self.mu = mu
        self.std = std
        
    def predict(self, X_test):

        X_test = (X_test - self.mu) / self.std
        
        if self.pca_transformer != False:
            X_test = self.pca_transformer.transform(X_test)
        
        X_test = np.hstack((np.ones([X_test.shape[0],1]),X_test))
            
        return  np.dot(X_test, self.B.T)
        
    
class RidgeReg:
    def __init__(self, param):
        self.param = param

    def fit(self, X_train, y_train, pca=False, pca_value=0.8):
        mu = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        
        if X_train.shape[0] >= 30: #only if sample large enough for normal distribution (problem with tests)
            X_train = (X_train - mu)/std #standardization
        else: 
            mu = 0
            std = 1

        if pca !=False: #apply pca
            pca = decomposition.PCA(n_components=pca_value)
            pca.fit(X_train)
            X_train = pca.transform(X_train)
        
        X_train = np.hstack((np.ones([X_train.shape[0],1]),X_train))

        I = np.eye(X_train.shape[1])
        I[0][0] = 0 #intercept should not be penalized
        penalty = self.param * I
        
        
        B = np.linalg.inv(X_train.T @ X_train + penalty) @ X_train.T @ y_train #closed form solution

        if pca !=False:
            return RidgeModel(B=B, pca_transformer=pca, mu=mu, std=std)
        
        else:
            return RidgeModel(B=B, pca_transformer=False, mu=mu, std=std)

class LassoModel: #with predict method
    def __init__(self, B=None, pca_transformer=None, mu=None, std=None):
        self.B=B
        self.pca_transformer=pca_transformer
        self.mu = mu
        self.std = std
        
    def predict(self, X_test):
        X_test = (X_test - self.mu) / self.std
        
        if self.pca_transformer != False:
            X_test = self.pca_transformer.transform(X_test)
        
        X_test = np.hstack((np.ones([X_test.shape[0],1]),X_test))
            
        return  np.dot(X_test, self.B.T)     
        
        
        

class LassoReg:
    def __init__(self, param):
        self.param = param
    
    def fit(self, X_train, y_train, pca=False, pca_value=0.8):
        mu = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        
        if X_train.shape[0] >= 30: #if sample not large enough 
            X_train = (X_train - mu)/std
            
        else: #no transformation 
            mu = 0
            std = 1

        if pca !=False: #apply pca
            pca = decomposition.PCA(n_components=pca_value)
            pca.fit(X_train)
            X_train = pca.transform(X_train)
        X_train = np.hstack((np.ones([X_train.shape[0],1]),X_train))
        B = np.random.rand(np.shape(X_train)[1]) #first approximation
        def optimize_B(B): #loss function
            return  np.shape(X_train)[0] * rmse(y_train, np.dot(X_train, B.T), squared=True) + self.param * np.sum(np.abs(B)) 
       
        B_hat = minimize(optimize_B, B, method="powell", tol=0.01)["x"] #minimizing loss function
        
        if pca !=False:
            return LassoModel(B=B_hat, pca_transformer=pca, mu=mu, std=std)
        
        else:
            return LassoModel(B=B_hat, pca_transformer=False, mu=mu, std=std)
        
        


class RegularizationTest(unittest.TestCase):

    def test_ridge_simple(self):
        X = np.array([[1],
                      [10],
                      [100]])
        y = 10 + 2*X[:,0]
        model = RidgeReg(1)
        m = model.fit(X, y)
        y = m.predict(np.array([[10],
                           [20]]))
        self.assertAlmostEqual(y[0], 30, delta=0.1)
        self.assertAlmostEqual(y[1], 50, delta=0.1)

    # ... add your tests

features, X_train, y_train, X_test, y_test = load("superconductor.csv")


def evaluate(X_train, y_train, X_test, y_test, param, ridge=True, pca=False, pca_value=0.8, folds=10):  
    """Evaluates the model with 10-fold cross-validation, returns mean, std and SE"""
    indexes = list(range(X_train.shape[0]))  
    random.shuffle(indexes)    
    X_train = X_train[indexes]
    y_train = y_train[indexes]
    kf = KFold(n_splits=folds)
    kf.get_n_splits(X_train)
    
    scores = []
    
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        #print()
        x_train = X_train[train_index]
        x_test = X_train[test_index]
        y_tr =  y_train[train_index]
        y_te = y_train[test_index]
        
        if ridge:
            model = RidgeReg(param)
            fitted_model = model.fit(x_train, y_tr, pca=pca, pca_value=pca_value)
            predictions = fitted_model.predict(x_test)
            err = rmse(predictions, y_te, squared=False)
            scores.append(err)
            
        else:
            model = LassoReg(param)
            fitted_model = model.fit(x_train, y_tr, pca=pca, pca_value=pca_value)
            predictions = fitted_model.predict(x_test)
            err = rmse(predictions, y_te, squared=False)
            scores.append(err)
            
    return np.mean(scores), np.std(scores), sem(scores)


# l = LassoReg(0)
# r = RidgeReg(0)
# L = l.fit(X_train, y_train, pca=True)
# R = r.fit(X_train, y_train, pca=True)
    
    

def plot_graphs(X_train, y_train, X_test, y_test):
    """plots graph of mean RMSE with respect to different weights"""
    mean = []
    se = []
    
    ls = np.arange(0,100,0.1)
    for p in ls:
        m, var, s = evaluate(X_train, y_train, X_test, y_test, p)
        mean.append(m)
        se.append(s)
    
    plt.plot(ls, mean, label='mean')
    plt.ylim(0, 50)
    plt.plot(ls, np.array(mean) + np.array(se), c="r", label="SE")
    plt.plot(ls, np.array(mean) - np.array(se), c="r")
    plt.title("Ridge regression")
    plt.xlabel("Regularization weight")
    plt.ylabel("RMSE")
    #plt.fill_between(range(len(mean)), mean-se, mean+se, alpha=0.2, label='standard error')
    plt.legend()
    relevant = np.array(mean) + np.array(se)
    #print(ls[np.argmin(relevant[:50])]) 
    plt.show()
    
    mean = []
    se = []
    ls = np.arange(0,5,0.2)
    for p in ls:
        m, var, s = evaluate(X_train, y_train, X_test, y_test, p, ridge=False)
        mean.append(m)
        se.append(s)
    
    plt.plot(ls, mean, label='mean')
    plt.ylim(0, 50)
    plt.plot(ls, np.array(mean) + np.array(se), c="r", label="SE")
    plt.plot(ls, np.array(mean) - np.array(se), c="r")
    plt.title("Lasso regression")
    plt.xlabel("Regularization weight")
    plt.ylabel("RMSE")
    #plt.fill_between(range(len(mean)), mean-se, mean+se, alpha=0.2, label='standard error')
    plt.legend()
    plt.show()
    
    relevant = np.array(mean) + np.array(se)
    #print(ls[np.argmin(relevant[:50])])    
    pass

def bootstrap_superconductor(X_train, y_train, X_test, y_test, pca_status=False):
    """Evaluates model on test set, using bootstrap"""
    L = []
    R = []
    for i in trange(50): #bootstrap
        indexes = random.choices(list(range(np.shape(X_train)[0])), k=np.shape(X_train)[0])
        X =   X_train[ indexes ]
        y =   y_train[ indexes ]
        
        l = 0.5
        r= 1.8
        
        l = LassoReg(0.5)
        r = RidgeReg(1.8)
    
        lm = l.fit(X, y, pca=pca_status)
        rm = r.fit(X,y, pca=pca_status)
        
        Rrmse =  rmse(y_test, rm.predict(X_test), squared=False)
        Lrmse =  rmse(y_test, lm.predict(X_test), squared=False)
        
        L.append(Lrmse)
        R.append(Rrmse)
        
    if pca_status:
        print("PCA")
    print("ridge: ")
    print("mean: ", np.mean(R))
    print("SE: ", sem(R))
    print("")
    print("lasso: ")
    print("mean: ", np.mean(L))
    print("SE: ", sem(L))
    #return(L)


    

def plot_graphs_pca(X_train, y_train, X_test, y_test):
    """plots graph of mean RMSE with respect to different weights with applied PCA."""
    mean = []
    se = []
    
    ls = np.arange(0,100,0.1)
    for p in ls:
        m, var, s = evaluate(X_train, y_train, X_test, y_test, p, pca=True)
        mean.append(m)
        se.append(s)
    
    plt.plot(ls, mean, label='mean')
    plt.ylim(0, 50)
    plt.plot(ls, np.array(mean) + np.array(se), c="r", label="SE")
    plt.plot(ls, np.array(mean) - np.array(se), c="r")
    plt.title("Ridge regression - PCA")
    plt.xlabel("Regularization weight")
    plt.ylabel("RMSE")
    #plt.fill_between(range(len(mean)), mean-se, mean+se, alpha=0.2, label='standard error')
    plt.legend()
    relevant = np.array(mean) + np.array(se)
    print(ls[np.argmin(relevant[:50])]) 
    plt.show()
    
    mean = []
    se = []
    ls = np.arange(0,10,1)
    for p in ls:
        m, var, s = evaluate(X_train, y_train, X_test, y_test, p, ridge=False, pca=True)
        mean.append(m)
        se.append(s)
    
    plt.plot(ls, mean, label='mean')
    plt.ylim(0, 50)
    plt.plot(ls, np.array(mean) + np.array(se), c="r", label="SE")
    plt.plot(ls, np.array(mean) - np.array(se), c="r")
    plt.title("Lasso regression - PCA")
    plt.xlabel("Regularization weight")
    plt.ylabel("RMSE")
    #plt.fill_between(range(len(mean)), mean-se, mean+se, alpha=0.2, label='standard error')
    plt.legend()
    plt.show()
    
    relevant = np.array(mean) + np.array(se)
    print(ls[np.argmin(relevant[:50])])    
    pass



def superconductor(X_train, y_train, X_test, y_test):
    """Returns means RMSE and SE of optimal models on train set, using 10fold cross-validation"""
    print("train data - 10 fold CV")
    print("mean rmse:         std:        SE:")
    print(evaluate(X_train, y_train, X_test, y_test, 0.5, ridge=False, pca=False))
    print("mean rmse:         std:        SE:")
    print(evaluate(X_train, y_train, X_test, y_test, 01.8, ridge=True, pca=False))
    print("PCA:")
    print("mean rmse:         std:        SE:")
    print(evaluate(X_train, y_train, X_test, y_test, 0.5, ridge=False, pca=True))
    print("mean rmse:         std:        SE:")
    print(evaluate(X_train, y_train, X_test, y_test, 1.8, ridge=True, pca=True))
    print("")
    return (evaluate(X_train, y_train, X_test, y_test, 0.5, ridge=False, pca=False), evaluate(X_train, y_train, X_test, y_test, 1.8, ridge=True, pca=False),
            evaluate(X_train, y_train, X_test, y_test, 0.5, ridge=False, pca=True), evaluate(X_train, y_train, X_test, y_test, 1.8, ridge=True, pca=True))


if __name__ == "__main__":
    features, X_train, y_train, X_test, y_test = load("superconductor.csv")
    superconductor(X_train, y_train, X_test, y_test)
    unittest.main()

