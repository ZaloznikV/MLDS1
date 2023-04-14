import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

class Network(object):
    def __init__(self, sizes, optimizer="sgd", decay_rate=0):
        # weights connect two layers, each neuron in layer L is connected to every neuron in layer L+1,
        # the weights for that layer of dimensions size(L+1) X size(L)
        # the bias in each layer L is connected to each neuron in L+1, the number of weights necessary for the bias
        # in layer L is therefore size(L+1).
        # The weights are initialized with a He initializer: https://arxiv.org/pdf/1502.01852v1.pdf
        self.weights = [((2/sizes[i-1])**0.5)*np.random.randn(sizes[i], sizes[i-1]) for i in range(1, len(sizes))]
        self.biases = [np.zeros((x, 1)) for x in sizes[1:]]
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        if self.optimizer == "adam":
            
            self.beta1 = 0.9 #starting parameters
            self.beta2 = 0.999
            m = []
            v = []
            m_b = []
            v_b = []
            for i in range(1,len(sizes)):
                _1 = np.zeros((sizes[i],sizes[i - 1]))
                _2 = np.zeros((sizes[i], 1))
                m.append(_1)
                v.append(_1)
                m_b.append(_2)
                v_b.append(_2)
            
            self.m = m
            self.v = v
            self.m_b = m_b
            self.v_b = v_b


    def train(self, training_data,training_class, val_data, val_class, epochs, mini_batch_size, eta):
        # training data - numpy array of dimensions [n0 x m], where m is the number of examples in the data and
        # n0 is the number of input attributes
        # training_class - numpy array of dimensions [c x m], where c is the number of classes
        # epochs - number of passes over the dataset
        # mini_batch_size - number of examples the network uses to compute the gradient estimation

        iteration_index = 0
        eta_current = eta

        n = training_data.shape[1]
        for j in range(epochs):
            print("Epoch"+str(j))
            loss_avg = 0.0
            mini_batches = [
                (training_data[:,k:k + mini_batch_size], training_class[:,k:k+mini_batch_size])
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                output, Zs, As = self.forward_pass(mini_batch[0])
                gw, gb = net.backward_pass(output, mini_batch[1], Zs, As)

                self.update_network(gw, gb, eta_current, j+1)

                # Implement the learning rate schedule for Task 5
                
                eta_current *= np.exp(-self.decay_rate*j)
                iteration_index += 1

                loss = cross_entropy(mini_batch[1], output)
                loss_avg += loss

            print("Epoch {} complete".format(j))
            print("Loss:" + str(loss_avg / len(mini_batches)))
            if j % 10 == 0:
                self.eval_network(val_data, val_class)



    def eval_network(self, validation_data,validation_class):
        # validation data - numpy array of dimensions [n0 x m], where m is the number of examples in the data and
        # n0 is the number of input attributes
        # validation_class - numpy array of dimensions [c x m], where c is the number of classes
        n = validation_data.shape[1]
        loss_avg = 0.0
        tp = 0.0
        for i in range(validation_data.shape[1]):
            example = np.expand_dims(validation_data[:,i],-1)
            example_class = np.expand_dims(validation_class[:,i],-1)
            example_class_num = np.argmax(validation_class[:,i], axis=0)
            output, Zs, activations = self.forward_pass(example)
            output_num = np.argmax(output, axis=0)[0]
            tp += int(example_class_num == output_num)

            loss = cross_entropy(example_class, output)
            loss_avg += loss
        print("Validation Loss:" + str(loss_avg / n))
        print("Classification accuracy: "+ str(tp/n))

    def update_network(self, gw, gb, eta, t):
        # gw - weight gradients - list with elements of the same shape as elements in self.weights
        # gb - bias gradients - list with elements of the same shape as elements in self.biases
        # eta - learning rate
        # SGD
        if self.optimizer == "sgd":
            for i in range(len(self.weights)):
                self.weights[i] -= eta * gw[i]
                self.biases[i] -= eta * gb[i]
        elif self.optimizer == "adam":
            for i in range(len(self.weights)):
                self.m[i] = self.beta1 * self.m[i] + (1-self.beta1) * gw[i]
                self.v[i] = self.beta2 * self.v[i] + (1-self.beta2) * (gw[i]**2)
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gb[i]
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (gb[i]**2)
                
                step = eta * np.sqrt(1-self.beta2 ** t) / (1-self.beta1 ** t)
                
                self.weights[i] -= step * (self.m[i] / (np.sqrt(self.v[i]) + 1e-7))
                self.biases[i] -= step * (self.m_b[i] / (np.sqrt(self.v_b[i]) + 1e-7))

        else:
            raise ValueError('Unknown optimizer:'+self.optimizer)



    def forward_pass(self, input):
        # input - numpy array of dimensions [n0 x m], where m is the number of examples in the mini batch and
        # n0 is the number of input attributes
        ########## Implement the forward pass
        Z = [] #z values
        A = [input] #activation values
        X = input #X values
        n = len(self.weights)
        for i in range(n):
            W = self.weights[i]
            b = self.biases[i]
            z = np.matmul(W,  X) + b
            
            Z.append(z)
            
            if i != n-1: #for all layers except last
                X = sigmoid(z) #update x values
                A.append(X)
        
        out = softmax(Z[-1])
        A.append(out)
        return out, Z, A

    def backward_pass(self, output, target, Zs, activations):
        ########## Implement the backward pass
        dw = list([None for w in self.weights]) #gradients of weights
        db = dw.copy() #gredients of biases

        
        prev_derivate = softmax_dLdZ(output, target) #first gradient of cost function
        
        for i in reversed(range(len(self.weights))): #update weights
            a = activations[i]

            dw[i] = np.matmul(prev_derivate, a.T) / a.shape[1]
            db[i] = prev_derivate.mean(axis=1, keepdims=True)
            if i != 0:
                prev_derivate = (np.matmul(self.weights[i].T , prev_derivate)) * sigmoid_prime(Zs[i-1])

        return dw, db
    


def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)

def softmax_dLdZ(output, target):
    # partial derivative of the cross entropy loss w.r.t Z at the last layer
    return output - target

def cross_entropy(y_true, y_pred, epsilon=1e-12):
    targets = y_true.transpose()
    predictions = y_pred.transpose()
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def load_data_cifar(train_file, test_file):
    train_dict = unpickle(train_file)
    test_dict = unpickle(test_file)
    train_data = np.array(train_dict['data']) / 255.0
    train_class = np.array(train_dict['labels'])
    train_class_one_hot = np.zeros((train_data.shape[0], 10))
    train_class_one_hot[np.arange(train_class.shape[0]), train_class] = 1.0
    test_data = np.array(test_dict['data']) / 255.0
    test_class = np.array(test_dict['labels'])
    test_class_one_hot = np.zeros((test_class.shape[0], 10))
    test_class_one_hot[np.arange(test_class.shape[0]), test_class] = 1.0
    return train_data.transpose(), train_class_one_hot.transpose(), test_data.transpose(), test_class_one_hot.transpose()

if __name__ == "__main__":
    train_file = "./data/train_data.pckl"
    test_file = "./data/test_data.pckl"
    train_data, train_class, test_data, test_class = load_data_cifar(train_file, test_file)
    val_pct = 0.1
    val_size = int(len(train_data) * val_pct)
    val_data = train_data[..., :val_size]
    val_class = train_class[..., :val_size]
    train_data = train_data[..., val_size:]
    train_class = train_class[..., val_size:]
    # The Network takes as input a list of the numbers of neurons at each layer. The first layer has to match the
    # number of input attributes from the data, and the last layer has to match the number of output classes
    # The initial settings are not even close to the optimal network architecture, try increasing the number of layers
    # and neurons and see what happens.
    net = Network([train_data.shape[0],100, 100, 10], optimizer="sgd") #100 x 100
    print("SGD: ")
    net.train(train_data,train_class, val_data, val_class, 40, 20, 0.01)  #epochs, mini_batch_size, eta 49% acc
    net.eval_network(test_data, test_class)
    print("")
    
    net = Network([train_data.shape[0],100, 100, 10], optimizer="adam") #100 x 100
    print("Adam: ")
    net.train(train_data,train_class, val_data, val_class, 40, 20, 0.01)  #epochs, mini_batch_size, eta 49% acc
    net.eval_network(test_data, test_class)
    print("")
    
    for i in [0.005, 0.01, 0.02]:
        print("learning rate: ", i)
        print("")
        net = Network([train_data.shape[0],100, 100, 10], optimizer="sgd") #100 x 100
        print("SGD: ")
        net.train(train_data,train_class, val_data, val_class, 40, 20, i)  #epochs, mini_batch_size, eta 49% acc
        net.eval_network(test_data, test_class)
        print("")
        
        net = Network([train_data.shape[0],100, 100, 10], optimizer="adam") #100 x 100
        print("Adam: ")
        net.train(train_data,train_class, val_data, val_class, 40, 20, i)  #epochs, mini_batch_size, eta 49% acc
        net.eval_network(test_data, test_class)
        print("")
        
    for decay in [ 0.001, 0.0001]:
        #print("learning rate: ", i)
        print("decay rate: ", decay)
        net = Network([train_data.shape[0],100, 100, 10], optimizer="sgd", decay_rate=decay) #100 x 100
        print("SGD: ")
        net.train(train_data,train_class, val_data, val_class, 40, 20, 0.02)  #epochs, mini_batch_size, eta 49% acc
        net.eval_network(test_data, test_class)
        print("")
        
        net = Network([train_data.shape[0],100, 100, 10], optimizer="adam", decay_rate=decay) #100 x 100
        print("Adam: ")
        net.train(train_data,train_class, val_data, val_class, 40, 20, 0.02)  #epochs, mini_batch_size, eta 49% acc
        net.eval_network(test_data, test_class)
        print("")
            
            
            
            
    
    

        
    
    

#v trainu eta, v konstruktorju mreže decay rate


