# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:57:27 2017

@author: Akshayaa
"""

import numpy as np
import pandas as pd
import string

def sigmoid(x):
    ''' Sigmoid function using tanh '''
    return np.tanh(x)

def dsigmoid(x):
    ''' Derivative of the function above '''
    return 1.0 - np.tanh(x)**2

class MLP:
    ''' Multi-layer perceptron class. '''

    def __init__(self, *args):
        ''' Initialization of the perceptron with given sizes.  '''

        self.shape = args
        n = len(args)

        ''' Build layers '''
        self.layers = []
        ''' Input layer + 1 for bias '''
        self.layers.append(np.ones(self.shape[0]+1))
        ''' Hidden layers + output layer '''
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[i]))

        ''' Build weights matrix randomly between -0.25 and +0.25 '''
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                         self.layers[i+1].size)))

        ''' dw will hold last change in weights '''
        self.dw = [0,]*len(self.weights)

        ''' Reset weights to a random value'''
        self.reset()

    def reset(self):
        ''' This function resets the weights '''
        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size,self.layers[i+1].size))
            self.weights[i][...] = (2*Z-1)*0.25

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''

        '''Set input layer '''
        self.layers[0][0:-1] = data

        ''' Propagate from layer 0 to layer n-1 using tanh as activation function '''
        for i in range(1,len(self.shape)):
            ''' Propagate through layers '''
            self.layers[i][...] = sigmoid(np.dot(self.layers[i-1],self.weights[i-1]))

        ''' Return output '''
        return self.layers[-1]


    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        ''' Back propagate error related to target using lrate and momentum. '''
        '''' Momentum will increase the size of the steps taken towards the minimum '''

        deltas = []

        ''' Compute error on output layer '''
        error = target - self.layers[-1]
        delta = error*dsigmoid(self.layers[-1])
        deltas.append(delta)

        ''' Compute error on hidden layers '''
        for i in range(len(self.shape)-2,0,-1):
            delta = np.dot(deltas[0],self.weights[i].T)*dsigmoid(self.layers[i])
            deltas.insert(0,delta)
            
        '''Update weights '''
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T,delta)
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.dw[i] = dw
        '''Return error'''
        return (error**2).sum()


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ''' epochs is set to 2500 '''
    def learn(network,samples, epochs=2500, lrate=.05, momentum=0.1):
        ''' Train the dataset ''' 
        for i in range(epochs):
            n = np.random.randint(samples.size)
            network.propagate_forward( samples['input'][n] )
            errors=network.propagate_backward( samples['output'][n], lrate, momentum )

        ''' Test the dataset '''
        output=[]
        expected=[]
        for i in range(samples.size):
            o = network.propagate_forward( samples['input'][i] )
            #print (i, samples['input'][i], '%.2f' % o[0])
            output.append(o[0])
            #print ('(expected %.2f)' % samples['output'][i])
            
            expected.append(samples['output'][i])
        print(errors)
        print
    
    
    

    ''' ************************************************************ 
                Training XOR function 
        *********************************************************** '''
    network = MLP(2,2,1)
    samples = np.zeros(4, dtype=[('input',  float, 2), ('output', float, 1)])
    print("***********************************************")        
    print ("Learning the XOR logical function")
    network.reset() # The weights are reset
    error=[]
    samples[0] = (0,0), 0
    samples[1] = (1,0), 1
    samples[2] = (0,1), 1
    samples[3] = (1,1), 0
    learn(network, samples)
    
    
    ''' ************************************************************ 
                Training Sine function 
        *********************************************************** '''
    print("**************************************************")
    print ("Learning the sin function- sin of four input units")
    network.reset()
    network = MLP(4,10,1)
    samples = np.zeros(50, dtype=[('input',  float, 4), ('output', float, 1)])
    for n in range(0,50):
        samples['input'][n] = np.random.uniform(low=-1, high=1, size=(4,)) 
        samples['output'][n] = np.sin(np.sum(samples['input'][n]))
    training= samples[0:40]
    print("(Training dataset) Error:")
    learn(network, training)
    plt.figure(figsize=(10,5))
    plt.plot(training['output'],color='b',lw=1)
    x,y = training['input'],training['output']
    for i in range(training.shape[0]):
        y[i] = network.propagate_forward(x[i])
    plt.plot(y,color='r',lw=3)
    plt.title("Expected versus MLP output for learning the sin function") #Display the title
    plt.ylabel("Output")
    plt.xlabel("Samples")
    plt.show()

    
    ''' Test data Prediction '''
    plt.figure(figsize=(10,5))
    test= samples[40:50]    
    y= test['output']
    expected = y
    plt.plot(expected,color='b',lw=1)
    for i in range(test.shape[0]):
        y[i]=network.propagate_forward(test['input'][i])
    output= y
    plt.plot(output,color='r',lw=3)
    plt.title("Expected versus MLP Prediction on test data") #Display the title
    plt.ylabel("Output")
    plt.xlabel("Samples")
    plt.show()
    squares=[]    
    for j in range(len(output)):
        squares.append((output[j]-expected[j]) ** 2)
    print("(Test data) Error:")
    print(np.sum(squares))
    
    
  
   
    ''' ************************************************* 
                    Letters Recognition 
        ************************************************* '''
    print("**************************************************")
    print ("letter recognition")    
    letters = pd.read_csv('letter-recognition.csv')
    letters1=letters.ix[:,1:]
    ''' The columns are normalized '''
    letters_norm = (letters1 - letters1.mean()) / (letters1.max() - letters1.min())
    inp= letters_norm.values
    ''' Test data output list '''
    out= letters.ix[:,0].values 

             
    ''' Create binary notations for all the alphabets '''
    out1=[]
    a=np.zeros(26).astype(int).tolist()
    for i in range(0,26):
        a[i]=1
        out1.append(a)
        a=np.zeros(26).astype(int).tolist()

    ''' Create a list of alphabets ''' 
    alphabets=[]
    for i in string.ascii_uppercase[:]:
        alphabets.append(i)
        
    ''' Create a list of corresponding binary notations of given data output '''
    out2=[]
    for i in out:
        out2.append(out1[alphabets.index(i)])  

    ''' Create training(16000 data points) and test dataset (4000 datapoints) '''        
    training = inp[0: 16000]
    test = inp[16000:]
    test_out= out[16000:]
    ''' 16 input units, 10 hidden units and 26 output units '''
    network = MLP(16,10,26) 
    samples = np.zeros(16000, dtype=[('input',  float, 16), ('output', int, 26)])
    
    for i in range(0,16000):
        samples[i]= tuple(training[i]) , tuple(out2[i])
    print("(Training set) Error:")
    ''' Train the network with training dataset '''
    learn(network, samples,) 
    finy=[]
    for i in range(len(training)):
        ''' Get the output returned by the network for the training dataset '''
        y=network.propagate_forward(training[i])
        ''' Replace the maximum value in the output array of each training input with 1 '''
        y[np.where(y==np.max(y))] = 1
        ''' Append output of each input training data to a list '''
        finy.append([abs(round(e)) for e in y.tolist()])
    ''' Link the corresponding alphabet of each output list of every training input '''
    training_alpha=[]
    for x in finy:
        training_alpha.append(alphabets[out1.index(x)])

        
    ''' Training data misclassifications '''
    correct=0
    misclass=0
    
    for a in range(len(training_alpha)):
        if training_alpha[a] == out[a]:
            correct += 1
        else:
            misclass += 1
    print("(Training Data)- Total Correctly classified letters:")
    print(correct)
    print("(Training Data)- Total misclassified letters")
    print(misclass)
    print("(Training Data)- Accuracy:")
    print((correct/len(training)) * 100)    

    ''' Test data Prediction '''
    fin=[]
    for i in range(len(test)):
        ''' Get the output returned by the network for the test dataset '''
        y=network.propagate_forward(test[i])
        ''' Replace the maximum value in the output array of each test input with 1 '''
        y[np.where(y==np.max(y))] = 1
        ''' Append predicted output of each input test data to a list '''
        fin.append([abs(round(e)) for e in y.tolist()])
        
    ''' Link the corresponding alphabet of each output list of every test input '''
    alpha=[]
    for x in fin:
        alpha.append(alphabets[out1.index(x)])
        
    ''' Test data misclassifications '''
    correct=0
    misclass=0
    
    for a in range(len(alpha)):
        if alpha[a] == test_out[a]:
            correct += 1
        else:
            misclass += 1
    print("(Test Data)-Total Correctly classified letters:")
    print(correct)
    print("(Test Data)-Total misclassified letters")
    print(misclass)
    print("(Test Data)-Accuracy:")
    print((correct/len(test)) * 100)
    
 
    


    
    
    

    


    
    
    

        
    
    
    
    