"""
Fundamental Structure of Neural Networks
Author - Ntc

Prototype for perceptron 
"""

import numpy as np
import matplotlib.pyplot as plt
from BasicFunctions.NeuralNetwork import *


class Perceptron:
    """
    Basic Structure 
    X = |---|    Y = |-|    W = |--|
        |---|n*m     |-|n*1     |--|(m+1)*o
        n - number of input set
        m - number of attributes
        o - number of outputs
    """
    def __init__(self, inputFile, outputFile="OUTPUT.txt"):

        self.inputFile = inputFile
        self.outputFile = outputFile
        X, Y = takeInput(self.inputFile)
        self.X, self.Y = X, Y
        self.W = setRandomWeights(len(X[0]), 1)
        self.learningRate = 0.3
        print(self.X.shape, self.Y.shape)

    def customData(self):
        with open(self.inputFile, 'r') as fp:
            fp.readline()
            temp = [[
                float(j[1:-1]) if j[0] == '"' else float(j)
                for j in i.split(',')[2:]
            ] + [1] for i in fp]
            temp = np.array(temp)
            temp = normalize(temp)
            X, Y = temp[:100, 1:], temp[:100, 0]
        return X, Y.reshape(len(X), 1)

    
    def perceptron(self, Xsmall, Ysmall):
        # print('INITIALIZED', Xsmall.shape, Ysmall.shape, self.W.shape)
        W = self.W
        a = net(Xsmall, W)
        # H = Linear(a)
        S = Sigmoid(a)
        n1 = self.learningRate
        numberOfTest = len(Xsmall)
        # print('Activation value - ', H)
        error = (S - Ysmall.reshape(len(Xsmall), 1))
        eprime = Sigmoid_prime(a)
        # eprime = 1
        # print(error, Xsmall, sum(error*Xsmall))
        # print('ERROR - ', error)
        # errorVal = np.sum(error) / numberOfTest
        # print('chg - ', n1 * sum(error * Xsmall * eprime) / numberOfTest)
        W1 = W - n1 * sum(error * Xsmall * eprime) / numberOfTest
        W = W1
        self.W = W
        # print('FINISHED')
    
    def TRAIN(self, batch=1):
        n = len(self.X)
        for i in range(0, n, batch):
            self.perceptron(self.X[i:i + batch], self.Y[i:i + batch])
        print(self.W)

    def run(self, epoch=1):
        showError = []
        # print("Initital weights", self.W)
        for _ in range(epoch):
            # print('EPOCH', _+1)
            self.TRAIN(5)
            error = errorNet(self.X, self.W, self.Y)
            print('ERROR ', error)
            showError.append(error)
        plt.plot(range(1, epoch + 1), showError)
        storeWeights(self.W, self.outputFile)
        plt.show()
        # print(self.X[0], self.Y[0])



a1 = Perceptron("INPUT.txt")
a1.run(10)


