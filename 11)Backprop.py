import numpy as np
import matplotlib.pyplot as plt
from BasicFunctions.NeuralNetwork import *


class ANN:
    def __init__(self, hiddenLayers, inputFile, outputFile="output_ann.txt"):
        # hiddenLayers  = [3, 5, output]
        # INPUT + means two hidden layers of three and 5 size + output
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.X, self.Y = self.customData()
        self.activationFunction = Linear
        self.primeFunction = lambda x: x
        self.weights = [] # n(i)*n(i+1)
        self.bias = [] # n(i)*1
        pv = len(self.X[0])
        self.layers = [pv] + hiddenLayers
        for i in hiddenLayers:
            self.bias.append(setRandomWeights(1, i))
            self.weights.append(setRandomWeights(pv, i))
            pv = i
        # self.bias = np.array(self.bias)
        # self.weights = np.array(self.weights)
        self.learningRate = 0.0003
        print('DISPLAY', self.layers, self.weights, self.bias, sep='\n')
    
    def customData(self):
        with open(self.inputFile, 'r') as fp:
            fp.readline()
            temp = [[
                float(j[1:-1]) if j[0] == '"' else float(j)
                for j in i.split(',')[2:]
            ] for i in fp]
            temp = np.array(temp)
            X, Y = temp[:100, 1:], temp[:100, 0]
        return normalize(X), normalize(Y.reshape(len(X), 1))
    
    def fun(self, X, W, B):
        return net(self.activationFunction(X), W) + B


    def feedFoward(self, X):
        X1 = [[X, 0.5]]

        for idx in range(len(self.weights)):
            w, b = self.weights[idx], self.bias[idx]
            X1.append([self.fun(X1[-1][0], w, b), idx])
        
        return X1
    
    def backPropagation(self, FF, Y):
        n1 = self.learningRate
        f1 = FF
        pv = 0
        chk = 0
        m = len(Y)
        for ii in range(len(f1)-1, 0, -1) :
            val, idx = f1[ii]
            w, b = self.weights[idx], self.bias[idx]
            # x = m * n(i-1)
            # w = n(i-1) * n(i)
            # b = n(i) * 1
            x = f1[ii-1][0]
            print( "X, W, B", x.shape, w.shape, b.shape)
            delta = 0
            if chk == 0:
                # ( m*o - m*o ) * ( m*o )
                C = (val - Y)
                # m*o
                delta = C * self.primeFunction(val)
                chk = 1
            else:
                # w1 = n(i) * n(i+1)
                # pv = m * n(i+1)
                w1  = self.weights[idx+1]
                delta =  np.dot( pv, w1.T) * self.primeFunction(val)
            # ni * 1
            one = np.ones([m,1])
            # print("delta - ", delta, self.bias[idx])
            self.bias[idx] = b - n1*(np.dot(one.T, delta) / m)
            print('bias', self.bias[idx], b, delta)
            self.weights[idx] = w - n1*( np.dot(x.T, delta) / m)
            pv = delta

        return np.sum(pv)/m

    def TRAIN(self, X, Y):
        # print('IN TRAIN,', *self.bias, sep='\n')
        # freed forward
        FF = self.feedFoward(X)
        # print('FEED- FORWARD', *FF, sep='\n')
        # check
        print(len(Y[0]), len(FF[-1][0][0]))
        if len(Y[0]) != len(FF[-1][0][0]):
            raise ValueError('INVALID OUTPUT LENGTH')

        # back-propagation
        error = self.backPropagation(FF, Y)
        return error

    def run(self, epoch=100, batch=1):
        X = self.X
        Y = self.Y
        erroCalc = []
        for _ in range(epoch):
            error = [
                self.TRAIN(X[i:i + batch], Y[i:i + batch])
                for i in range(0, len(X), batch)
            ]
            erroCalc.append(sum(error) / len(error))
            print('epoch{}'.format(_+1), self.bias, 'error- {}'.format(error))
        plt.plot(range(1, epoch + 1), erroCalc)
        plt.show()



a1 = ANN([5, 5, 5, 1], "TRAIN/kc_house_data.csv")
a1.run(5, 10)