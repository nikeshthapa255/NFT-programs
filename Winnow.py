"""
Winnow model 
wi+1 = wi*(alpha^(t-a))

Hebbian 
wi+1 = wi + t*xi
"""

import numpy as np
import matplotlib.pyplot as plt
from BasicFunctions.NeuralNetwork import *

def hx_hebbian(net):
    a =  sum([1 if i else -1 for i in (net>0).flatten()])
    return a

def hx_willow(net, theta):
    a =  np.sum((net>theta) + 0)
    return a

def update_weights_winnow( t, a, weights, Alpha):
    # print("change weights", t, a)
    w1 = weights*(Alpha**(t-a + 0.0))
    return w1

def update_weights_hebbian(t, a, weights, x1):
    print('HB - ', t, x1)
    w1 = weights + t*(x1.T)
    return w1
class Winnow:
    def __init__(self, inputFile, outputFile="output_MCP.txt"):
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.X, self.Y = takeInput(self.inputFile)
        print('Parameters entered!', self.Y)
        self.test_cases = len(self.X)
        self.num_attributes = len(self.X[0])
        self.weights =  setRandomWeights(*[self.num_attributes, 1])
        self.Alpha = 2
        self.theta = self.num_attributes - 0.1
        self.hx = lambda net: hx_hebbian(net)
        # self.hx = lambda net: hx_willow(net, self.theta)
        self.update_weights = lambda t, a, x1: update_weights_hebbian(t, a, self.weights, x1)
        # self.update_weights = lambda t, a, x1: update_weights_winnow(t, a, self.weights, self.Alpha)
        self.train()

    def get_net(self, x_mini):
        net = np.dot(x_mini, self.weights)
        return net
    
    def set_x(self, x1):
        x_mini = x1.reshape(1, self.num_attributes)
        return x_mini

    def train(self, epoch = 10):
        print('Training Begin')
        for _ in range(epoch):
            ch = 0
            for i in range(self.test_cases):
                print('weights - ', self.weights.T)
                t = self.Y[i, 0]
                x1 = self.set_x(self.X[i])
                net = self.get_net(x1)
                a = self.hx(net)
                print('x - {}, net = {}, threshold - {}'.format(x1, net, a))
                # update weights
                if t!=a:
                    ch+=1
                    self.weights = self.update_weights(t, a, x1)
            if ch==0:
                break
        print('END of Training for {} epochs'.format(epoch))

    def test(self, x1):
        x1 = np.array([1] + x1)
        print('TESt INPut - ', x1)
        net = self.get_net(x1)
        a = self.hx(net) # winnow

        return a
        

inputFile = "INPUT.txt"

classification_test(Winnow, inputFile, type=int)

