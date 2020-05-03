"""
Winnow model 
wi+1 = wi*(alpha^(t-a))
"""

import numpy as np
import matplotlib.pyplot as plt
from BasicFunctions.NeuralNetwork import *

class Winnow:
    def __init__(self, inputFile, outputFile="output_MCP.txt"):
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.X, self.Y = takeInput(self.inputFile)
        print('Parameters entered!')
        self.test_cases = len(self.X)
        self.num_attributes = len(self.X[0])
        self.weigths =  setRandomWeights(*[self.num_attributes, 1])
        self.Alpha = 2
        self.theta = self.num_attributes - 0.1
        self.train()
    def get_net(self, x1):
        x_mini = self.X[i].reshape(1, self.num_attributes)
        net = np.dot(x_mini, self.weigths)
        return net

    def hx_willow(self, net):
        a =  np.sum((net>self.theta) + 0)
        return a
    
    def update_weights_winnow(self, t, a):
        print("change weights", t, a)
        w1 = self.weigths*(self.Alpha**(t-a + 0.0))
        self.weigths = w1
    
    def train(self, epoch = 10):
        print('Training Begin')
        for _ in range(epoch):
            ch = 0
            for i in range(self.test_cases):
                print('weights - ', self.weigths.T)
                t = self.Y[i, 0]
                net = self.get_net(self.X[i])
                print('x - {}, net = {}, threshold - {}'.format(x_mini, net, self.theta))
                # update weights
                a = self.hx_willow(net)
                if t!=a:
                    ch+=1
                    self.update_weights_winnow(t, a)
            if ch==0:
                break
        print('END of Training for {} epochs'.format(epoch))

    def test(self, x1):
        x1 = np.array(x1)
        net = self.get_net(x1)
        a = self.hx_willow(net)
        return a
        

inputFile = "INPUT.txt"

classification_test(Winnow, inputFile)

