"""
McCulloh-Pitts model
weights - +pv or -ve
theta = n*pw = nw
n - number of attributes
pw - number of +ve weights
nw - number of -ve weights
if h(x)>theta  1 else 0
if t!=a
change signs
"""

import numpy as np
import matplotlib.pyplot as plt
from BasicFunctions.NeuralNetwork import *

class McCulloh:
    def __init__(self, inputFile, theta, outputFile="output_MCP.txt"):
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.X, self.Y = takeInput(self.inputFile)
        print('Parameters entered!')
        self.test_cases = len(self.X)
        self.num_attributes = len(self.X[0])
        self.weigths =  setRandomWeights(*[self.num_attributes, 1])
        self.theta = theta

    def get_net(self, x1):
        x_mini = x1.reshape(1, self.num_attributes)
        net = np.dot(x_mini, self.weigths)
        return net
    # def train(self):

    def test(self, x1):
        x1 = np.array([1] + x1)
        print('TESt INPut - ', x1)
        net = np.sum((self.get_net(x1)>self.theta) + 0)
        return net
        



# TODO