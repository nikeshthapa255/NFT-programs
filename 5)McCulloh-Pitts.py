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
    def __init__(self, inputFile, outputFile="output_MCP.txt"):
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.X, self.Y = takeInput(self.inputFile)
        print('Parameters entered!')
        self.test_cases = len(self.X)

    def train(self):

    def test(self):

# TODO