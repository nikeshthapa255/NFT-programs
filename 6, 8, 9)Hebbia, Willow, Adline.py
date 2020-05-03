
import numpy as np
import matplotlib.pyplot as plt
from BasicFunctions.NeuralNetwork import *
# def update_adline
class BASE:
    """
    @param
    net : np.darray
    hx(net)

    @param
    t: int32
    a: int32
    x1: np.darray
    update_weights(t, a, x1)

    call self.train(epoch) for custom epochs

    """
    def __init__(self, inputFile, hx, update_weights,outputFile="output_MCP.txt", learning_rate = 2):
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.X, self.Y = takeInput(self.inputFile)
        print('Parameters entered!', self.Y)
        self.test_cases = len(self.X)
        self.num_attributes = len(self.X[0])
        self.theta = self.num_attributes - 0.1
        self.weights =  setRandomWeights(*[self.num_attributes, 1])
        self.hx = hx
        self.update_weights = update_weights
        self.learning_rate = learning_rate
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

"""
Winnow model 
wi+1 = wi*(alpha^(t-a))
"""
        
class Winnow(BASE):
    def __init__(self, inputFile, outputFile='output_MCP.txt', learning_rate=2):
        self.Alpha = 2
        super().__init__(inputFile, self.hx, self.update_weights, outputFile=outputFile, learning_rate=learning_rate)
        print(dir(Winnow))

    def hx(self, net):
        a =  np.sum((net>self.theta) + 0)
        return a
    def update_weights(self, t, a, x1):
        w1 = self.weights*(self.Alpha**(t-a + 0.0))
        return w1

"""
Hebbian 
wi+1 = wi + t*xi
"""

class Hebbian(BASE):
    """
    It supports bipolar outputs (ex - 1, -1)
    other will not train this model
    """
    def __init__(self, inputFile, outputFile='output_MCP.txt', learning_rate=2):
        super().__init__(inputFile, self.hx, self.update_weights, outputFile=outputFile, learning_rate=learning_rate)
    
    def hx(self, net):
        a =  sum([1 if i else -1 for i in (net>0).flatten()])
        return a
    
    def update_weights(self, t, a, x1):
        # print('HB - ', t, x1)
        w1 = self.weights + t*(x1.T)
        return w1

"""
ADLINE
wi+1 = wi + learning_rate*(t-net)*x
"""

class ADLINE(BASE):
    def __init__(self, inputFile, outputFile='output_MCP.txt', learning_rate=0.003):
        super().__init__(inputFile, self.hx, self.update_weights, outputFile=outputFile, learning_rate=learning_rate)
    
    def hx(self, net):
        return np.sum(net)
    
    def update_weights(self, t, a, x1):
        w1 = self.weights + self.learning_rate*(t-a)*(x1.T)
        return w1
    

inputFile = "INPUT.txt"
classification_test(ADLINE, inputFile, type=int)

