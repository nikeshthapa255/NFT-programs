
"""
Bayesian classifier
P(Ci/x1, x2) = P(x1/Ci) * P(x2/Ci) * P(Ci)
"""

# general - use general_prob()
# m estimate - use  m_estimate()

import numpy as np
import matplotlib.pyplot as plt
from BasicFunctions.NeuralNetwork import *

class BayesianClassifier:

    def __init__(self, inputFile, outputFile = "OUTPUT_bc.txt"):
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.X, self.Y = self.getData()
        print('Parameters entered!')
        self.test_cases = len(self.X)
        self.sol_classes = self.get_sol_class(self.Y[:, 0])
        print('Solution classes identified')
        self.P = self.get_prior_experience()
        self.M = 2
        # self.trained = False
    def get_prior_experience(self):
        ans = []
        l = len(self.X[0])
        for i in range(l):
            ans.append(len(self.get_sol_class(self.X[:, i])))
        # print('P - ', ans)
        return [1/i for i in ans]
    def get_sol_class(self, X):
        sol = {}
        for i in X:
            if i in sol:
                sol[i] += 1
            else:
                sol[i] = 1
        # print('solution set - ', sol)
        return [[i, sol[i]] for i in sol]

    def getData(self):
        with open(self.inputFile, 'r') as fp:
            X = [line.split() for line in fp]
            X = np.array(X)
            X, Y = X[:, :-1], X[:, -1].reshape(len(X), 1)
            # print('DATA - ', X, Y)
        return X, Y

    def conditional_probablity(self, x, ci, a1):
        # print('CHECK for - ', x, ci, a1)
        a2 = [1 for i, j in enumerate(a1) if x==j and self.Y[i, 0]==ci]
        # print(a2)
        return sum(a2)
    
    def general_prob(self, x1, count):
        return x1/count
    
    def m_estimate(self, nc, n, ai):
        # (nc + m*p)/(n+m)
        return (nc + self.M*self.P[ai])/(n + self.M)
        


    def test(self, X):
        # print('TEST of - ', X)
        ans = []
        for ci, count in self.sol_classes:
            ch = 1
            ch1 = 0
            for ai, val in enumerate(X):
                try:
                    x1 = self.conditional_probablity(val, ci ,self.X[:, ai])
                    ch *= self.m_estimate(x1, count, ai)
                    ch1 = 1
                except:
                    print('ERROR at {} with class {}'.format(ai, ci))
            ch *= count/self.test_cases
            # print(ch, ch1, ci)
            ans.append([ch*ch1, ci])
        print("Probability of different classes - ", ans)
        if ans:
            mx = max(ans)
            return mx[1]
        return -1

classification_test(BayesianClassifier, 'INPUT.txt')

"""
Test input for INPUT.txt file 
0 0 1
"""
