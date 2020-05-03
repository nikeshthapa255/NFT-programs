"""
K-NN Classification
S1 - find distance of x, y w.r.t all point 
S2 - select k nearest neighbours
S3 - select one with highest probability
"""
import numpy as np
from BasicFunctions.NeuralNetwork import *


def euclidean_distance(a1, a2):
    return np.sqrt(sum([pow(a1[i]-a2[i], 2) for i in range(len(a1))]))

def manhattan_distance(a1, a2):
    # print(a1,  a2, sum([abs(a1[i]-a2[i]) for i in range(len(a1))]))
    return sum([abs(a1[i]-a2[i]) for i in range(len(a1))])


class KNN_classifier:
    def __init__(self, K, inputFile, outputFile = "OUTPUT_bc.txt"):
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.X, self.Y = self.getData()
        self.test_cases = len(self.X)
        self.get_distance = manhattan_distance
        self.K = K


    def getData(self):
        with open(self.inputFile, 'r') as fp:
            X = [list(map(int, line.split())) for line in fp]
            X = np.array(X)
            X, Y = X[:, :-1], X[:, -1].reshape(len(X), 1)
            print('DATA - ', X, Y)
        return X, Y
    
    def calc_distance(self, a1):
        ans = [[self.get_distance(a1, self.X[i]), self.Y[i, 0]] for i in range(self.test_cases)]
        ans.sort()
        return ans

    def get_max(self, count):
        ans = -1
        for i in count:
            if ans == -1:
                ans = i
            if count[i]>count[ans]:ans = i
        return ans

    def normal_knn(self, dist):
        count = {}
        d1 = dist[:self.K]
        # print('normal - ', d1)
        for i, j in d1:
            try:
                count[j]+= 1
            except:
                count[j] = 1
        
        return self.get_max(count)

    def weighted_knn(self, dist):
        # (dk-di)/(dk-d1)
        D = dist[:self.K]
        d1 = D[0][0]
        dk = D[-1][0]
        if dk==d1:
            print('NORMAL CAN only be done!')
            return self.normal_knn(dist)
        ans = {}
        for i, j in D:
            x1 = 1 if dk==i else (dk-i)/(dk-d1)
            try:
                ans[j] += x1
            except:
                ans[j] = x1
        return self.get_max(ans)

    def test(self, X):
        dist = self.calc_distance(X)
        print('distance calculated - ', dist)
        sol = self.weighted_knn(dist)
        return sol
        

classification_test(KNN_classifier, 3, 'INPUT.txt', type = int)

    
