import numpy as np
"""
    Basic Structure 
    X = |---|    Y = |-|    W = |--|
        |---|n*m     |-|n*1     |--|(m+1)*o
        n - number of input set
        m - number of attributes
        o - number of outputs
"""

def normalize(a):
        for i in range(len(a[0])):
            a1 = a[:, i]
            mx, mn = max(a1), min(a1)
            if mx == mn:
                continue
            a[:, i] = (a[:, i] - mn) / (mx - mn)
        return a



def takeInput(input):
    # space seprated inputs in file
    with open(input, 'r') as fp:
        X = [ [1]+list(map(int, line.split())) for line in fp]
        X = np.array(X)
        X, Y = X[:, :-1], X[:, -1].reshape(len(X), 1)
    return X, Y


def setRandomWeights(n, m):
    W = np.random.uniform(0, 1, n * m).reshape(n, m)
    return W


def net(X, W):
    return np.dot(X, W)


def Sigmoid(A):
    return 1.0 / (1.0 + np.exp(-A))

def Sigmoid_prime(A):
    return Sigmoid(A)*(1-Sigmoid(A))


def Linear(A):
    return (A > 0) - 0


def storeWeights(W, output):
    with open(output, 'w') as fo:
        n = len(W)
        print(n, file=fo)
        for weights in W:
            print("ITEST", weights)
            print(*weights, file=fo)


def errorNet(X, W, Y):
    error = abs(Y - Linear(np.dot(X, W.T)))
    return np.sum(error) / len(error)

def classification_test(model, *args, type = str):
    print('Model initializing')
    t1 = model(*args)
    print('LETS Start')
    print('PRESS q to exit else enter attributes to get class ')
    while 1:
        x = input()
        if x=='Q' or x=='q':
            break
        x = list(map(type, x.split()))
        print("Your class of solution is - ", t1.test(x))