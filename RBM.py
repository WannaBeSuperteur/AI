import math
import random

# read data from file
def getData():
    # get data
    file = open('RBM.txt', 'r')
    read = file.readlines()
    file.close

    data = [] # training data
    cols = 0 # number of columns (attribute and answer)
    for i in range(len(read)):
        read[i] = read[i].replace('\n', '')
        row = read[i].split(' ')
        for i in range(len(row)):
            row[i] = int(row[i])
        data.append(row)

    return data

def sigmoid(value):
    return round(1 / (1 + math.exp(-value)), 2)

# main: RBM
def RBM(v, W):
    # print
    print('v:')
    for i in range(len(v)):
        print(v[i])
    print('')
    print('W:')
    for i in range(len(W)):
        print(W[i])
    print('')
    
    # 1. p(h|v) = sigm(vW)
    phv = [[0]*len(W[0]) for i in range(len(v))]
    for i in range(len(v)):
        for j in range(len(W[0])):
           for k in range(len(W)):
               phv[i][j] += v[i][k] * W[k][j]
           phv[i][j] = sigmoid(phv[i][j])

    print('p(h|v) = sigm(vW)')
    for i in range(len(phv)):
        print(phv[i])
    print('')

    # 2. h = {p(h|v) >= random value}
    h = [[0]*len(W[0]) for i in range(len(v))]
    for i in range(len(v)):
        for j in range(len(W[0])):
            if phv[i][j] >= 0.5: h[i][j] = 1
            else: h[i][j] = 0

    print('h = {p(h|v) >= randomval}')
    for i in range(len(h)):
        print(h[i])
    print('')

    # 3. p(v1|h) = sigm(h*W^T)
    pv1h = [[0]*len(v[0]) for i in range(len(v))]
    for i in range(len(v)):
        for j in range(len(W)):
            for k in range(len(W[0])):
                pv1h[i][j] += h[i][k] * W[j][k]
            pv1h[i][j] = sigmoid(pv1h[i][j])

    print('p(v1|h) = sigm(h*W^T)')
    for i in range(len(pv1h)):
        print(pv1h[i])
    print('')

    # 4. v(1) = {p(v1|h) >= random value}
    v1 = [[0]*len(v[0]) for i in range(len(v))]
    for i in range(len(v)):
        for j in range(len(W)):
            if pv1h[i][j] >= 0.5: v1[i][j] = 1
            else: v1[i][j] = 0

    print('v1 = {p(v1|h) >= randomval}')
    for i in range(len(v1)):
        print(v1[i])
    print('')

    # 5. h(1) = {sigm(v(1)*W) >= randomval}
    h1 = [[0]*len(W[0]) for i in range(len(v1))]
    for i in range(len(v1)):
        for j in range(len(W[0])):
           for k in range(len(W)):
               h1[i][j] += v1[i][k] * W[k][j]
           if sigmoid(h1[i][j]) >= 0.5: h1[i][j] = 1
           else: h1[i][j] = 0

    print('h(1) = {sigm(v(1)*W) >= randomval}')
    for i in range(len(h1)):
        print(h1[i])
    print('')

weight = [[0.2, -0.2], [-0.7, 0.1], [0.8, -0.5], [-0.4, -0.2], [0.2, 0.4], [-0.1, 0.1]]
data = getData()
RBM(data, weight)
