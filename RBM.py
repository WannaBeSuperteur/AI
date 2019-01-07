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

    # read number of hidden units
    hiddens = int(read[0])

    # read data
    for i in range(1, len(read)-1):
        read[i] = read[i].replace('\n', '')
        row = read[i].split(' ')
        for i in range(len(row)):
            row[i] = int(row[i])
        data.append(row)

    # read input data
    inputD = read[len(read)-1].split(' ')

    return (hiddens, data, inputD)

def sigmoid(value):
    return round(1 / (1 + math.exp(-value)), 6)

# make random value
def makeRandom(value):
    result = 0
    for i in range(value):
        if random.random() >= 0.5: result += 1/value
    result = round(result, 6)
    return result

# average
def avg(array):
    if len(array) == 0: return 0
    return sum(array, 0.0) / len(array)

# standard deviation
def sd(array):
    if len(array) < 2: return 1

    average = avg(array)
    sum0 = 0 # sum of (xi-avg)^2
    for i in range(len(array)):
        sum0 += (array[i] - average)*(array[i] - average)
    return math.sqrt(sum0 / len(array))

# main: RBM
def RBM(v, colsOfW, lr):
    
    # initialize weight
    W = [[0]*colsOfW for i in range(len(v[0]))]
    for i in range(len(W)):
        for j in range(len(W[0])):
            W[i][j] = round((-1) + makeRandom(200)*2, 6)
    
    # print
    print('v:')
    for i in range(len(v)):
        print(v[i])
    print('')
    print('W:')
    for i in range(len(W)):
        print(W[i])
    print('')

    # repeat until convergence
    count = 0
    while(1):
        count += 1
        print('ROUND ' + str(count))
    
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
                if phv[i][j] >= makeRandom(100): h[i][j] = 1
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
                if pv1h[i][j] >= makeRandom(100): v1[i][j] = 1
                else: v1[i][j] = 0

        print('v1 = {p(v1|h) >= randomval}')
        for i in range(len(v1)):
            print(v1[i])
        print('')

        # 5. h(1) = {sigm(v1*W) >= randomval}
        h1 = [[0]*len(W[0]) for i in range(len(v1))]
        for i in range(len(v1)):
            for j in range(len(W[0])):
               for k in range(len(W)):
                   h1[i][j] += v1[i][k] * W[k][j]
               if sigmoid(h1[i][j]) >= makeRandom(100): h1[i][j] = 1
               else: h1[i][j] = 0

        print('h(1) = {sigm(v(1)*W) >= randomval}')
        for i in range(len(h1)):
            print(h1[i])
        print('')

        # 6. dW = lr*(v^T*h - v1^T*h1)
        vTh = [[0]*len(W[0]) for i in range(len(W))] # array for v^T*h
        v1Th1 = [[0]*len(W[0]) for i in range(len(W))] # array for v1^T*h1
        dW = [[0]*len(W[0]) for i in range(len(W))] # array for dW
        
        for i in range(len(W)):
            for j in range(len(W[0])):
                vTh[i][j] = 0
                v1Th1[i][j] = 0
                for k in range(len(v1)):
                    vTh[i][j] += v[k][i] * h[k][j]
                    v1Th1[i][j] += v1[k][i] * h1[k][j]
                    dW[i][j] = lr*(vTh[i][j] - v1Th1[i][j]) # calculate dW

                    # truncate error
                    vTh[i][j] = round(vTh[i][j], 6)
                    v1Th1[i][j] = round(v1Th1[i][j], 6)
                    dW[i][j] = round(dW[i][j], 6)

        print('dW = (learning rate)*(v^T*h - v(1)^T*h(1))')
        for i in range(len(dW)):
            print(dW[i])
        print('')
        
        # 7. W^(1) = W^(0)+dW
        updateSum = 0
        for i in range(len(W)):
            for j in range(len(W[0])):
                W[i][j] += dW[i][j]
                updateSum += abs(dW[i][j])
                
                W[i][j] = round(W[i][j], 6) # truncate error

        print('updated W: W(t+1) = W(t)+dW')
        for i in range(len(W)):
            print(W[i])
        print('sum of update: ' + str(updateSum))
        print('')

        # if converged, break
        if updateSum < len(W)*len(W[0])*0.001:
            return W
            break

(hiddens, data, inputD) = getData()
weight = RBM(data, hiddens, 0.01)

# modify weight data (average of abs = 1)
for i in range(len(weight[0])):
    
    sumAbs = 0 # sum of absolute values
    for j in range(len(weight)):
        sumAbs += abs(weight[j][i])
    avgAbs = sumAbs / len(weight) # average of absolute values

    for j in range(len(weight)):
        weight[j][i] = round(weight[j][i]/avgAbs, 6)

print('normalized weight:')
for i in range(len(weight)):
    print(weight[i])

# fill in data
# find the least-square error hidden node
lsh = -1 # index of least-square error hidden node
elsh = 10000 # error of lsh

for i in range(len(weight[0])):
    error = 0 # square of error (using least-square)
    for j in range(len(weight)):
        if inputD[j] != '-': # find error for indexes where inputD is a number
            er = int(inputD[j]) - weight[j][i]
            error += (er * er)

    # update lash
    if error < elsh:
        elsh = error
        lsh = i

print('')
print('least-square hidden node: ' + str(lsh) + ' (error: ' + str(round(elsh, 6)) + ')')

# fill in data
for i in range(len(inputD)):
    if inputD[i] == '-':
        if weight[i][lsh] >= 0: inputD[i] = 1
        else: inputD[i] = 0

print('')
print('final result: ')
print(inputD)
