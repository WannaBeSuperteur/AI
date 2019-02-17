import math
import random

def sigmoid(value):
    return round(1 / (1 + math.exp(-value)), 6)

# print data
def printVector(vec, n):
    result = '['
    for i in range(len(vec)): result += ' ' + str(round(vec[i], n)) + ' '
    result += ']'
    return result

# make random value
def makeRandom(value):
    result = 0
    for i in range(value):
        if random.random() >= 0.5: result += 1/value
    result = round(result, 6)
    return result

# read data from file
def getData():
    # get data
    file = open('DBM.txt', 'r')
    read = file.readlines()
    file.close

    data = [] # training data
    cols = 0 # number of columns (attribute and answer)

    # read number of hidden units
    hiddens = read[0].split('\n')[0].split(' ')
    n = int(read[1]) # use hidden layer whose index value is n
    for i in range(len(hiddens)): hiddens[i] = int(hiddens[i])

    # read data
    for i in range(2, len(read)-1):
        read[i] = read[i].replace('\n', '')
        row = read[i].split(' ')
        for i in range(len(row)):
            row[i] = int(row[i])
        data.append(row)

    # read input data
    inputD = read[len(read)-1].split(' ')

    return (hiddens, data, inputD, n)

# main: DBM
def DBM(v, hiddens, lr, prt):
    HLs = len(hiddens) # number of hidden layers
    neurons = [len(v[0])] + hiddens # number of neurons (input ~ hidden)

    # initialize matrices
    W = [] # weight matrix
    h = [] # hidden layer matrix
    pba = [] # p(h[1]|v), p(h[2]|h[1]), ..., p(h[n]|h[n-1])
    pa1b = [] # p(v1|h[1]), p(h1[1]|h[2]), ..., p(h1[n-1]|h[n])
    v1 = [[0]*len(v[0]) for i in range(len(v))] # v1
    h1 = [] # h1[1], h1[2], ..., h1[n]
    aTb = [] # array for v^T*h[1], h[1]^T*h[2], ..., h[n-1]^T*h[n]
    a1Tb1 = [] # array for v1^T*h1[1], h1[1]^T*h1[2], ..., h1[n-1]^T*h1[n]
    dW = [] # array for dW
    
    for i in range(HLs):
        Woflayer = [[0]*neurons[i+1] for x in range(neurons[i])]

        # weight matrix
        for j in range(len(Woflayer)):
            for k in range(len(Woflayer[0])):
                Woflayer[j][k] = round((-1) + makeRandom(200)*2, 6)
        W.append(Woflayer)

        # hidden layer matrix
        if i == 0: hoflayer = [[0]*len(W[0][0]) for x in range(len(v))]
        else: hoflayer = [[0]*len(W[i][0]) for x in range(len(h[i-1]))]
        h.append(hoflayer)

        # h1[1], h1[2], ..., h1[n]
        if i == 0: h1oflayer = [[0]*len(W[0][0]) for x in range(len(v1))]
        else: h1oflayer = [[0]*len(W[i][0]) for x in range(len(h[i-1]))]
        h1.append(h1oflayer)

        # p(h[1]|v), p(h[2]|h[1]), ..., p(h[n]|h[n-1])
        if i == 0: pbaoflayer = [[0]*len(W[0][0]) for x in range(len(v))]
        else: pbaoflayer = [[0]*len(W[i][0]) for x in range(len(h[i-1]))]
        pba.append(pbaoflayer)

        # p(v1|h[1]), p(h1[1]|h[2]), ..., p(h1[n-1]|h[n])
        if i == 0: pa1boflayer = [[0]*len(v[0]) for x in range(len(v))]
        else: pa1boflayer = [[0]*len(h[i-1][0]) for x in range(len(h[i-1]))]
        pa1b.append(pa1boflayer)

        # array for v^T*h[1], h[1]^T*h[2], ..., h[n-1]^T*h[n]
        aTboflayer = [[0]*len(W[i][0]) for x in range(len(W[i]))]
        aTb.append(aTboflayer)

        # array for v1^T*h1[1], h1[1]^T*h1[2], ..., h1[n-1]^T*h1[n]
        a1Tb1oflayer = [[0]*len(W[i][0]) for x in range(len(W[i]))]
        a1Tb1.append(a1Tb1oflayer)

        # array for dW
        dWoflayer = [[0]*len(W[i][0]) for x in range(len(W[i]))]
        dW.append(dWoflayer)

    # print
    if prt != 0:
        print('< visible data >')
        for i in range(len(v)):
            print(v[i])
        print('')

        # print initialized weight of each layer
        for i in range(len(W)):
            print('< W of layer ' + str(i) + ' >')
            for j in range(len(W[i])): print(W[i][j])
            print('')

    # repeat until convergence
    count = 0
    while(1):
        count += 1
        print('ROUND ' + str(count))

        # 1. p(h[1]  |v     ) = sigm(v     *W[1]   + h[2]*W[2]^T)
        #    p(h[2]  |h[1]  ) = sigm(h[1]  *W[2]   + h[3]*W[3]^T)
        #    ...
        #    p(h[n-1]|h[n-2]) = sigm(h[n-2]*W[n-1] + h[n]*W[n]^T)
        #    p(h[n]  |h[n-1]) = sigm(h[n-1]*W[n]                )
        if prt != 0: print(' **** p(h[k+1]|h[k]) = sigm(h[k]*W[k+1] + h[k+2]*W[k+2]^T) ****')
        for layer in range(HLs):

            # h[k]*W[k+1]
            for i in range(len(v)):
                for j in range(len(W[layer][0])):
                   for k in range(len(W[layer])):
                       if layer == 0: pba[layer][i][j] += v[i][k] * W[layer][k][j]
                       else: pba[layer][i][j] += h[layer-1][i][k] * W[layer][k][j]
            # h[k+2]*W[k+2]^T
            if layer < HLs-1:
                for i in range(len(v)):
                    for j in range(len(W[layer+1])):
                       for k in range(len(W[layer+1][0])):
                           pba[layer][i][j] += h[layer+1][i][k] * W[layer+1][j][k]

            # sigmoid
            for i in range(len(pba[layer])):
                for j in range(len(pba[layer][i])):
                    pba[layer][i][j] = sigmoid(pba[layer][i][j])

            if prt != 0:
                if layer == 0: print(' ---- p(h[1]|v) ----')
                else: print(' ---- p(h[' + str(layer+1) + ']|h[' + str(layer) + ']) ----')
                    
                for i in range(len(pba[layer])): print(printVector(pba[layer][i], 6))
                print('')

        # 2. h[1]   = {p(h[1]  |v     ) >= random value}
        #    h[2]   = {p(h[2]  |h[1]  ) >= random value}
        #    ...
        #    h[n-1] = {p(h[n-1]|h[n-2]) >= random value}
        #    h[n]   = {p(h[n]  |h[n-1]) >= random value}
        if prt != 0: print(' **** h[k] = {p(h[k]|h[k-1]) >= randomval} ****')
        for layer in range(HLs):   
            for i in range(len(v)):
                for j in range(len(W[layer][0])):
                    if pba[layer][i][j] >= makeRandom(100): h[layer][i][j] = 1
                    else: h[layer][i][j] = 0

            if prt != 0:
                print(' ---- h[' + str(layer+1) + '] ----')
                for i in range(len(h[layer])):
                    print(printVector(h[layer][i], 6))
                print('')

        # 3. p(v1     |h[1]  ) = sigm(h[1]  *W[1]  ^T                )
        #    p(h1[1]  |h[2]  ) = sigm(h[2]  *W[2]  ^T + v     *W[1]  )
        #    ...
        #    p(h1[n-2]|h[n-1]) = sigm(h[n-1]*W[n-1]^T + h[n-3]*W[n-2])
        #    p(h1[n-1]|h[n]  ) = sigm(h[n]  *W[n]  ^T + h[n-2]*W[n-1])
        if prt != 0: print(' **** p(h1[k]|h[k+1]) = sigm(h[k+1]*W[k+1]^T + h[k-1]*W[k]) ****')
        for layer in range(HLs):
            # h1[k]*W[k]^T
            if layer < HLs:
                for i in range(len(h[layer])):
                    for j in range(len(W[layer])):
                        for k in range(len(W[layer][0])):
                            pa1b[layer][i][j] += h[layer][i][k] * W[layer][j][k]
            # h[k-2]*W[k-1]
            if layer > 0:
                if layer > 1: tempLen = len(h[layer-2])
                else: tempLen = len(v)
                
                for i in range(tempLen):
                    for j in range(len(W[layer-1][0])):
                        for k in range(len(W[layer-1])):
                            if layer > 1: pa1b[layer][i][j] += h[layer-2][i][k] * W[layer-1][k][j]
                            else: pa1b[layer][i][j] += v[i][k] * W[layer-1][k][j]

            # sigmoid
            for i in range(len(pa1b[layer])):
                for j in range(len(pa1b[layer][i])):
                    pa1b[layer][i][j] = sigmoid(pa1b[layer][i][j])

            if prt != 0:
                if layer == 0: print(' ---- p(v1|h[1]) ----')
                else: print(' ---- p(h1[' + str(layer) + ']|h[' + str(layer+1) + ']) ----')
                    
                for i in range(len(pa1b[layer])):
                    print(printVector(pa1b[layer][i], 6))
                print('')

        # 4. v1      = {sigm(                 h[1]*W[1]^T) >= random value}
        #    h1[1]   = {sigm(v1     *W[1]   + h[2]*W[2]^T) >= random value}
        #    h1[2]   = {sigm(h1[1]  *W[2]   + h[3]*W[3]^T) >= random value}
        #    ...
        #    h1[n-1] = {sigm(h1[n-2]*W[n-1] + h[n]*W[n]^T) >= random value}
        #    h1[n]   = {sigm(h1[n-1]*W[n]                ) >= random value}

        # for v1
        if prt != 0: print(' **** v1 = {sigm(h[1]*W[1]^T) >= randomval} ****')
        for i in range(len(v1)):
            for j in range(len(v1[0])):
                if pa1b[0][i][j] >= makeRandom(100): v1[i][j] = 1
                else: v1[i][j] = 0

        if prt != 0:
            for i in range(len(v1)): print(printVector(v1[i], 1))
            print('')

        # for h1[1], h1[2], ..., h1[n]
        if prt != 0: print(' **** h1[k] = {sigm(h1[k-1]*W[k] + h[k+1]*W[k+1]^T) >= randomval} ****')
        for layer in range(HLs):
            # for h1[k-1]*W[k]
            if layer == 0: tempLen = len(v1)
            else: tempLen = len(h1[layer-1])
            
            for i in range(tempLen):
                for j in range(len(W[layer][0])):
                   for k in range(len(W[layer])):
                       if layer == 0: h1[layer][i][j] += v1[i][k] * W[layer][k][j]
                       else: h1[layer][i][j] += h1[layer-1][i][k] * W[layer][k][j]

            # for h[k+1]*W[k+1]^T
            if layer < HLs-1:
                for i in range(len(v)):
                    for j in range(len(W[layer+1])):
                        for k in range(len(W[layer+1][0])):
                            h1[layer][i][j] += h[layer+1][i][k] * W[layer+1][j][k]

            # sigmoid
            for i in range(len(h1[layer])):
                for j in range(len(h1[layer][i])):
                    if sigmoid(h1[layer][i][j]) > makeRandom(100): h1[layer][i][j] = 1
                    else: h1[layer][i][j] = 0

            if prt != 0:
                print(' ---- h1[' + str(layer+1) + '] ----')
                for i in range(len(h1[layer])): print(printVector(h1[layer][i], 6))
                print('')

        # 5. dW[1]   = lr*(v     ^T*h[1]   - v1     ^T*h1[1]  )
        #    dW[2]   = lr*(h[1]  ^T*h[2]   - h1[1]  ^T*h1[2]  )
        #    ...
        #    dW[n-1] = lr*(h[n-2]^T*h[n-1] - h1[n-2]^T*h1[n-1])
        #    dW[n]   = lr*(h[n-1]^T*h[n]   - h1[n-1]^T*h1[n]  )
        if prt != 0: print(' **** dW[k] = lr*(h[k-1]^T*h[k] - h1[k-1]^T*h1[k]) ****')
        for layer in range(HLs):
            for i in range(len(W[layer])):
                for j in range(len(W[layer][0])):
                    aTb[layer][i][j] = 0
                    a1Tb1[layer][i][j] = 0

                    if layer == 0:
                        for k in range(len(v1)):
                            aTb[0][i][j] += v[k][i] * h[0][k][j]
                            a1Tb1[0][i][j] += v1[k][i] * h1[0][k][j]
                        dW[0][i][j] = lr*(aTb[0][i][j] - a1Tb1[0][i][j]) # calculate dW
                    else:
                        for k in range(len(v1)):
                            aTb[layer][i][j] += h[layer-1][k][i] * h[layer][k][j]
                            a1Tb1[layer][i][j] += h1[layer-1][k][i] * h1[layer][k][j]
                        dW[layer][i][j] = lr*(aTb[layer][i][j] - a1Tb1[layer][i][j]) # calculate dW

                    # truncate error
                    aTb[layer][i][j] = round(aTb[layer][i][j], 6)
                    a1Tb1[layer][i][j] = round(a1Tb1[layer][i][j], 6)
                    dW[layer][i][j] = round(dW[layer][i][j], 6)

            if prt != 0:
                print(' ---- dW[' + str(layer+1) + '] ----')
                for i in range(len(dW[layer])): print(printVector(dW[layer][i], 6))
                print('')

        # 7. W[k] += dW[k] for k=1,2,...,n
        updateSum = 0

        print(' **** updated W: W(t+1) = W(t)+dW ****')
        for layer in range(HLs):
            updateSumlayer = 0
            
            for i in range(len(W[layer])):
                for j in range(len(W[layer][0])):
                    W[layer][i][j] += dW[layer][i][j]
                    updateSumlayer += abs(dW[layer][i][j])
                    
                    W[layer][i][j] = round(W[layer][i][j], 6) # truncate error

            print(' ---- W[' + str(layer+1) + '] ----')
            for i in range(len(W[layer])): print(printVector(W[layer][i], 6))
            print('sum of update: ' + str(round(updateSumlayer, 6)))
            print('')

            updateSum += updateSumlayer
        print('TOTAL sum of update: ' + str(round(updateSum, 6)))

        # if converged, break
        if updateSum < 0.005:
            return W
            break

if __name__ == '__main__':
    (hiddens, data, inputD, n) = getData()

    print('< DATA >')
    for i in range(len(data)): print(data[i])
    print('')
    
    weight = DBM(data, hiddens, 0.01, 0)[n] # using 1st weight

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
