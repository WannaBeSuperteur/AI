# FOR DEEP NEURAL NETWORK THAT RETURNS 1 OUTPUT VECTOR

import math
import random

# read data from file
def getData():
    # get data
    file = open('neuralNetwork.txt', 'r')
    read = file.readlines()
    file.close

    # make list of number of neurons
    elements = read[0].split('\n')[0].split(' ')
    eles = len(elements)
    neurons = []
    for i in range(eles): neurons.append(int(elements[i]))

    # make matrix
    matrix = [[0]*eles for i in range(eles)]

    # configurations
    config = read[1].split('\n')[0].split(' ')
    printDetail = int(config[0])
    prt = int(config[1])
    lr = float(config[2])
    maxError = float(config[3])

    # connect elements
    for i in range(2, len(read)):
        x = int(read[i].split('\n')[0].split(' ')[0])
        y = int(read[i].split('\n')[0].split(' ')[1])
        matrix[x][y] = 1

    return (neurons, matrix, printDetail, lr, prt, maxError)

# print data
def printVector(vec, n):
    result = '['
    for i in range(len(vec)): result += ' ' + str(round(vec[i], n)) + ' '
    result += ']'
    return result

# DFS and return chains
def DFSreturnChains(matrix, startNode, nextNode):
    node = nextNode
    stack = []
    path = [startNode, nextNode]
    chains = []
    eles = len(matrix)

    while 1:
        count = 0 # number of adjacent nodes
        # search for adjacent nodes
        for k in range(eles):
            if matrix[node][k] == 1:
                stack.append([k, node]) # path: node->k
                count += 1

        # if output node (out-degree is 0) then append to chain list
        if count == 0:
            # add to chain
            temp = []
            for i in range(len(path)): temp.append(path[i])
            chains.append(temp)
            
            if len(stack) == 0: break # finish DFS
            
            nextSearch = stack[len(stack)-1]
            backTo = nextSearch[1] # back to this node(layer)
            
            for k in range(len(path)-1, -1, -1):
                if path[k] == backTo: break
                path.pop()

        # pop and search adjacent nodes of new node
        node = stack.pop()[0]
        path.append(node)

    print('Neural Net Construction: ' + str(chains))
    return chains

# activation function
def sigmoid(value):
    return 1/(1+math.exp(-value))

# forward propagation
def forward(input_, iNn, oNn, oNt, oNw, prt, layerN):

    # input layer -> output layer
    # j INPUTs and i OUTPUTs
    outputInput = [] # output layer input
    for i in range(oNn):
        oX = -oNt[i]
        for j in range(iNn):
            oX += oNw[i][j]*float(input_[j])
        outputInput.append(oX)

    if prt != 0: print('Layer ' + str(layerN) + ' Input : ' + printVector(outputInput, 6))

    # find output layer output
    outputOutput = [] # output layer output
    for i in range(oNn): outputOutput.append(sigmoid(outputInput[i]))

    if prt != 0: print('Layer ' + str(layerN) + ' Output: ' + printVector(outputOutput, 6))

    return (outputInput, outputOutput)

# forward propagation for all layers
def forwardAll(matrix, wM, oM, tM, prt):
    eles = len(matrix)

    # initialize output data
    for i in range(1, len(oM)):
        for j in range(len(oM[0])):
            oM[i][j] = 0.0

    # sum of layer input
    oISum = []
    for i in range(eles):
        temp = []
        for j in range(len(oM[i])):
            temp.append(0)
        oISum.append(temp)

    # check layer A did forwarding from A to B
    forwarded = []
    for i in range(eles): forwarded.append(0)

    # forward propagation for all connected layers (i->j)
    for i in range(eles):
        for j in range(eles):
            if matrix[i][j] == 1:
                (oI, oO) = forward(oM[i], len(oM[i]), len(oM[j]), tM[j], wM[i][j], prt, j)
                forwarded[i] = 1

                # update input
                for x in range(len(oM[j])): oISum[j][x] += oI[x]
                # if no predecessor that didn't do forwarding, update output matrix of layer j using sum of input
                forwardCount = 0
                for k in range(eles):
                    if matrix[i][j] == 1 and forwarded[i] == 0: forwardCount += 1
                if forwardCount == 0:
                    for x in range(len(oM[j])):
                        oM[j][x] = sigmoid(oISum[j][x])

    if prt != 0: print('')

# define aNn, bNn and bNw
def aNnbNnbNw(chain, wM, oM, l):
    aOutput = oM[chain[l]] # output of each layer a
    aNn = len(aOutput) # number of neurons in layer a
    bOutput = oM[chain[l+1]] # output of each layer b
    bNn = len(bOutput) # number of neurons in layer b
    bNw = wM[chain[l]][chain[l+1]] # weight between layer a and b (a->b)

    return (aNn, bNn, bNw, aOutput, bOutput)

# backpropagation
def Back(matrix, wM, oM, lr, destOutput, chains):
    eles = len(matrix)

    # backpropagation for each chain, calculate dE/dW of i->j
    for k in range(len(chains)):
                    
        # to store S values
        S = [] # index 0(input) ~ index N-1(output)
        for l in range(len(chains[k])): S.append([])

        # for each output layer neuron
        outputOutput = oM[chains[k][len(S)-1]] # output of output(last) layer of the cain
        oNn = len(outputOutput) # number of output layer neurons (last layer of chain)
        for l in range(oNn):
            Oo = outputOutput[l]
            S[len(S)-1].append((destOutput[l]-Oo)*Oo*(1-Oo))

        # for each hidden layer of the chain (layer a->b)
        for l in range(len(S)-2, 0, -1):
            (aNn, bNn, bNw, aOutput, bOutput) = aNnbNnbNw(chains[k], wM, oM, l)

            # calculate value of S
            for m in range(aNn):
                Sum = 0
                for n in range(bNn):
                    Sum += S[l+1][n]*bNw[n][m]
                Oh = aOutput[m]
                S[l].append(Sum*Oh*(1-Oh))

        # update a-b weights: n 'a's and m 'b's
        for l in range(len(S)-2, -1, -1):
            (aNn, bNn, bNw, aOutput, bOutput) = aNnbNnbNw(chains[k], wM, oM, l)
                        
            for m in range(bNn):
                for n in range(aNn):
                    wM[chains[k][l]][chains[k][l+1]][m][n] += lr * S[l+1][m] * aOutput[n]

# make weight matrices and output matrices
def makeWeightAndOutputMatrices(input_, matrix, neurons):
    eles = len(neurons)

    # initialize weight, output and threshold matrix
    wM = [[0]*eles for x in range(eles)] # weight matrix
    oM = [] # output matrix
    tM = [] # threshold matrix
    for i in range(eles):
        oM.append(0)
        tM.append(0)
    
    # make weight matrix for each '1' element in 'matrix'
    for i in range(eles):
        for j in range(eles):
            if matrix[i][j] == 1:
                w = [[0]*neurons[i] for x in range(neurons[j])] # weight of layer a->b
                wM[i][j] = w

    # make output and threshold matrix
    for i in range(eles):
        o = [] # output of layer i
        t = [] # threshold of layer i
        for j in range(neurons[i]):
            o.append(0)
            t.append(0)
        oM[i] = o
        tM[i] = t

    # consider input as output of input layer
    for i in range(len(oM[0])):
        oM[0][i] = input_[i]
    
    return (wM, oM, tM)

# initialize weight and threshold (a->b)
def initWeightAndThreshold(aNn, bNn, bNt, bNw):
    # set weights and thresholds
    for i in range(bNn):
        weights = [] # array of weight of B layer neurons
        for j in range(aNn):
            bNw[i][j] = random.random() # aNs weights/neuron
        bNt[i] = random.random() # 1 threshold/neuron

# print neuron info
def printNeuronInfo(aNn, bNn, bNt, bNw, a, b):
    print('<NEURON LAYER ' + str(b) + '>')
    for i in range(bNn):
        print('Neuron ' + str(i) + ': thr = ' + str(round(bNt[i], 6)))
        for j in range(aNn):
            print('[ layer ' + str(a) + ' N ' + str(j) + ' ] weight = ' + str(round(bNw[i][j], 6)))
    print('')

# print change of weight (a->b)
def printWeightChange(name, bNn, aNn, bNw, bNw_):
    print(name + ' neurons:')
    for i in range(bNn):
        print(str(nId) + ' Neuron ' + str(i) + ':')
        for j in range(aNn):
            before = str(round(bNw_[i][j], 6))
            after = str(round(bNw[i][j], 6))
            print('[ inputN ' + str(j) + ' ] weight = ' + before + '->' + after)

# train Neural Network
def Backpropagation(input_, destOutput_, neurons, matrix, printDetail, lr, prt, maxError):

    # backpropagation for all connected layers (i->j)
    # make chains
    chains = [] # array of chains
    eles = len(matrix)
    for i in range(eles):
        # check predecessors of layer i
        iPre = 0 # number of predecessors of layer i
        for k in range(eles):
            if matrix[k][i] == 1: iPre += 1
                
        for j in range(eles):
            if matrix[i][j] == 1 and iPre == 0:
                dEdW = [] # array of dE/dWs
                        
                # 0. make all chains from each output layer to this weight(i->j)
                # (must be the form of a layer -> hidden -> hidden -> ... -> output layer)
                # using DFS
                chains += DFSreturnChains(matrix, i, j)
    print('chains                 : ' + str(chains))

    iNn = len(input_[0]) # number of input neurons
    oNn = len(destOutput_[0]) # number of output neurons

    # make weight matrices and output matrices
    (wM, oM, tM) = makeWeightAndOutputMatrices(input_[0], matrix, neurons)
    
    # initialize weight and threshold
    for i in range(eles):
        for j in range(eles):
            if matrix[i][j] == 1:
                initWeightAndThreshold(len(oM[i]), len(oM[j]), tM[j], wM[i][j])

    # repeat until convergence
    count = 0
    while 1:
        count += 1

        # print neuron info
        if printDetail >= -1 or count == 1 or count % 100 == 0:
            print('')
            print('ROUND ' + str(count))
        if printDetail >= 1 or count == 1:
            for i in range(eles):
                for j in range(eles):
                    if matrix[i][j] == 1:
                        printNeuronInfo(len(oM[i]), len(oM[j]), tM[j], wM[i][j], i, j)

        # learning
        # for each training data
        errorSum = 0.0
        for d in range(len(input_)):

            # forward propagation
            forwardAll(matrix, wM, oM, tM, prt)

            # backpropagation
            Back(matrix, wM, oM, lr, destOutput_[d], chains)

            if printDetail >= 0:
                print('input data    : ' + printVector(input_[d], 6))
                print('output data   : ' + printVector(oM[len(oM)-1], 6))
                print('dest output   : ' + printVector(destOutput_[d], 6))
            
            # calculate the sum of error
            # find all output layers
            outputL = [] # list of output layers
            for i in range(eles):
                count_ = 0
                for j in range(eles):
                    if matrix[i][j] == 1: count_+= 1
                if count_ >= 1: outputL.append(0)
                else: outputL.append(1)

            # calculate sum of error
            for i in range(eles):
                # do not count error for non-output layer
                if outputL[i] == 0: continue

                # number of output in each output layer
                outputs = len(oM[i])
                
                for j in range(outputs):
                    errorSum += abs(destOutput_[d][j] - oM[i][j])

        # check stop condition
        if errorSum / oNn < maxError or (count >= 20000 and printDetail >= -1) or count >= 1000000:
            print('[ FINAL : ' + str(count) + ' ] sum of error: ' + str(round(errorSum, 6)))
            break
        if printDetail >= -1 or count % 100 == 0:
            print('sum of error: ' + str(round(errorSum, 6)))

    # return layer info (number of neurons, thresholds and weights)
    return (neurons, matrix, wM, oM, tM)

if __name__ == '__main__':
    (neurons, matrix, printDetail, lr, prt, maxError) = getData()
    Backpropagation([[1, 2, 3, 4]], [[0.5, 0.7, 0.9, 0.95]], neurons, matrix, printDetail, lr, prt, maxError)
