# FOR DEEP NEURAL NETWORK (GENERALIZED)

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
def DFSreturnChains(matrix, startNode, nextNode, outputList):
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

    for i in range(len(chains)): outputList.append(chains[i][len(chains[i])-1])
    
    return chains

# activation function
def sigmoid(value):
    return 1/(1+math.exp(-value))

# forward propagation
def forward(input_, iNn, oNn, oNt, oNw, fw, prt, layerN):

    # input layer -> output layer
    # j INPUTs and i OUTPUTs
    outputInput = [] # output layer input
    for i in range(oNn):
        oX = -oNt[i]
        for j in range(iNn):
            oX += fw * oNw[i][j] * float(input_[j])
        outputInput.append(oX)

    if prt != 0: print('Layer ' + str(layerN) + ' Input : ' + printVector(outputInput, 6))

    # find output layer output
    outputOutput = [] # output layer output
    for i in range(oNn): outputOutput.append(sigmoid(outputInput[i]))

    if prt != 0: print('Layer ' + str(layerN) + ' Output: ' + printVector(outputOutput, 6))

    return (outputInput, outputOutput)

# forward propagation for all layers
def forwardAll(matrix, wM, oM, tM, fwM, prt):
    eles = len(matrix)

    # initialize output data
    for i in range(eles):
        pred = 0 # number of predecessors of this layer
        for j in range(eles): pred += matrix[j][i]
        # there are 1 or more predecessors -> not an input layer -> initialize
        if pred > 0:
            for j in range(len(oM[i])):
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
                (oI, oO) = forward(oM[i], len(oM[i]), len(oM[j]), tM[j], wM[i][j], fwM[i][j], prt, j)
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
def Back(matrix, wM, oM, bwM, lr, destOutput, chains):
    eles = len(matrix)

    # backpropagation for each chain, calculate dE/dW of i->j
    for k in range(len(chains)):
                    
        # to store S values
        S = [] # index 0(input) ~ index N-1(output)
        for l in range(len(chains[k])): S.append([])

        # for each output layer neuron
        lastLayer = chains[k][len(S)-1] # ID of output(last) layer of the chain
        outputOutput = oM[lastLayer] # output of output(last) layer of the cain
        oNn = len(outputOutput) # number of output layer neurons (last layer of chain)
        for l in range(oNn):
            Oo = outputOutput[l]
            S[len(S)-1].append((destOutput[lastLayer][l]-Oo)*Oo*(1-Oo))

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

            a = chains[k][l]
            b = chains[k][l+1]
                        
            for m in range(bNn):
                for n in range(aNn):
                    wM[a][b][m][n] += lr * bwM[a][b] * S[l+1][m] * aOutput[n]

# make weight matrices and output matrices
def makeWeightAndOutputMatrices(input_, matrix, neurons):
    eles = len(neurons)

    # initialize weight, output and threshold matrix
    wM = [[0]*eles for x in range(eles)] # weight matrix
    oM = [] # output matrix
    tM = [] # threshold matrix
    fwM = [[0]*eles for x in range(eles)] # forwarding weight matrix
    bwM = [[0]*eles for x in range(eles)] # backpropagation weight matrix
    
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

    # make forwarding weight matrix (weight between layer a and b)
    for i in range(eles):
        for j in range(eles):
            fwM[i][j] = 1.0

    # make backpropagation weight matrix (weight between layer a and b)
    for i in range(eles):
        for j in range(eles):
            bwM[i][j] = 1.0

    # consider input as output of input layer
    initInput(oM, input_)
            
    return (wM, oM, tM, fwM, bwM)

# modify forwarding and backpropagation weights
def modifyWM(fwM, bwM, input_, d, count):
    eles = len(fwM)
    for i in range(eles):
        for j in range(eles):
            fwM[i][j] = 1.0
            bwM[i][j] = 1.0

# initialize input
def initInput(oM, input_): 
    # consider input as output of input layer
    for i in range(len(oM)):
        if len(input_[i]) > 0: # for each input layer
            for j in range(len(oM[i])): oM[i][j] = input_[i][j]
        else:
            for j in range(len(oM[i])): oM[i][j] = 0

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

# train Neural Network
def Backpropagation(input_, destOutput_, neurons, matrix, printDetail, lr, prt, maxError, modifyWMFunc):
    outputList = [] # ID of output layers

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
                chains += DFSreturnChains(matrix, i, j, outputList)
    if prt == 1: print('chains                 : ' + str(chains))

    # make weight matrices and output matrices
    (wM, oM, tM, fwM, bwM) = makeWeightAndOutputMatrices(input_[0], matrix, neurons)
    
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

            # initialize output matrix
            initInput(oM, input_[d])

            # modify weight (fwM and bwM)
            modifyWMFunc(fwM, bwM, input_, d, count)

            # forward propagation
            forwardAll(matrix, wM, oM, tM, fwM, prt)

            # backpropagation
            Back(matrix, wM, oM, bwM, lr, destOutput_[d], chains)

            if printDetail >= 0:
                for i in range(len(input_[d])):
                    if len(input_[d][i]) > 0:
                        print('input data (' + str(i) + '): ' + printVector(input_[d][i], 6))
                print('')
                for i in range(len(destOutput_[d])):
                    if len(destOutput_[d][i]) > 0:
                        print('dest output (' + str(i) + '): ' + printVector(destOutput_[d][i], 6))
                print('')
                for i in range(len(outputList)):
                    layerID = outputList[i]
                    print('output data (' + str(layerID) + '): ' + printVector(oM[layerID], 6))
            
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
            count_ = -1
            for i in range(eles):
                # do not count error for non-output layer
                if outputL[i] == 0: continue

                # number of output in each output layer
                outputs = len(oM[i])
                count_ += 1
                
                for j in range(outputs):
                    errorSum += abs(destOutput_[d][i][j] - oM[i][j])

        # check stop condition
        if errorSum < maxError or (count >= 20000 and printDetail >= -1) or count >= 1000000:
            print('')
            print('[ FINAL : ' + str(count) + ' ] sum of error: ' + str(round(errorSum, 6)))

            # print weights
            for i in range(eles):
                for j in range(eles):
                    if matrix[i][j] == 1:
                        printNeuronInfo(len(oM[i]), len(oM[j]), tM[j], wM[i][j], i, j)
                        
            break
        if printDetail >= -1 or count % 100 == 0:
            print('sum of error: ' + str(round(errorSum, 6)))

    # return layer info (number of neurons, thresholds and weights)
    return (neurons, matrix, wM, oM, tM, fwM)

# test neural network (return output)
def test(testInput, matrix, wM, oM, tM, fwM):
    print(' ******** TEST ********')
    print('')

    for i in range(len(testInput)):
        if len(testInput[i]) > 0:
            print('test input  (' + str(i) + '): ' + printVector(testInput[i], 6))
    print('')

    layers = len(matrix) # number of layers in this neural network
    matrix_ = [[0]*layers for i in range(layers)]
    for i in range(layers):
        for j in range(layers):
            matrix_[i][j] = matrix[i][j]

    # consider input as output of input layer
    initInput(oM, testInput)

    # do forwarding
    forwardAll(matrix_, wM, oM, tM, fwM, 1)

    # print oM
    for i in range(layers):
        succ = 0 # number of successors of this layer
        for j in range(layers): succ += matrix[i][j]

        # no successor -> output layer
        if succ == 0: print('output data (' + str(i) + '): ' + printVector(oM[i], 6))

    return oM

if __name__ == '__main__':
    (neurons, matrix, printDetail, lr, prt, maxError) = getData()

    # make input data
    input_ = [[[], [], [], [], [], [], [], [], [], []]] # input data
    input_[0][0] = [1, 2, 2] # input of layer 0 is [1, 2, 2]
    input_[0][3] = [1, 3, 4] # input of layer 3 is [1, 3, 4]
    input_[0][6] = [4, 1, 0] # input of layer 6 is [4, 1, 0]

    # make destination output data
    destO = [[[], [], [], [], [], [], [], [], [], []]] # dest output
    destO[0][2] = [0.5, 0.7, 0.9, 0.95] # destination output of layer 2 is [0.5, 0.7, 0.9, 0.95]
    destO[0][5] = [0.5, 0.6, 0.7, 0.8] # destination output of layer 5 is [0.5, 0.6, 0.7, 0.8]
    destO[0][9] = [0.45, 0.55, 0.65, 0.725] # destination output of layer 9 is [0.45, 0.55, 0.65, 0.725]

    # neural network learning
    (useless, matrix, wM, oM, tM, fwM) = Backpropagation(input_, destO, neurons, matrix, printDetail, lr, prt, maxError, modifyWM)

    # make test input data
    testInput = [[], [], [], [], [], [], [], [], [], []] # test input data
    testInput[0] = [111, 222, 333]
    testInput[3] = [1, 3, 4]
    testInput[6] = [4, 1, 0]

    # neural network test
    test(testInput, matrix, wM, oM, tM, fwM)
