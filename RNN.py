import math
import random

# read data from file
def getData():
    # get data
    file = open('RNN.txt', 'r')
    read = file.readlines()
    file.close

    temp = read[0].split('\n')[0].split(' ')
    steps = int(temp[0]) # number of steps
    hNn = int(temp[1])
    offset = int(temp[2]) # offset between input and destination data
    
    # read data to learn and test data
    tolearn = read[1].split('\n')[0]
    testdata = read[2].split('\n')[0]

    prt = int(read[3]) # print data?
        
    return (steps, hNn, tolearn, testdata, prt, offset)

# activation function (tanh)
def activate(value):
    return 1/(1+math.exp(-value))

# print array
def printArray(array):
    result = '['
    for i in range(len(array)):
        result += ' ' + str(round(array[i], 3)) + ' '
    return result + ']'

# forward propagation
def forwardPropagation(input_, hiddeni, hidden_, outputi, output_, hNw, oNw, hNhNw):
    steps = len(input_)

    iNn = len(input_[0])
    hNn = len(hidden_[0])
    oNn = len(output_[0])
    
    ## forward propagation
    # about next hidden layers : st = f(U*x[t] + W*s[t-1])
    for i in range(steps): # for each step
        for j in range(hNn): # for each hidden layer
            sumProduct = 0
            for k in range(iNn): # for each input unit
                sumProduct += (input_[i][k] * hNw[i][j][k])
            for k in range(hNn): # for each hidden unit
                if i >= 1:
                    sumProduct += (hidden_[i-1][k] * hNhNw[i][j][k])
                    
            hiddeni[i][j] = sumProduct
            hidden_[i][j] = activate(sumProduct)
                    
    # about output layers : o[t] = f(V*s[t])
    for i in range(steps):
        for j in range(oNn):
            sumProduct = 0
            for k in range(hNn):
                sumProduct += hidden_[i][k] * oNw[i][j][k]

            outputi[i][j] = sumProduct
            output_[i][j] = activate(sumProduct)

# print weight
def printWeight(steps, hNn, oNn, hNw, oNw, hNhNw):
    for t in range(steps):
        print('** STEP ' + str(t) + ' **')
        print('input-hidden weight')
        for i in range(hNn):
            print(' [' + str(i) + '] : ' + printArray(hNw[t][i]))
        print('')
        print('hidden-output weight')
        for i in range(oNn):
            print(' [' + str(i) + '] : ' + printArray(oNw[t][i]))
        print('')

    print('** HIDDEN-HIDDEN **')
    for t in range(steps-1):
        print('hidden-hidden weight')
        for i in range(hNn):
            print(' [' + str(i) + '] : ' + printArray(hNhNw[t][i]))
        print('')

# return mul of array
def arrayMul(array1, array2):
    while str(array1)[1] != '[': array1 = [array1]
    while str(array2)[1] != '[': array2 = [array2]
    
    result = []
    for i in range(len(array1)):
        temp = []
        for j in range(len(array2[0])):
            sumProduct = 0
            for k in range(len(array1[0])):
                sumProduct += (array1[i][k] * array2[k][j])
            temp.append(sumProduct)
        result.append(temp)
    return result

# vector derivatiion (dVEC1/dVEC2)
def vecD(vec1, vec2):
    ele1 = len(vec1)
    ele2 = len(vec2)
    result = [[0]*ele2 for i in range(ele1)]

    for i in range(ele1):
        for j in range(ele2):
            result[i][j] = ele1*(1-ele1)

    return result

# train Neural Network
def RNN(steps, hNn, lr, tolearn, testdata, prt, offset):

    print('    +------------------+')
    print('    |     TRAINING     |')
    print('    +------------------+')
    print('')

    # decide iNn (number of input neurons)
    alldata = tolearn + testdata
    setInp = set(alldata) # set of input data (by character)
    print(setInp)
    iNn = len(setInp) # number of input neurons in each step
    oNn = len(setInp) # number of output neurons in each step
    listInp = list(setInp) # make list of characters in 'tolearn' and 'testdata'

    # make input, hidden and output neurons
    input_ = [[0.25]*iNn for i in range(steps)] # input data
    hiddeni = [[0]*hNn for i in range(steps)] # hidden layer input
    hidden_ = [[0]*hNn for i in range(steps)] # hidden layer output
    outputi = [[0]*oNn for i in range(steps)] # output layer input
    output_ = [[0]*oNn for i in range(steps)] # output layer output
    dest_ = [[0.25]*iNn for i in range(steps)] # destination output

    hNw = [[[0]*iNn for j in range(hNn)] for i in range(steps)]
    oNw = [[[0]*hNn for j in range(oNn)] for i in range(steps)]
    hNhNw = [[[random.random()]*hNn for j in range(hNn)] for i in range(steps)]

    # initialize weight
    for t in range(steps):
        for i in range(hNn):
            for j in range(iNn):
                hNw[t][i][j] = random.random()*2-1
        for i in range(oNn):
            for j in range(hNn):
                oNw[t][i][j] = random.random()*2-1

    # initialize weight between hidden layers
    for t in range(steps-1):
        for i in range(hNn):
            for j in range(hNn):
                hNhNw[t][i][j] = random.random()*2-1

    # RNN learning
    count = 0
    while 1:
        count += 1
        if prt >= 1:
            print('******** ROUND ' + str(count) + ' ********')
            if prt >= 2: print('')
        
        ## initialize
        # initialize input and destination data(step 0 to iNn-1)
        for i in range(steps):
            input_[i][listInp.index(tolearn[i])] = 0.75
            dest_[i][listInp.index(tolearn[i+offset])] = 0.75

        ## forward propagation
        forwardPropagation(input_, hiddeni, hidden_, outputi, output_, hNw, oNw, hNhNw)

        # print neuron info
        if prt >= 2:
            for i in range(steps):
                print('< STEP ' + str(i) + ' >')
                print('input  layer : ' + printArray(input_[i]))
                print('hidden layer : ' + printArray(hidden_[i]))
                print('output layer : ' + printArray(output_[i]))
                print('dest   output: ' + printArray(dest_[i]))
                print('')

        ## backpropagation
        # using least-square: error = -Sum(i){(D[i] * O[i])^2}/2
        
        # update dE/dW, dE/dU, dE/dV
        DmO = [] # D[t]-O[t] = error using least-square
        for i in range(steps):

            # calculate D[t]-O[t] for each t
            DmOtemp = []
            for j in range(oNn):
                DmOtemp.append(dest_[i][j] - output_[i][j])
            DmO.append(DmOtemp)

        dEdW = [[[0]*hNn for j in range(hNn)] for i in range(steps-1)]
        dEdU = [[[0]*iNn for j in range(hNn)] for i in range(steps)]
        dEdV = [[[0]*hNn for j in range(oNn)] for i in range(steps)]
        
        SO = [[0]*oNn for i in range(steps)] # SO = (d-O)*O*(1-O)
        SH = [[0]*hNn for i in range(steps)] # SH = Sum(SO*W*O*(1-O))
        temp = [[0]*hNn for i in range(steps)] # start calculating from SH
        for t in range(steps):
            
            # E[t] : error at step t
            # D[t] : destination vector at step t (dest_[t])
            
            # U[t] : input-hidden weight at step t (hNw[t])
            # V[t] : hidden-output weight at step t (oNw[t])
            # W[i,i+1] : hidden-hidden weight (hNhNw)

            # K[t] : hidden layer input at step t (hiddeni[t])
            # S[t] : hidden layer output at step t (hidden_[t])
            # Z[t] : output layer input at step t (outputi[t])
            # O[t] : output layer output at step t (output_[t])

            # dE[t]/dV[t]     = dE[t]/dO[t] * dO[t]/dZ[t] * dZ[t]/dV[t]
            for i in range(oNn):
                SO[t][i] = DmO[t][i] * output_[t][i] * (1-output_[t][i])
                for j in range(hNn):
                    dEdV[t][i][j] = SO[t][i] * hidden_[t][j]

            # calculate SH = Sum(SO*W)*H*(1-H)
            for i in range(hNn):
                SH[t][i] = 0
                for j in range(oNn):
                    SH[t][i] += SO[t][j] * oNw[t][j][i]
                SH[t][i] *= (hidden_[t][j] * (1-hidden_[t][j]))
                temp[t][i] = SH[t][i]
            
            # dE[t]/dW        = sum of dE[t]/dW[k,k+1]
            # dE[t]/dW[k,k+1] = (dE[t]/dO[t] * dO[t]/dZ[t] * dZ[t]/dS[t]) *
            #                   (dS[t]/dW[t-1,t] * dS[t-1]/dW[t-2,t-1] * ... * dS[k+1]/dW[k,k+1])
            # (k for 0, 1, 2, ..., t-1)
            for k in range(t):

                # multiply hidden-hidden weights
                for u in range(t-1, k-1, -1):
                    # calculate next value of S[layer] -> temp2
                    for i in range(hNn):
                        temp[u][i] = 0
                        for j in range(hNn):
                            temp[u][i] += temp[u+1][j] * hNhNw[u][j][i]
                        temp[u][i] *= (hidden_[u][j] * (1-hidden_[u][j]))
                        
                # calculate final dEdW
                for i in range(hNn):
                    for j in range(hNn):
                        dEdW[k][i][j] += temp[k+1][i] * hidden_[k][j]

            # dE[t]/dU        = sum of dE[t]/dU[k]
            # dE[t]/dU[k]     = (dE[t]/dO[t] * dO[t]/dZ[t] * dZ[t]/dS[t]) *
            #                   (dS[t]/dU[t] * dS[t-1]/dU[t-1] * ... * dS[k]/dU[k])
            # (k for 0, 1, 2, ..., t)
            for k in range(t+1):
                # calculate final dEdU
                for i in range(hNn):
                    for j in range(iNn):
                        dEdU[k][i][j] += temp[k][i] * input_[k][j]

        # update weight using dE/dW, dE/dU and dE/dV
        for t in range(steps):
            for i in range(hNn):
                for j in range(iNn):
                    hNw[t][i][j] += lr * dEdU[t][i][j] # update input-hidden
            for i in range(oNn):
                for j in range(hNn):
                    oNw[t][i][j] += lr * dEdV[t][i][j] # update hidden-output
        for t in range(steps-1):
            for i in range(hNn):
                for j in range(hNn):
                    hNhNw[t][i][j] += lr * dEdW[t][i][j] # update hidden-hidden

        # print changed weight
        if prt >= 3:
            print('**** WEIGHT CHANGED ****')
            printWeight(steps, hNn, oNn, hNw, oNw, hNhNw)

        ## calculate sum of error
        totalErrorSum = 0 # sum of errorSum
        for i in range(steps):
            errorSum = 0
            for j in range(oNn):
                errorSum += DmO[i][j] # sum of error for each step
            totalErrorSum += abs(errorSum)

            if prt >= 1: print('Sum of error at step ' + str(i) + ': ' + str(round(errorSum, 6)))

        if prt >= 2: print('')
        if prt >= 0 or count % 100 == 0:
            print('Sum of error (ROUND ' + str(count) + '): ' + str(round(totalErrorSum, 6)))
            if prt == 0 and count % 10 == 0: print('')
        if prt >= 1: print('')

        # if sum of error < 0.001 -> return layers
        if totalErrorSum < 0.001:
            print('')
            print('<FINAL>')
            for i in range(steps):
                print('< STEP ' + str(i) + ' >')
                print('hidden layer: ' + printArray(hidden_[i]))
                print('output layer: ' + printArray(output_[i]))
                print('')

            # print weight
            print('**** FINAL WEIGHT INFO ****')
            printWeight(steps, hNn, oNn, hNw, oNw, hNhNw)
            
            return (listInp, hNw, oNw, hNhNw, steps)

# print max value
def printMax(array, listInp):
    
    result = ''
    for i in range(len(array)):
        maxIndex = 0 # index of element that has max value
        maxValue = 0 # value of element at maxIndex

        for j in range(len(array[0])):    
            if array[i][j] > maxValue:
                maxIndex = j
                maxValue = array[i][j]

        val = str(listInp[maxIndex])
        print('value        : ' + val)
        result += val
    print('result : ' + result)

# return output using RNN layers
def getOutput(testdata, listInp, hNw, oNw, hNhNw, steps):

    print('    +------------------+')
    print('    |       TEST       |')
    print('    +------------------+')
    print('')

    iNn = len(listInp) # number of input neurons in each step
    hNn = len(hNw[0]) # number of hidden neurons in each step
    oNn = len(listInp) # number of output neurons in each step

    # print weight
    print('**** WEIGHT ****')
    printWeight(steps, hNn, oNn, hNw, oNw, hNhNw)
    
    # make data
    input_ = [[0.25]*iNn for i in range(steps)] # input data (test data)
    for i in range(steps):
        input_[i][listInp.index(testdata[i])] = 0.75
    hiddeni = [[0]*hNn for i in range(steps)] # hidden layer input
    hidden_ = [[0]*hNn for i in range(steps)] # hidden layer output
    outputi = [[0]*oNn for i in range(steps)] # output layer input
    output_ = [[0]*oNn for i in range(steps)] # output layer output

    # find max element in input layer
    print('******** TEST INPUT ********')
    printMax(input_, listInp)
    print('')
    
    print('******** TEST OUTPUT ********')

    # calculate output
    ## forward propagation
    forwardPropagation(input_, hiddeni, hidden_, outputi, output_, hNw, oNw, hNhNw)

    for i in range(steps):
        print('< STEP ' + str(i) + ' >')
        print('input  layer : ' + printArray(input_[i]))
        print('hidden layer : ' + printArray(hidden_[i]))
        print('output layer : ' + printArray(output_[i]))
        print('')

    # find max element in output layer
    printMax(output_, listInp)
        
(steps, hNn, tolearn, testdata, prt, offset) = getData()

# make layers using RNN
(listInp, hNw, oNw, hNhNw, steps) = RNN(steps, hNn, 3.25, tolearn, testdata, prt, offset)
getOutput(testdata, listInp, hNw, oNw, hNhNw, steps)
