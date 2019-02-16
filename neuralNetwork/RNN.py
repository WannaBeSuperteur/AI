import math
import random
import neuralNetwork as NN

# read data from file
def getData():
    # get data
    file = open('RNN.txt', 'r')
    read = file.readlines()
    file.close

    # input,and test data
    input_ = []
    temp = read[0].split('\n')[0]
    for i in range(len(temp)): input_.append(temp[i])
    test_ = []
    temp = read[1].split('\n')[0]
    for i in range(len(temp)): test_.append(temp[i])

    # configurations
    config = read[2].split('\n')[0].split(' ')
    printDetail = int(config[0])
    prt = int(config[1])
    lr = float(config[2])
    maxError = float(config[3])
    Ns = int(config[4]) # number of hidden neurons for each hidden layer

    return (input_, printDetail, lr, prt, maxError, Ns, test_)

if __name__ == '__main__':
    (input_, printDetail, lr, prt, maxError, Ns, test_) = getData()

    L = len(input_)-1 # length of input (must be same value as length of output)
    inputSet = list(set(input_))
    kinds = len(inputSet) # number of kinds of input characters

    # inputArr and testInputArr : one-hot array of input
    # for example, ['0'] -> [1, 0, 0], ['1'] -> [0, 1, 0], ['2'] -> [0, 0, 1]
    inputArr = []
    testInputArr = []

    # for inputArr
    for i in range(len(input_)):
        temp = [] # one-hot array
        for j in range(kinds): temp.append(0.25)

        index = 0 # index of value input_[i] in inputSet
        for j in range(kinds):
            if inputSet[j] == input_[i]:
                index = j
                break
        temp[index] = 0.75
        inputArr.append(temp)

    # for testInputArr
    for i in range(len(test_)):
        temp = [] # one-hot array
        for j in range(kinds): temp.append(0.25)

        index = 0 # index of value test_[i] in inputSet
        for j in range(kinds):
            if inputSet[j] == test_[i]:
                index = j
                break
        temp[index] = 0.75
        testInputArr.append(temp)

    # make matrix
    matrix = [[0]*(L*3) for i in range(L*3)]
    for i in range(L):
        matrix[i*3][i*3+1] = 1 # connect input-hidden
        matrix[i*3+1][i*3+2] = 1 # connect hidden-output
        if i != L-1: matrix[i*3+1][i*3+4] = 1 # connect hidden-hidden

    # make input data
    inputData = []
    temp = []
    for i in range(L*3):
        if i % 3 == 0: temp.append(inputArr[int(i/3)])
        else: temp.append([])
    inputData.append(temp)

    # make destination output data
    destOutput = []
    temp = []
    for i in range(L*3):
        if i % 3 == 2: temp.append(inputArr[int((i+1)/3)])
        else: temp.append([])
    destOutput.append(temp)

    # make test data
    testData = []
    for i in range(L*3):
        if i % 3 == 0: testData.append(testInputArr[int(i/3)])
        else: testData.append([])

    # number of neurons for each layer
    neurons = []
    for i in range(L*3):
        if i % 3 == 0: neurons.append(kinds) # input
        elif i % 3 == 2: neurons.append(kinds) # output
        else: neurons.append(Ns) # hidden

    print('< INPUT DATA >')
    print(inputData)
    print('')
    print('< DEST OUTPUT >')
    print(destOutput)
    print('')
    print('< TEST DATA >')
    print(testData)
    print('')
    print('< NUMBER OF NEURONS IN EACH LAYER >')
    print(neurons)
    print('')
    
    # RNN learning
    (useless, matrix, wM, oM, tM, fwM) = NN.Backpropagation(inputData, destOutput, neurons, matrix, printDetail, lr, prt, maxError, NN.modifyWM)

    # test
    testResult = NN.test(testData, matrix, wM, oM, tM, fwM)
    testOutput = []
    for i in range(len(testResult)):
        # for each item of test result, find index of the maximum output value
        maxIndex = 0 # index of maximum output
        maxValue = 0.0 # maximum output value
        for j in range(kinds):
            if maxValue < testResult[i][j]:
                maxValue = testResult[i][j]
                maxIndex = j
                
        testOutput.append(inputSet[maxIndex])

    print('')
    print('< TEST RESULT >')
    print(testOutput)
