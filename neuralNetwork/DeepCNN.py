import os, imp
imp.load_source('CNN', os.path.join(os.path.dirname(__file__), "../CNN.py"))
import CNN
import neuralNetwork as NN
import math
import random

if __name__ == '__main__':
    # ConvolWidth : number of columns in each convolution result
    # ConvolHeight : number of rows in each convolution result
    # resultArrays : convolution result of each image data (N images),
    #                (length of [0], [1], ..., [N-1]) = ConvolWidth * ConvolHeight * Filters
    # testArray : convolution result of test image data (1 image),
    #             (length of [0]) = ConvolWidth * ConvolHeight * Filters
    
    (data, rows, cols, images, filtersize, filters, testimgdata, prt) = CNN.getData()

    # 1. make INPUT ARRAY : resultArrays
    # make filters
    flts = CNN.makeFilters(data, rows, cols, images, filtersize, filters, prt)
    # CNN using the filter - data image
    resultArrays = CNN.CNN(data, rows, cols, images, filtersize, filters, flts, prt)
    # CNN using the filter - test image
    testArray = CNN.CNN([testimgdata], rows, cols, 1, filtersize, filters, flts, prt)

    print(' **** INPUT: convolution result ****')
    for i in range(len(resultArrays)):
        if len(resultArrays[i]) > 20:
            prtVec = NN.printVector(resultArrays[i][0:20], 3)
            print(str(prtVec[0:len(prtVec)-2]) + ' ... ]')
        else: print(NN.printVector(resultArrays[i], 3))
    print('')
    print(' **** convolution result for test data ****')
    if len(resultArrays[i]) > 20:
        prtVec = NN.printVector(testArray[0][0:20], 3)
        print(str(prtVec[0:len(prtVec)-2]) + ' ... ]')
    else: print(NN.printVector(testArray[0], 3))
    print('')

    # 2. make OUTPUT ARRAY : one-hot vectors
    imgLabels = []
    for i in range(images):
        imgLabels.append(data[i][0])
    imgLabelSet = list(set(imgLabels))

    output_ = [[0]*len(imgLabelSet) for i in range(images)]
    for i in range(images):
        for j in range(len(imgLabelSet)):
            if imgLabelSet[j] == imgLabels[i]: output_[i][j] = 1

    print(' **** OUTPUT: one-hot vector for each image ****')
    print('< labels >')
    print(imgLabels)
    print('')
    print('< labels (removed duplicated) >')
    print(imgLabelSet)
    print('')
    print('< OUTPUT: one-hot vectors >')
    print(output_)
    
    # 3. NEURAL NETWORK
    # make matrix
    matrix = [[0]*5 for i in range(5)]
    for i in range(4): matrix[i][i+1] = 1

    # make input data
    inputData = []
    for i in range(images): inputData.append([resultArrays[i], [], [], [], []])

    # make destination output data
    destOutput = []
    for i in range(images): destOutput.append([[], [], [], [], output_[i]])

    # make test data
    testData = [testArray[0], [], [], [], []]

    # number of neurons for each layer
    neurons = [len(resultArrays[0]), 10, 10, 10, len(imgLabelSet)]
    
    # Neural Network learning
    (useless, matrix, wM, oM, tM, fwM) = NN.Backpropagation(inputData, destOutput, neurons, matrix, -2, 3.25, 0, 0.25, NN.modifyWM)

    # test
    testResult = NN.test(testData, matrix, wM, oM, tM, fwM)
    print(testResult)

    # print result
    maxIndex = 0
    maxValue = -1
    for i in range(len(testResult[0])):
        if maxValue < testResult[0][i]:
            maxIndex = i
            maxValue = testResult[0][i]
    print('')
    print('classified: ' + str(imgLabelSet[maxIndex]))
