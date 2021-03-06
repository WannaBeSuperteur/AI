import math
import random

# read data from file
def getData():
    # get data
    file = open('DNN.txt', 'r')
    read = file.readlines()
    file.close

    input_ = [] # input
    output_ = [] # desired output

    # read input and output
    for i in range(0, len(read)-1):
        row = read[i].split('\n')[0].split('/')
        input_.append(row[0].split(' '))
        output_.append(row[1].split(' '))
    # read test data
    testdata = read[len(read)-1].split('\n')[0].split(' ')
        
    return (input_, output_, testdata)

# activation function
def sigmoid(value, stocastic):
    # stocastic node
    if stocastic != 0:
        result = 1/(1+math.exp(-value))
        if random.random() < result: return 1
        else: return 0
    # basic sigmoid function
    return 1/(1+math.exp(-value))

# forward propagation
def forward(input_, d, stoc, printDetail, last, iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw):
    # input layer -> hidden layer 0
    # j INPUTs and i HIDDEN0s
    hidden0Input = [] # hidden layer 0 input
    for i in range(h0Nn):
        hX = -h0Nt[i]
        for j in range(iNn):
            hX += h0Nw[i][j]*float(input_[d][j])
        hidden0Input.append(hX)

    if printDetail >= 2 or last == 1:
        his = 'Hidden Layer 0 Input: [ '
        for i in range(h0Nn): his += str(round(hidden0Input[i], 6)) + '  '
        print(his + ']')

    # find hidden layer 0 output
    hidden0Output = [] # hidden layer 0 output
    for i in range(h0Nn): hidden0Output.append(sigmoid(hidden0Input[i], stoc))

    if printDetail >= 2 or last == 1:
        hos = 'Hidden Layer 0 Output: [ '
        for i in range(h0Nn): hos += str(round(hidden0Output[i], 6)) + '  '
        print(hos + ']')

    # hidden layer 0 -> hidden layer 1
    # j HIDDEN0s and i HIDDEN1s
    hidden1Input = [] # hidden layer 1 input
    for i in range(h1Nn):
        hX = -h1Nt[i]
        for j in range(h0Nn):
            hX += h1Nw[i][j]*float(hidden0Output[j])
        hidden1Input.append(hX)

    if printDetail >= 2 or last == 1:
        his = 'Hidden Layer 1 Input: [ '
        for i in range(h1Nn): his += str(round(hidden1Input[i], 6)) + '  '
        print(his + ']')

    # find hidden layer 1 output
    hidden1Output = [] # hidden layer 1 output
    for i in range(h1Nn): hidden1Output.append(sigmoid(hidden1Input[i], stoc))

    if printDetail >= 2 or last == 1:
        hos = 'Hidden Layer 1 Output: [ '
        for i in range(h1Nn): hos += str(round(hidden1Output[i], 6)) + '  '
        print(hos + ']')

    # hidden layer 1 -> hidden layer 2
    # j HIDDEN1s and i HIDDEN2s
    hidden2Input = [] # hidden layer 2 input
    for i in range(h2Nn):
        hX = -h2Nt[i]
        for j in range(h1Nn):
            hX += h2Nw[i][j]*float(hidden1Output[j])
        hidden2Input.append(hX)

    if printDetail >= 2 or last == 1:
        his = 'Hidden Layer 2 Input: [ '
        for i in range(h2Nn): his += str(round(hidden2Input[i], 6)) + '  '
        print(his + ']')

    # find hidden layer 2 output
    hidden2Output = [] # hidden layer 2 output
    for i in range(h2Nn): hidden2Output.append(sigmoid(hidden2Input[i], stoc))

    if printDetail >= 2 or last == 1:
        hos = 'Hidden Layer 2 Output: [ '
        for i in range(h2Nn): hos += str(round(hidden2Output[i], 6)) + '  '
        print(hos + ']')

    # hidden layer 2 -> output layer
    # j HIDDEN2s and i OUTPUTs
    outputInput = [] # output layer input
    for i in range(oNn):
        oX = -oNt[i]
        for j in range(h2Nn):
            oX += oNw[i][j]*float(hidden2Output[j])
        outputInput.append(oX)

    if printDetail >= 2 or last == 1:
        ois = 'Output Layer Input: [ '
        for i in range(oNn): ois += str(round(outputInput[i], 6)) + '  '
        print(ois + ']')

    # find output layer output
    outputOutput = [] # output layer output
    for i in range(oNn): outputOutput.append(sigmoid(outputInput[i], stoc))

    if printDetail >= 2 or last == 1:
        oos = 'Output Layer Output: [ '
        for i in range(oNn): oos += str(round(outputOutput[i], 6)) + '  '
        print(oos + ']')

    return (hidden0Input, hidden0Output, hidden1Input, hidden1Output, hidden2Input, hidden2Output, outputInput, outputOutput)

# backpropagation (return updated weights)
def Back(hidden0Output, hidden1Output, hidden2Output, outputOutput, iNn, h0Nn, h0Nw, h1Nn, h1Nw, h2Nn, h2Nw, oNn, oNw, input_, output_, d, lr):
    # find So and Sh
    # So = (do-Oo)*Oo*(1-Oo)
    # Sh2 = Sum(So*Wh2o) * Oh2*(1-Oh2)
    # Sh1 = Sum(Sh2*Wh1h2) * Oh1*(1-Oh1)
    # Sh0 = Sum(Sh1*Wh0h1) * Oh0*(1-Oh0)
    So = [] # for each output layer neuron
    for i in range(oNn):
        Oo = outputOutput[i]
        So.append((float(output_[d][i])-Oo)*Oo*(1-Oo))
                
    Sh2 = [] # for each hidden layer 2 neuron
    Sh1 = [] # for each hidden layer 1 neuron
    Sh0 = [] # for each hidden layer 0 neuron
    for i in range(h2Nn):
        Sum = 0
        for j in range(oNn):
            Sum += So[j]*oNw[j][i]
        Oh2 = hidden2Output[i]
        Sh2.append(Sum*Oh2*(1-Oh2))
    for i in range(h1Nn):
        Sum = 0
        for j in range(h2Nn):
            Sum += Sh2[j]*h2Nw[j][i]
        Oh1 = hidden1Output[i]
        Sh1.append(Sum*Oh1*(1-Oh1))
    for i in range(h0Nn):
        Sum = 0
        for j in range(h1Nn):
            Sum += Sh1[j]*h1Nw[j][i]
        Oh0 = hidden0Output[i]
        Sh0.append(Sum*Oh0*(1-Oh0))

    # update Hidden2-Output weights (Wh2o += lr*So*Oh2)
    # j HIDDEN2s and i OUTPUTs
    for i in range(oNn):
        for j in range(h2Nn):
            oNw[i][j] += lr*So[i]*hidden2Output[j]

    # update Hidden1-Hidden2 weights (Wh1h2 += lr*Sh2*Oh1)
    # j HIDDEN1s and i HIDDEN2s
    for i in range(h2Nn):
        for j in range(h1Nn):
            h2Nw[i][j] += lr*Sh2[i]*hidden1Output[j]

    # update Hidden0-Hidden1 weights (Wh0h1 += lr*Sh1*Oh0)
    # j HIDDEN0s and i HIDDEN1s
    for i in range(h1Nn):
        for j in range(h0Nn):
            h1Nw[i][j] += lr*Sh1[i]*hidden0Output[j]

    # update Input-Hidden0 weights (Wih0 += lr*Sh0*Xi)
    # j INPUTs and i HIDDEN0s
    for i in range(h0Nn):
        for j in range(iNn):
            h0Nw[i][j] += lr*Sh0[i]*float(input_[d][j])

    return (h0Nw, h1Nw, h2Nw, oNw)

# initialize weight and threshold
def initWeightAndThreshold(iNn, h0Nn, h1Nn, h2Nn, oNn, h0Nw, h1Nw, h2Nw, oNw, h0Nt, h1Nt, h2Nt, oNt):
    # set weights and thresholds
    for i in range(h0Nn):
        weights = [] # array of weight of hidden layer 0 neurons
        for j in range(iNn):
            weights.append(random.random())
        h0Nw.append(weights) # iNs weights/neuron
        h0Nt.append(random.random()) # 1 threshold/neuron
    for i in range(h1Nn):
        weights = [] # array of weight of hidden layer 1 neurons
        for j in range(h0Nn):
            weights.append(random.random())
        h1Nw.append(weights) # iNs weights/neuron
        h1Nt.append(random.random()) # 1 threshold/neuron
    for i in range(h2Nn):
        weights = [] # array of weight of hidden layer 2 neurons
        for j in range(h1Nn):
            weights.append(random.random())
        h2Nw.append(weights) # iNs weights/neuron
        h2Nt.append(random.random()) # 1 threshold/neuron
    for i in range(oNn):
        weights = [] # array of weight of output layer neurons
        for j in range(h2Nn):
            weights.append(random.random())
        oNw.append(weights) # hNs weights/neuron
        oNt.append(random.random()) # 1 threshold/neuron

# print neuron info
def printNeuronInfo(iNn, h0Nn, h1Nn, h2Nn, oNn, h0Nt, h0Nw, h1Nt, h1Nw, h2Nt, h2Nw, oNt, oNw):
    print('')
    print('<HIDDEN 0>')
    for i in range(h0Nn):
        print('Neuron ' + str(i) + ': thr = ' + str(round(h0Nt[i], 6)))
        for j in range(iNn):
            print('[ inputN ' + str(j) + ' ] weight = ' + str(round(h0Nw[i][j], 6)))
    print('')
    print('<HIDDEN 1>')
    for i in range(h1Nn):
        print('Neuron ' + str(i) + ': thr = ' + str(round(h1Nt[i], 6)))
        for j in range(h0Nn):
            print('[ hidden0N ' + str(j) + ' ] weight = ' + str(round(h1Nw[i][j], 6)))
    print('')
    print('<HIDDEN 2>')
    for i in range(h2Nn):
        print('Neuron ' + str(i) + ': thr = ' + str(round(h2Nt[i], 6)))
        for j in range(h1Nn):
            print('[ hidden1N ' + str(j) + ' ] weight = ' + str(round(h2Nw[i][j], 6)))
    print('')
    print('<OUTPUT>')
    for i in range(oNn):
        print('Neuron ' + str(i) + ': thr = ' + str(round(oNt[i], 6)))
        for j in range(h2Nn):
            print('[ hidden2N ' + str(j) + ' ] weight = ' + str(round(oNw[i][j], 6)))    

# train Neural Network
def Backpropagation(input_, output_, h0Nn, h1Nn, h2Nn, lr, printDetail, testdata, stoc, wReturn):

    iNn = len(input_[0]) # number of input neurons
    oNn = len(output_[0]) # number of output neurons

    # make input, hidden and output layer
    # weight
    h0Nw = [] # for hidden layer 0
    h1Nw = [] # for hidden layer 1
    h2Nw = [] # for hidden layer 2
    oNw = [] # for output layer
    # threshold
    h0Nt = [] # for hidden layer 0
    h1Nt = [] # for hidden layer 0
    h2Nt = [] # for hidden layer 0
    oNt = [] # for output layer

    # initialize weight and threshold
    initWeightAndThreshold(iNn, h0Nn, h1Nn, h2Nn, oNn, h0Nw, h1Nw, h2Nw, oNw, h0Nt, h1Nt, h2Nt, oNt)

    # repeat until convergence
    last = 0 # check if last loop
    count = 0
    while 1:
        count += 1

        # print neuron info
        if printDetail >= -1 or count == 1 or last == 1 or count % 100 == 0:
            print('')
            print('ROUND ' + str(count))
        if printDetail >= 2 or count == 1 or last == 1:
            printNeuronInfo(iNn, h0Nn, h1Nn, h2Nn, oNn, h0Nt, h0Nw, h1Nt, h1Nw, h2Nt, h2Nw, oNt, oNw)

        # save current weights
        h0Nw_ = []
        h1Nw_ = []
        h2Nw_ = []
        oNw_ = []
        for i in range(h0Nn):
            h0Nw__ = []
            for j in range(iNn):
                h0Nw__.append(round(h0Nw[i][j], 6))
            h0Nw_.append(h0Nw__)
        for i in range(h1Nn):
            h1Nw__ = []
            for j in range(h0Nn):
                h1Nw__.append(round(h1Nw[i][j], 6))
            h1Nw_.append(h1Nw__)
        for i in range(h2Nn):
            h2Nw__ = []
            for j in range(h1Nn):
                h2Nw__.append(round(h2Nw[i][j], 6))
            h2Nw_.append(h2Nw__)
        for i in range(oNn):
            oNw__ = []
            for j in range(h2Nn):
                oNw__.append(round(oNw[i][j], 6))
            oNw_.append(oNw__)

        # learning
        # for each training data
        error = []
        for d in range(len(input_)):
            
            # if last loop, about test data
            if last == 1:
                if wReturn != 0: return (h0Nw, h1Nw, h2Nw, oNw)
                input_[d] = testdata
            
            if printDetail >= 2 or last == 1:
                print('')
                print('input data    : ' + str(input_[d]))
                # do not print output if last loop
                if last <= 0: print('desired output: ' + str(output_[d]))

            # forward propagation
            (hidden0Input, hidden0Output, hidden1Input, hidden1Output, hidden2Input, hidden2Output, outputInput, outputOutput) = forward(input_, d, stoc, printDetail, last, iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw)

            # for test data, no need of backpropagation
            if last == 1: break

            # backpropagation
            (h0Nw, h1Nw, h2Nw, oNw) = Back(hidden0Output, hidden1Output, hidden2Output, outputOutput, iNn, h0Nn, h0Nw, h1Nn, h1Nw, h2Nn, h2Nw, oNn, oNw, input_, output_, d, lr)

            # calculate the sum of error
            if printDetail >= 2: print('')
            for i in range(oNn):
                if printDetail >= 1:
                    print('output=' + str(round(outputOutput[i], 6)) + ', desired=' + str(output_[d][i]))
                error.append(outputOutput[i]-float(output_[d][i]))

        if last == 1: break                

        # check stop condition (set last to 1)
        errorSum = 0
        if printDetail >= 0: print('')
        for i in range(len(error)):
            errorSum += abs(error[i])
            if printDetail >= 0:
                print('error for input ' + str(i) + ': ' + str(round(error[i], 6)))
        if printDetail >= -1 or count % 100 == 0:
            print('sum of error: ' + str(round(errorSum, 6)))
        if errorSum / (len(input_) * len(output_[0])) < 0.001 or (count >= 20000 and printDetail >= -1) or count >= 1000000:
            last = 1

        # print change of weight
        if printDetail >= 1:
            print('')
            print('<weight change>')
            print('HIDDEN0 neurons:')
            for i in range(h0Nn):
                print('HIDDEN0 Neuron ' + str(i) + ':')
                for j in range(iNn):
                    before = str(round(h0Nw_[i][j], 6))
                    after = str(round(h0Nw[i][j], 6))
                    print('[ inputN ' + str(j) + ' ] weight = ' + before + '->' + after)
            print('HIDDEN1 neurons:')
            for i in range(h1Nn):
                print('HIDDEN1 Neuron ' + str(i) + ':')
                for j in range(h0Nn):
                    before = str(round(h1Nw_[i][j], 6))
                    after = str(round(h1Nw[i][j], 6))
                    print('[ hidden0N ' + str(j) + ' ] weight = ' + before + '->' + after)
            print('HIDDEN2 neurons:')
            for i in range(h2Nn):
                print('HIDDEN2 Neuron ' + str(i) + ':')
                for j in range(h1Nn):
                    before = str(round(h2Nw_[i][j], 6))
                    after = str(round(h2Nw[i][j], 6))
                    print('[ hidden1N ' + str(j) + ' ] weight = ' + before + '->' + after)
            print('OUTPUT neurons:')
            for i in range(oNn):
                print('OUTPUT Neuron ' + str(i) + ':')
                for j in range(h2Nn):
                    before = str(round(oNw_[i][j], 6))
                    after = str(round(oNw[i][j], 6))
                    print('[ hidden2N ' + str(j) + ' ] weight = ' + before + '->' + after)

    # return layer info (number of neurons, thresholds and weights)
    return (iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw)

if __name__ == '__main__':
    (input_, output_, testdata) = getData()
    Backpropagation(input_, output_, 6, 6, 6, 3.25, -1, testdata, 0, 0)
