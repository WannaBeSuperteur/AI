import math
import random
import StackedRBM as sr
import RBMPackage as rp

# read data from file
def getData():
    # get data
    file = open('DeepBeliefNetwork.txt', 'r')
    read = file.readlines()
    file.close

    input_ = [] # input
    output_ = [] # desired output

    # read number of hidden layer neurons
    hiddens = read[0].split('\n')[0].split(' ')
    for i in range(len(hiddens)):
        hiddens[i] = int(hiddens[i])
    # read input and output
    for i in range(1, len(read)-1):
        row = read[i].split('\n')[0].split('/')
        input_.append(row[0].split(' '))
        output_.append(row[1].split(' '))
    # read test data
    testdata = read[len(read)-1].split('\n')[0].split(' ')
        
    return (input_, output_, hiddens, testdata)

# input data -> hidden layer output
def inputToOutput(inp, weight0, weight1, weight2, count):
    inplen = len(inp) # length of input data
    hid0N = len(weight0[0]) # numbers of neurons in hidden layer 0
    hid1N = len(weight1[0]) # numbers of neurons in hidden layer 1
    hid2N = len(weight2[0]) # numbers of neurons in hidden layer 2

    # input -> hidden 0
    h0 = [] # output of hidden layer 0
    for i in range(hid0N):
        weightSum = 0
        for j in range(inplen):
            weightSum += float(inp[j]) * weight0[j][i]
        h0.append(sr.sigmoidStocastic(weightSum))

    # hidden 0 -> hidden 1
    h1 = [] # output of hidden layer 1
    for i in range(hid1N):
        weightSum = 0
        for j in range(hid0N):
            weightSum += h0[j] * weight1[j][i]
        h1.append(sr.sigmoidStocastic(weightSum))

    # hidden 1 -> hidden 2
    h2 = [] # output of hidden layer 2
    for i in range(hid2N):
        weightSum = 0
        for j in range(hid1N):
            weightSum += h1[j] * weight2[j][i]
        h2.append(sr.sigmoidStocastic(weightSum))

    # print
    if count == 1:
        print('')
        print('input -> hidden layer 0, 1 and 2 -> hidden 2 output')
        print('<INPUT>')
        print(inp)
        print('<OUTPUT OF HIDDEN LAYER 0>')
        print(h0)
        print('<OUTPUT OF HIDDEN LAYER 1>')
        print(h1)
        print('<OUTPUT OF HIDDEN LAYER 2>')
        print(h2)
        print('')

    return h2

# train Neural Network
def DeepBeliefNetwork(input_, output_, hiddens, h3Nn, lr, printDetail, testdata):

    ## 1. training input layer to hidden layer 1 (unsupervised using RBM)
    ## 2. training hidden layer 1 to output layer (supervised)

    # make input, hidden and output layer
    # number of neurons
    h0Nn = hiddens[0]
    h1Nn = hiddens[1]
    h2Nn = hiddens[2]
    # weight
    h0Nw = [] # for hidden layer 0
    h1Nw = [] # for hidden layer 1
    h2Nw = [] # for hidden layer 2
    h3Nw = [] # for hidden layer 3
    oNw = [] # for output layer
    # threshold
    h3Nt = [] # for hidden layer 3
    oNt = [] # for output layer

    ## set weights for hidden layer 0, 1 and 2 using RBM
    print('')
    print('******** [ROUND 1: unsupervised learning using RBM] ********')
    
    iNn = len(input_[0]) # number of input neurons
    oNn = len(output_[0]) # number of output neurons
    
    for i in range(len(input_)):
        for j in range(iNn):
            input_[i][j] = float(input_[i][j])
            if input_[i][j] % 1.0 == 0.0: input_[i][j] = int(input_[i][j])
            
    (h0Nw_, h1Nw_, h2Nw_) = sr.StackedRBM(0.01, 'DeepBeliefNetwork.txt', hiddens, input_)

    # print input data
    print('')
    print('<input data>')
    for i in range(len(input_)):
        print(input_[i])

    # create weight matrices (transpose)
    for i in range(len(h0Nw_[0])):
        temp = []
        for j in range(len(h0Nw_)):
            temp.append(h0Nw_[j][i])
        h0Nw.append(temp)
    for i in range(len(h1Nw_[0])):
        temp = []
        for j in range(len(h1Nw_)):
            temp.append(h1Nw_[j][i])
        h1Nw.append(temp)
    for i in range(len(h2Nw_[0])):
        temp = []
        for j in range(len(h2Nw_)):
            temp.append(h2Nw_[j][i])
        h2Nw.append(temp)
    
    # set weights and thresholds (hidden layer 3 and output layer)
    for i in range(h3Nn):
        weights = [] # array of weight of hidden layer 3 neurons
        for j in range(h2Nn):
            weights.append(random.random())
        h3Nw.append(weights) # iNs weights/neuron
        h3Nt.append(random.random()) # 1 threshold/neuron
    for i in range(oNn):
        weights = [] # array of weight of output layer neurons
        for j in range(h3Nn):
            weights.append(random.random())
        oNw.append(weights) # hNs weights/neuron
        oNt.append(random.random()) # 1 threshold/neuron

    # print HIDDEN layers (unsupervised learning using RBM)
    print('')
    print('<HIDDEN 0> - unsupervised (using RBM)')
    for i in range(h0Nn):
        print('Neuron ' + str(i))
        for j in range(iNn):
            print('[ inputN ' + str(j) + ' ] weight = ' + str(round(h0Nw[i][j], 6)))
    print('')
    print('<HIDDEN 1> - unsupervised (using RBM)')
    for i in range(h1Nn):
        print('Neuron ' + str(i))
        for j in range(h0Nn):
            print('[ hidden0N ' + str(j) + ' ] weight = ' + str(round(h1Nw[i][j], 6)))
    print('')
    print('<HIDDEN 2> - unsupervised (using RBM)')
    for i in range(h2Nn):
        print('Neuron ' + str(i))
        for j in range(h1Nn):
            print('[ hidden1N ' + str(j) + ' ] weight = ' + str(round(h2Nw[i][j], 6)))

    ## supervised learning
    # repeat until convergence
    print('')
    print('******** [ROUND 2: supervised learning] ********')
    last = 0 # check if last loop
    count = 0
    while 1:
        count += 1

        # print neuron info (supervised)
        if printDetail >= -1 or count == 1 or last == 1 or count % 100 == 0:
            print('')
            print('ROUND ' + str(count))
        if printDetail >= 2 or count == 1 or last == 1:
            print('')
            print('<HIDDEN 3> - supervised')
            for i in range(h3Nn):
                print('Neuron ' + str(i) + ': thr = ' + str(round(h3Nt[i], 6)))
                for j in range(h2Nn):
                    print('[ hidden2N ' + str(j) + ' ] weight = ' + str(round(h3Nw[i][j], 6)))
            print('')
            print('<OUTPUT> - supervised')
            for i in range(oNn):
                print('Neuron ' + str(i) + ': thr = ' + str(round(oNt[i], 6)))
                for j in range(h3Nn):
                    print('[ hidden3N ' + str(j) + ' ] weight = ' + str(round(oNw[i][j], 6)))

        # save current weights
        h3Nw_ = []
        oNw_ = []
        for i in range(h3Nn):
            h3Nw__ = []
            for j in range(h2Nn):
                h3Nw__.append(round(h3Nw[i][j], 6))
            h3Nw_.append(h3Nw__)
        for i in range(oNn):
            oNw__ = []
            for j in range(h3Nn):
                oNw__.append(round(oNw[i][j], 6))
            oNw_.append(oNw__)

        # learning
        # for each training data
        error = []
        for d in range(len(input_)):
            
            # if last loop, about test data
            if last == 1: input_[d] = testdata

            # calculate output of hidden layer 2 using RBM
            hidden2Output = inputToOutput(input_[d], h0Nw_, h1Nw_, h2Nw_, count)
            
            if printDetail >= 2 or last == 1:
                print('input data    : ' + str(input_[d]))
                print('hidden2 output: ' + str(hidden2Output))
                # do not print output if last loop
                if last <= 0: print('desired output: ' + str(output_[d]))
                print('')

            # hidden layer 2 -> hidden layer 3
            # j HIDDEN2s and i HIDDEN3s
            hidden3Input = [] # hidden layer 3 input
            for i in range(h3Nn):
                hX = -h3Nt[i]
                for j in range(h2Nn):
                    hX += h3Nw[i][j]*float(hidden2Output[j])
                hidden3Input.append(hX)

            if printDetail >= 2 or last == 1:
                his = 'Hidden Layer 3 Input: [ '
                for i in range(h3Nn): his += str(round(hidden3Input[i], 6)) + '  '
                print(his + ']')

            # find hidden layer 3 output
            hidden3Output = [] # hidden layer 3 output
            for i in range(h3Nn): hidden3Output.append(rp.sigmoid(hidden3Input[i]))

            if printDetail >= 2 or last == 1:
                hos = 'Hidden Layer 3 Output: [ '
                for i in range(h3Nn): hos += str(round(hidden3Output[i], 6)) + '  '
                print(hos + ']')

            # hidden layer 3 -> output layer
            # j HIDDEN3s and i OUTPUTs
            outputInput = [] # output layer input
            for i in range(oNn):
                oX = -oNt[i]
                for j in range(h3Nn):
                    oX += oNw[i][j]*float(hidden3Output[j])
                outputInput.append(oX)

            if printDetail >= 2 or last == 1:
                ois = 'Output Layer Input: [ '
                for i in range(oNn): ois += str(round(outputInput[i], 6)) + '  '
                print(ois + ']')

            # find output layer output
            outputOutput = [] # output layer output
            for i in range(oNn): outputOutput.append(rp.sigmoid(outputInput[i]))

            if printDetail >= 2 or last == 1:
                oos = 'Output Layer Output: [ '
                for i in range(oNn): oos += str(round(outputOutput[i], 6)) + '  '
                print(oos + ']')

            # for test data, no need of backpropagation
            if last == 1: break

            # find So and Sh
            # So = (do-Oo)*Oo*(1-Oo)
            # Sh3 = Sum(So*Wh3o) * Oh3*(1-Oh3)
            So = [] # for each output layer neuron
            for i in range(oNn):
                Oo = outputOutput[i]
                So.append((float(output_[d][i])-Oo)*Oo*(1-Oo))
                
            Sh3 = [] # for each hidden layer 3 neuron
            for i in range(h3Nn):
                Sum = 0
                for j in range(oNn):
                    Sum += So[j]*oNw[j][i]
                Oh3 = hidden3Output[i]
                Sh3.append(Sum*Oh3*(1-Oh3))

            # find gradient
            H3Ograd = [] # gradient between HIDDEN 3 and OUTPUT layer
            H2H3grad = [] # gradient between HIDDEN 2 and HIDDEN 3 layer
            
            # j HIDDEN3s and i OUTPUTs
            # dE/dWh3o = (do-Oo)*Oo*(1-Oo)*Oh3 = So*Oh3
            for i in range(oNn):
                Ograd = []
                for j in range(h3Nn):
                    Ograd.append(So[i]*hidden3Output[j])
                H3Ograd.append(Ograd)
                    
            # j HIDDEN2s and i HIDDEN3s
            # dE/dWh2h3 = Sum{(do-Oo)*Oo*(1-Oo)*Wh3o} * Oh3*(1-Oh3)*Oh2
            #         = Sum(So*Wh3o) * Oh3*(1-Oh3)*Oh2 = Sh3*Oh2
            for i in range(h3Nn):
                H3grad = []
                Oh3 = hidden3Output[i]
                for j in range(h2Nn):
                    H3grad.append(Sh3[i]*float(hidden2Output[j]))
                H2H3grad.append(H3grad)

            # update Hidden3-Output weights (Wh3o += lr*So*Oh3)
            # j HIDDEN3s and i OUTPUTs
            for i in range(oNn):
                for j in range(h3Nn):
                    oNw[i][j] += lr*So[i]*hidden3Output[j]

            # update Hidden2-Hidden3 weights (Wh2h3 += lr*Sh3*Oh2)
            # j HIDDEN2s and i HIDDEN3s
            for i in range(h3Nn):
                for j in range(h2Nn):
                    h3Nw[i][j] += lr*Sh3[i]*hidden2Output[j]

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
        if errorSum / (len(input_) * len(output_)) < 0.001 or (count >= 20000 and printDetail >= -1) or count >= 1000000:
            last = 1

        # print change of weight
        if printDetail >= 1:
            print('')
            print('<weight change>')
            print('HIDDEN3 neurons:')
            for i in range(h3Nn):
                print('HIDDEN3 Neuron ' + str(i) + ':')
                for j in range(h2Nn):
                    before = str(round(h3Nw_[i][j], 6))
                    after = str(round(h3Nw[i][j], 6))
                    print('[ hidden2N ' + str(j) + ' ] weight = ' + before + '->' + after)
            print('OUTPUT neurons:')
            for i in range(oNn):
                print('OUTPUT Neuron ' + str(i) + ':')
                for j in range(h3Nn):
                    before = str(round(oNw_[i][j], 6))
                    after = str(round(oNw[i][j], 6))
                    print('[ hidden3N ' + str(j) + ' ] weight = ' + before + '->' + after)

(input_, output_, hiddens, testdata) = getData()
DeepBeliefNetwork(input_, output_, hiddens, 6, 3.25, -1, testdata)
