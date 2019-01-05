import math
import random

# read data from file
def getData():
    # get data
    file = open('Backpropagation.txt', 'r')
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
def sigmoid(value):
    return 1/(1+math.exp(-value))

# train Neural Network
def Backpropagation(input_, output_, hNn, lr, printDetail, testdata):

    iNn = len(input_[0]) # number of input neurons
    oNn = len(output_[0]) # number of output neurons

    # make input, hidden and output layer
    # weight
    hNw = [] # for hidden layers
    oNw = [] # for output layers
    # threshold
    hNt = [] # for hidden layers
    oNt = [] # for output layers
    # set weights and thresholds
    for i in range(hNn):
        weights = [] # array of weight of hidden layer neurons
        for j in range(iNn):
            weights.append(random.random())
        hNw.append(weights) # iNs weights/neuron
        hNt.append(random.random()) # 1 threshold/neuron
    for i in range(oNn):
        weights = [] # array of weight of output layer neurons
        for j in range(hNn):
            weights.append(random.random())
        oNw.append(weights) # hNs weights/neuron
        oNt.append(random.random()) # 1 threshold/neuron

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
            print('')
            print('<HIDDEN>')
            for i in range(hNn):
                print('Neuron ' + str(i) + ': thr = ' + str(round(hNt[i], 6)))
                for j in range(iNn):
                    print('[ inputN ' + str(j) + ' ] weight = ' + str(round(hNw[i][j], 6)))
            print('')
            print('<OUTPUT>')
            for i in range(oNn):
                print('Neuron ' + str(i) + ': thr = ' + str(round(oNt[i], 6)))
                for j in range(hNn):
                    print('[ hiddenN ' + str(j) + ' ] weight = ' + str(round(oNw[i][j], 6)))

        # save current weights
        hNw_ = []
        oNw_ = []
        for i in range(hNn):
            hNw__ = []
            for j in range(iNn):
                hNw__.append(round(hNw[i][j], 6))
            hNw_.append(hNw__)
        for i in range(oNn):
            oNw__ = []
            for j in range(hNn):
                oNw__.append(round(oNw[i][j], 6))
            oNw_.append(oNw__)

        # learning
        # for each training data
        error = []
        for d in range(len(input_)):
            
            # if last loop, about test data
            if last == 1: input_[d] = testdata
            
            if printDetail >= 2 or last == 1:
                print('')
                print('input data    : ' + str(input_[d]))
                # do not print output if last loop
                if last <= 0: print('desired output: ' + str(output_[d]))
            
            # input layer -> hidden layer
            # j INPUTs and i HIDDENs
            hiddenInput = [] # hidden layer input
            for i in range(hNn):
                hX = -hNt[i]
                for j in range(iNn):
                    hX += hNw[i][j]*float(input_[d][j])
                hiddenInput.append(hX)

            if printDetail >= 2 or last == 1:
                his = 'Hidden Layer Input: [ '
                for i in range(hNn): his += str(round(hiddenInput[i], 6)) + '  '
                print(his + ']')

            # find hidden layer output
            hiddenOutput = [] # hidden layer output
            for i in range(hNn): hiddenOutput.append(sigmoid(hiddenInput[i]))

            if printDetail >= 2 or last == 1:
                hos = 'Hidden Layer Output: [ '
                for i in range(hNn): hos += str(round(hiddenOutput[i], 6)) + '  '
                print(hos + ']')

            # hidden layer -> output layer
            # j HIDDENs and i OUTPUTs
            outputInput = [] # hidden layer input
            for i in range(oNn):
                oX = -oNt[i]
                for j in range(hNn):
                    oX += oNw[i][j]*float(hiddenOutput[j])
                outputInput.append(oX)

            if printDetail >= 2 or last == 1:
                ois = 'Output Layer Input: [ '
                for i in range(oNn): ois += str(round(outputInput[i], 6)) + '  '
                print(ois + ']')

            # find output layer output
            outputOutput = [] # output layer output
            for i in range(oNn): outputOutput.append(sigmoid(outputInput[i]))

            if printDetail >= 2 or last == 1:
                oos = 'Output Layer Output: [ '
                for i in range(oNn): oos += str(round(outputOutput[i], 6)) + '  '
                print(oos + ']')

            # for test data, no need of backpropagation
            if last == 1: break

            # find So and Sh
            # So = (do-Oo)*Oo*(1-Oo)
            # Sh = Sum(So*Who) * Oh*(1-Oh)
            So = [] # for each output layer neuron
            for i in range(oNn):
                Oo = outputOutput[i]
                So.append((float(output_[d][i])-Oo)*Oo*(1-Oo))
            Sh = [] # for each hidden layer neuron
            for i in range(hNn):
                Sum = 0
                for j in range(oNn):
                    Sum += So[j]*oNw[j][i]
                Oh = hiddenOutput[i]
                Sh.append(Sum*Oh*(1-Oh))

            # find gradient
            HOgrad = [] # gradient between HIDDEN and OUTPUT layer
            # j HIDDENs and i OUTPUTs
            # dE/dWho = (do-Oo)*Oo*(1-Oo)*Oh = So*Oh
            for i in range(oNn):
                Ograd = []
                Oo = outputOutput[i]
                for j in range(hNn):
                    Ograd.append(So[i]*hiddenOutput[j])
                HOgrad.append(Ograd)
                    
            IHgrad = [] # gradient between INPUT and HIDDEN layer
            # j INPUTs, i HIDDENs and k OUTPUTs
            # dE/dWih = Sum{(do-Oo)*Oo*(1-Oo)*Who} * Oh*(1-Oh)*Xi
            #         = Sum(So*Who) * Oh*(1-Oh)*Xi = Sh*Xi
            for i in range(hNn):
                Hgrad = []
                Oh = hiddenOutput[i]
                for j in range(iNn):
                    Hgrad.append(Sh[i]*float(input_[d][j]))
                IHgrad.append(Hgrad)

            # update Hidden-Output weights (Who += lr*So*Oh)
            # j HIDDENs and i OUTPUTs
            for i in range(oNn):
                for j in range(hNn):
                    oNw[i][j] += lr*So[i]*hiddenOutput[j]

            # update Input-Hidden weights (Who += lr*Sh*Xi)
            # j INPUTs and i HIDDENs
            for i in range(hNn):
                for j in range(iNn):
                    hNw[i][j] += lr*Sh[i]*float(input_[d][j])

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
            print('HIDDEN neurons:')
            for i in range(hNn):
                print('HIDDEN Neuron ' + str(i) + ':')
                for j in range(iNn):
                    before = str(round(hNw_[i][j], 6))
                    after = str(round(hNw[i][j], 6))
                    print('[ inputN ' + str(j) + ' ] weight = ' + before + '->' + after)
            print('OUTPUT neurons:')
            for i in range(oNn):
                print('OUTPUT Neuron ' + str(i) + ':')
                for j in range(hNn):
                    before = str(round(oNw_[i][j], 6))
                    after = str(round(oNw[i][j], 6))
                    print('[ hiddenN ' + str(j) + ' ] weight = ' + before + '->' + after)
        
(input_, output_, testdata) = getData()

Backpropagation(input_, output_, 6, 3.25, -2, testdata)
