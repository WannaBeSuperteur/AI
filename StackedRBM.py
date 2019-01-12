import math
import random
import RBMPackage as rp

# read data from file
def getData(fn):
    # get data
    file = open(fn, 'r')
    read = file.readlines()
    file.close

    data = [] # training data
    cols = 0 # number of columns (attribute and answer)

    # read number of hidden units
    fl = read[0].split(' ')
    hid0 = int(fl[0]) # for hidden layer 0
    hid1 = int(fl[1]) # for hidden layer 1
    hid2 = int(fl[2]) # for hidden layer 2

    # read data
    for i in range(1, len(read)-1):
        read[i] = read[i].replace('\n', '')
        row = read[i].split('/')[0].split(' ')
        for i in range(len(row)):
            row[i] = int(row[i])
        data.append(row)

    # read input data
    inputD = read[len(read)-1].split(' ')

    return (hid0, hid1, hid2, data, inputD)

def sigmoidStocastic(value):
    sigm = rp.sigmoid(value)
    if sigm >= 0.5: return 1
    else: return 0

def StackedRBM(lr, fn):
    # read data and make 'hiddens' array
    (hid0, hid1, hid2, data, inputD) = getData(fn)
    hiddens = [hid0, hid1, hid2]

    # calculate final weight array (weight of hidden layer 2) using RBM
    weight0 = [] # weight for hidden layer 0
    weight1 = [] # weight for hidden layer 1
    weight2 = [] # weight for hidden layer 2
    for i in range(3):
        print('')
        print('******** [WEIGHT OF HIDDEN LAYER ' + str(i) + '] ********')
        print('')

        # set weight
        # if hidden layer 1 or 2, using output of previous layer
        if i > 0:
            newData = [] # hidden layer 0 output (using sigmoid)
            
            # for each training data, calculate output
            for j in range(len(data)):
                temp = [] # hidden layer 0 output for each data
                for k in range(len(weight0[0])):
                    weightedSum = 0
                    for l in range(len(weight0)):
                        weightedSum += data[j][l] * weight0[l][k]
                    temp.append(sigmoidStocastic(weightedSum))
                newData.append(temp)

            # if hidden layer 1, use newData immediately
            if i == 1:
                print('<Hidden Layer 0 Output>')
                for j in range(len(newData)):
                    print(newData[j])
                weight1 = rp.RBM(newData, hiddens[1], lr, 0)

            # if hidden layer 2, calculate hidden layer 1 output
            else:
                newData_ = [] # hidden layer 1 output (using sigmoid)
                
                # for each training data, calculate output
                for j in range(len(newData)):
                    temp = [] # hidden layer 1 output for each data
                    for k in range(len(weight1[0])):
                        weightedSum = 0
                        for l in range(len(weight1)):
                            weightedSum += newData[j][l] * weight1[l][k]
                        temp.append(sigmoidStocastic(weightedSum))
                    newData_.append(temp)

                print('<Hidden Layer 1 Output>')
                for j in range(len(newData_)):
                    print(newData_[j])
                weight2 = rp.RBM(newData_, hiddens[2], lr, 0)

        # else, using input
        else:
            print('<Input Layer>')
            for j in range(len(data)):
                print(data[j])
            weight0 = rp.RBM(data, hiddens[0], lr, 0)
        print('')

        # print weights
        print('weight ' + str(i) + ':')
        if i == 0:
            for j in range(len(weight0)):
                print(weight0[j])
        elif i == 1:
            for j in range(len(weight1)):
                print(weight1[j])
        elif i == 2:
            for j in range(len(weight2)):
                print(weight2[j])
        print('')

    return (inputD, weight0, weight1, weight2)

# execute stacked RBM
if(__name__ == '__main__'):
    (inputD, weight0, weight1, weight2) = StackedRBM(0.01, 'StackedRBM.txt')
    print('')
    print('weight 0')
    for i in range(len(weight0)):
        print(weight0[i])
    print('')
    print('weight 1')
    for i in range(len(weight1)):
        print(weight1[i])
    print('')
    print('weight 2')
    for i in range(len(weight2)):
        print(weight2[i])
