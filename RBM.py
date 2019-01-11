import math
import random
import RBMPackage as rp

# read data from file
def getData():
    # get data
    file = open('RBM.txt', 'r')
    read = file.readlines()
    file.close

    data = [] # training data
    cols = 0 # number of columns (attribute and answer)

    # read number of hidden units
    hiddens = int(read[0])

    # read data
    for i in range(1, len(read)-1):
        read[i] = read[i].replace('\n', '')
        row = read[i].split(' ')
        for i in range(len(row)):
            row[i] = int(row[i])
        data.append(row)

    # read input data
    inputD = read[len(read)-1].split(' ')

    return (hiddens, data, inputD)

(hiddens, data, inputD) = getData()
weight = rp.RBM(data, hiddens, 0.01)

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
