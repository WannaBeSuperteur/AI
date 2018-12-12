import math
import random

# read data from file
def getData():
    # get data
    file = open('Hopfield.txt', 'r')
    read = file.readlines()
    file.close

    data = [] # training data
    cols = 0 # number of columns (attribute and answer)
    for i in range(len(read)):
        read[i] = read[i].replace('\n', '')
        row = read[i].split(' ')
        for i in range(len(row)):
            row[i] = int(row[i])
        data.append(row)

    return data

# main: Hopfield memory
def Hopfield(data, newData):
    # make array
    array = [[0]*len(newData) for i in range(len(newData))]
    for i in range(len(newData)):
        for j in range(len(newData)):
            if i == j: continue
            for k in range(len(data)):
                array[i][j] += data[k][i] * data[k][j]

    # new data
    count = 0
    print('u(0) = ' + str(newData))
    print('')
    while 1:
        count += 1
        
        # get Array
        testArray = []
        for i in range(len(newData)): testArray.append(0)
        
        for i in range(len(newData)):
            for j in range(len(newData)):
                testArray[i] += newData[j] * array[j][i]

        # print array
        if count == 1:
            print('Array:')
            for i in range(len(newData)):
                print(array[i])
            print('')

        # update newData
        updated = 0
        for i in range(len(newData)):
            temp = newData[i]
            
            if testArray[i] >= 0: newData[i] = 1
            else: newData[i] = -1
            
            if temp != newData[i]: updated = 1

        # print u(x)
        print('u(' + str(count) + ') = ' + str(newData))

        # converge
        if updated == 0: return newData

data = getData()
newData = [1, 1, -1, 1, 1, -1, 1, 1, -1, 1]
print(Hopfield(data, newData))
