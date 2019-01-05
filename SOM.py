import math
import random

# read data from file
def getData():
    # get data
    file = open('SOM.txt', 'r')
    read = file.readlines()
    file.close

    # width and height
    wh = read[0].split(' ')
    width = int(wh[0]) # width of output neurons
    height = int(wh[1]) # height of output neurons
    maxcount = int(wh[2]) # maximum repeat count

    # input data
    input_ = []
    for i in range(1, len(read)):
        read[i] = read[i].split('\n')[0]
        input_.append(read[i].split(' ')) # input data

    return (width, height, input_, maxcount)

# print weight
def printWeight(weight):
    height = len(weight)
    width = len(weight[0])
    elements = len(weight[0][0])
    
    for i in range(height):
        rowstr = ''
        for j in range(width):
            rowstr += '['
            for k in range(elements):
                elestr = str(round(weight[i][j][k], 3))
                elestr += '0'*(5-len(elestr))
                rowstr += elestr
                if k < elements-1: rowstr += ' '
            rowstr += '] '
        print('y=' + str(i) + ' -> ' + rowstr)

# main: SOM
def SOM(width, height, input_, maxcount, lr):
    inputs = len(input_) # number of inputs
    elements = len(input_[0]) # number of elements in an input vector

    # initialize weight
    weight = []
    for i in range(height):
        temp0 = []
        for j in range(width):
            temp1 = []
            for k in range(elements): # each element <-> each weight
                temp1.append(round(random.random(), 6))
            temp0.append(temp1)
        weight.append(temp0)

    # print
    print('INITIALIZED WEIGHT:')
    printWeight(weight)

    # learning
    count = 0
    while 1:
        count += 1
        print('')
        print('ROUND ' + str(count))
        
        for i in range(inputs):
            # new input pattern
            inp = input_[i]

            # calculate distance and select winner neuron
            wnX = -1 # x axis of winner neuron
            wnY = -1 # y axis of winner neuron
            minDist = 9999 # distance between input vector and output neuron
            for j in range(height):
                for k in range(width):
                    # dist = Sum((inp[e]-weight[n][e])^2)
                    dist = 0
                    for e in range(elements):
                        inp[e] = float(inp[e])
                        dist += (inp[e]-weight[j][k][e])*(inp[e]-weight[j][k][e])
                    dist = math.sqrt(dist)
                    # update winner neuron
                    if dist < minDist:
                        minDist = dist
                        wnX = k
                        wnY = j

            # set radius
            radius = 0
            if count < maxcount * 0.1: radius = 6
            elif count < maxcount * 0.2: radius = 5
            elif count < maxcount * 0.3: radius = 4
            elif count < maxcount * 0.4: radius = 3
            elif count < maxcount * 0.6: radius = 2
            elif count < maxcount * 0.8: radius = 1

            # update weight
            for j in range(wnY-radius, wnY+radius+1):
                if j < 0 or j >= height: continue
                for k in range(wnX-radius, wnX+radius+1):
                    if k < 0 or k >= width: continue
                    
                    # weight += lr*(input-weight)
                    for e in range(elements):
                        weight[j][k][e] += lr*(inp[e]-weight[j][k][e])

        # print result
        print('UPDATED WEIGHT:')
        printWeight(weight)

        if count >= maxcount: break

(width, height, input_, maxcount) = getData()

SOM(width, height, input_, maxcount, 0.5)
