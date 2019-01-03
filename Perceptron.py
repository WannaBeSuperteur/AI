import math
import random

# read data from file
def getData():
    # get data
    file = open('Perceptron.txt', 'r')
    read = file.readlines()
    file.close

    input_ = [] # input
    output_ = [] # desired output

    # read input and output
    for i in range(0, len(read)-1):
        row = read[i].split('\n')[0].split(' ')

        # append
        output_.append(row[len(row)-1])
        row.pop(len(row)-1)
        input_.append(row)

    # read the symbol of activation function
    func = read[len(read)-1].split('\n')[0]
        
    return (input_, output_, func)

# activation function
def activation(value, func):
    
    # using Hard Limiter function
    if func == 'H':
        if value >= 0: return 1
        else: return 0

    # using Linear function
    elif func == 'L':
        return value

    # using Sigmoid function
    elif func == 'S':
        return 1/(1+math.exp(-value))

# train perceptron
def perceptron(input_, output_, func):
    numOfWeight = len(input_[0]) # number of weights

    # randomize weight and threshold
    weight = []
    for i in range(numOfWeight):
        weight.append(round(random.random(), 6))
    threshold = round(0.1+random.random()*0.8, 6)

    print('Weight   : ' + str(weight))
    print('Threshold: ' + str(threshold))

    # repeat until difference between f(x-threshold) is small
    count = 0
    while 1:
        count += 1
        print('')
        print('ROUND ' + str(count))
        weightSave = []
        for i in range(len(weight)): weightSave.append(weight[i])
        
        # for each input, update weight
        weightStr = ''
        for i in range(len(weight)): weightStr += (str(round(weight[i], 6)) + ' ')
        print('WEIGHT   : ' + weightStr)
        print('THRESHOLD: ' + str(threshold))
        print('')
        sumOfDiff = 0

        # select training input data randomly
        for i in range(len(input_)*3):
            if i < len(input_): select = i
            else: select = random.randint(0, len(input_)-1)
            if i == len(input_): print('')
            
            # calculate p1w1+p2w2+...+pnwn
            x = 0
            for j in range(numOfWeight):
                x += float(input_[select][j]) * weight[j]

            # calculate the value of activation function
            y = activation(x-threshold, func) # y=f(x-threshold)
            diff = abs(y - float(output_[select]))
            print('input: ' + str(input_[select]) + ', x: ' + str(round(x, 6)) + ', x-t: ' +
                  str(round(x-threshold, 6)) + ', f(x-t): ' + str(round(y, 6)) +
                  ', desired output: ' + output_[select] + ', diff: ' + str(round(diff, 6)))

            # update weight: W(t+1) = W(t) + a*(d-y)*x
            # decrease learning rate (learning rate: 1/(9+count))
            for j in range(numOfWeight):
                weight[j] += (1/(9+count))*(float(output_[select])-y) * float(input_[select][j])

            # calculate stop condition
            sumOfDiff += diff

        print('')
        for i in range(len(weight)):
            diff = weight[i] - weightSave[i]
            print('weight ' + str(i) + ': ' + str(round(weightSave[i], 6)) +
                  ' -> ' + str(round(weight[i], 6)) + ' (diff=' + str(round(diff, 6)) + ')')

        # stop condition check
        if sumOfDiff / len(input_) < 0.001: break
        
(input_, output_, func) = getData()
print(input_)
print(output_)
print(func)
perceptron(input_, output_, func)
