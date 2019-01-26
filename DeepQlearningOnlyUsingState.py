import math
import random
import sys
import DNN as DNN

# read data
def getData():
    file = open('DeepQlearningOnlyUsingState.txt', 'r')
    read = file.readlines()
    file.close

    # read info
    x = read[0].split(' ')
    actions = int(x[0]) # number of next action(direction)s
    iters = int(x[1]) # number of max iterations
    h0Nn = int(x[2]) # number of neurons in hidden 0 layer
    h1Nn = int(x[3]) # number of neurons in hidden 1 layer
    h2Nn = int(x[4]) # number of neurons in hidden 2 layer
    prtinfo = int(x[5])
    prtstate = int(x[6])
    onehot = int(x[7]) # if this value not equal to 0, use one-hot DNN input

    # read data
    data = []
    for i in range(1, len(read)):
        data.append(read[i].split('\n')[0].split(' '))
        
    return (data, actions, iters, h0Nn, h1Nn, h2Nn, prtinfo, prtstate, onehot)

# activation function
def sigmoid(value):
    return 1/(1+math.exp(-value))

# print array
def printArray(array, n, align):
    leng = len(array[0])
    result = ''
    for i in range(len(array)):
        result += '['
        for j in range(leng):
            if array[i][j] == '-':
                result += ' ------ '
            else:
                temp = str(round(float(array[i][j]), n))
                if align == 0:
                    result += ' '*(7-len(temp)) + temp + ' '
                else:
                    result += ' ' + temp + ' '*(7-len(temp))
        if i < len(array)-1: result += ']\n'
    return (result + ']')

# make list of next action (id of state) based on now state
def makeActionList(state, actions, rows, cols):
    actList = []

    if actions == 4 or actions == 8:        
        # 4 actions (directions): UP/DOWN/LEFT/RIGHT
        if state >= cols:
            actList.append(state-cols) # UP
        else: actList.append('-')
        if state < (rows-1)*cols:
            actList.append(state+cols) # DOWN
        else: actList.append('-')
        if state % cols >= 1:
            actList.append(state-1) # LEFT
        else: actList.append('-')
        if state % cols < cols-1:
            actList.append(state+1) # RIGHT
        else: actList.append('-')

        # 8 actions (directions)
        if actions == 8:
            if state >= cols and state % cols >= 1:
                actList.append(state-cols-1) # UP-LEFT
            else: actList.append('-')
            if state >= cols and state % cols < cols-1:
                actList.append(state-cols+1) # UP-RIGHT
            else: actList.append('-')
            if state < (rows-1)*cols and state % cols >= 1:
                actList.append(state+cols-1) # DOWN-LEFT
            else: actList.append('-')
            if state < (rows-1)*cols and state % cols < cols-1:
                actList.append(state+cols+1) # DOWN-RIGHT
            else: actList.append('-')

        return actList
    
    else:
        print('[ERROR] Value of actions (number of next actions) must be 4 or 8.')
        sys.exit(1)

# take BEST action
# find reward value of each next action and find best among them
def takeBest(stateT, actList, outputOutput, times):
    if times < 30:
        maxIndex = -1
        maxValue = 0
        for i in range(len(actList)):
            if actList[i] == '-' or stateT[actList[i]][1] == '-': continue # check if '-'
            if maxValue < stateT[actList[i]][1]:
                maxIndex = i
                maxValue = stateT[actList[i]][1]

        # take next action randomly, among best-reward-value actions
        bestAction = []
        for i in range(len(actList)):
            if actList[i] == '-' or stateT[actList[i]][1] == '-': continue # check if '-'
            if stateT[actList[i]][1] == maxValue:
                bestAction.append(actList[i])
        nextNum = random.randint(0, len(bestAction)-1)
        state = bestAction[nextNum]

    else: # return state that value of outputOutput is the biggest
        maxOutput = 0
        maxIndex = 0
        for i in range(len(outputOutput)):
            if actList[i] == '-' or stateT[actList[i]][1] == '-': continue # check if '-'
            
            if outputOutput[i] > maxOutput:
                maxOutput = outputOutput[i]
                maxIndex = i
        state = actList[maxIndex]

    return state

# initialize table
def initializeTable(data, prt):
    stateT = [] # state Q table
    immedT = [] # immediate reward table
    
    rows = len(data) # number of rows in the array
    cols = len(data[0]) # number of columns in the array
    for i in range(rows):
        for j in range(cols):
            if data[i][j] == '-':
                stateT.append([[i, j], data[i][j]])
                immedT.append([[i, j], data[i][j]])
            else:
                stateT.append([[i, j], round(float(data[i][j]), 1)])
                immedT.append([[i, j], round(float(data[i][j]), 1)])

    print('<Initial Data>')
    print(printArray(data, 1, 0))
    print('')

    if prt != 0:
        print('<Initial State Q Table> = <Immediate Reward Table>')
        for i in range(rows*cols):
            print(stateT[i])
        print('')

    return (rows, cols, stateT, immedT)

# do Q Learning with Deep Neural Network
def DeepQlearning(data, lr, actions, iters, h0Nn, h1Nn, h2Nn, prtinfo, prtstate, onehot):
    # make table including information about each state
    (rows, cols, stateT, immedT) = initializeTable(data, 1)

    # make DNN
    # make input, hidden and output layer
    # weight
    h0Nw = [] # for hidden layer 0
    h1Nw = [] # for hidden layer 1
    h2Nw = [] # for hidden layer 2
    oNw = [] # for output layer
    # threshold
    h0Nt = [] # for hidden layer 0
    h1Nt = [] # for hidden layer 1
    h2Nt = [] # for hidden layer 2
    oNt = [] # for output layer

    # initialize
    iNn = len(stateT[0][0]) # input: each axis(x and y) value : value is 2
    oNn = actions # output: reward for each action
    DNN.initWeightAndThreshold(iNn, h0Nn, h1Nn, h2Nn, oNn, h0Nw, h1Nw, h2Nw, oNw, h0Nt, h1Nt, h2Nt, oNt)

    # Perform Deep Q Learning
    # update weights when new input and output are provided
    times = 0
    lastModifiedTimes = 0
    while times < iters:
        times += 1 # count
        state = 0 # id (also index) of state
        doRandom = 10/(times+10) # probability of searching randomly

        print(' ******** ROUND ' + str(times) + ' ********')

        # take action for each state and update weights
        while state != len(stateT)-1:
            prevState = state # previous state

            # input : state
            input_ = []
            if onehot != 0:
                for i in range(rows*cols):
                    if i == state: input_.append(1.0)
                    else: input_.append(0.0)
            else:
                input_ = [float(int(prevState / cols)), float(int(prevState % cols))]

            # output : result of each action (number of neurons == number of actions)
            # actions->4 : UP/DOWN/LEFT/RIGHT, actions->8 : UP/DOWN/LEFT/RIGHT/UP-LEFT/.../DOWN-RIGHT
            actList = [] # list of next actions (id/index of next states)
            output_ = [] # list of output (value of sigmoid(reward) for each next action)

            # add actions            
            actList = makeActionList(state, actions, rows, cols)

            # print info
            if prtinfo != 0:
                print('**** now state: ' + str(stateT[state][0]) + ' next possible moves ****')
                for i in range(actions):
                    if actList[i] == '-': continue

                    temp = ''
                    if stateT[actList[i]][1] == '-':
                        temp = '-'
                    else:
                        temp = str(round(stateT[actList[i]][1], 6))
                    
                    print('next state   : ' + str(stateT[actList[i]][0]) + ' / reward: ' + temp)
                print('')

            # calculate reward for each action : (immediate reward) + lr * max{Q(s', a')}
            maxVal = 0 # max Q-value among next actions
            for i in range(actions):
                tempState = actList[i] # updated state if each action was taken

                # check if '-' -> add 0 and continue
                if tempState == '-' or immedT[tempState][1] == '-':
                    output_.append(0.0)
                    continue

                # get immediate reward
                value = immedT[tempState][1]
                # get max reward
                tempActList = makeActionList(tempState, actions, rows, cols)
                maxReward = 0.0
                for j in range(len(tempActList)):
                    # check if '-'
                    if tempActList[j] == '-' or stateT[tempActList[j]][1] == '-': continue
                    # update max reward
                    if stateT[tempActList[j]][1] > maxReward:
                        maxReward = stateT[tempActList[j]][1]
                value += lr * maxReward # value = (immed) + lr * (maxReward)
                
                if maxVal < value: maxVal = value
                output_.append(sigmoid(math.log(value+0.01, 2)))

            print('input vector : ' + printArray([input_], 6, 1))
            print('output vector: ' + printArray([output_], 6, 1))

            # update weights of DNN
            # forward propagation
            (hidden0Input, hidden0Output, hidden1Input, hidden1Output, hidden2Input, hidden2Output, outputInput, outputOutput) = DNN.forward([input_], 0, 0, -2, 0, iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw)
            print('DNN output   : ' + printArray([outputOutput], 6, 1))

            # print error : Sum(outputOutput[i] - output_[i])^2 - 'i' is index of each action
            error = 0
            for i in range(oNn):
                error += (outputOutput[i] - output_[i]) * (outputOutput[i] - output_[i])
            print('sum of error : ' + str(round(error, 6)))
            print('')
            
            # backpropagation
            (temp1, temp2, temp3, temp4) = DNN.Back(hidden0Output, hidden1Output, hidden2Output, outputOutput, iNn, h0Nn, h0Nw, h1Nn, h1Nw, h2Nn, h2Nw, oNn, oNw, [input_], [output_], 0, lr)
            h0Nw = temp1
            h1Nw = temp2
            h2Nw = temp3
            oNw = temp4

            # print updated weight
            if prtinfo != 0:
                print(' **** WEIGHT UPDATE ****')
                print('updated hidden 0 layer weight')
                print(printArray(h0Nw, 6, 1))
                print('')
                print('updated hidden 1 layer weight')
                print(printArray(h1Nw, 6, 1))
                print('')
                print('updated hidden 2 layer weight')
                print(printArray(h2Nw, 6, 1))
                print('')
                print('updated output layer weight')
                print(printArray(oNw, 6, 1))
                print('')

            # take action a': max(a')(Q(s', a')) or random
            doRandom = 10/(times+10) # probability of searching randomly

            # RANDOM
            if random.random() < doRandom:
                print('state update : RANDOM')
                nextNum = random.randint(0, len(actList)-1)
                while actList[nextNum] == '-' or stateT[actList[nextNum]][1] == '-':
                    nextNum = random.randint(0, len(actList)-1)
                state = actList[nextNum]
            # TAKE BEST ACTION
            else:
                print('state update : BEST')
                # find reward value of each next action and find best among them
                # if times>=30, consider neural network output
                state = takeBest(stateT, actList, outputOutput, times)

            # update state info
            stateT[prevState][1] = immedT[prevState][1] + lr * maxVal

            print('state changed: ' + str(stateT[prevState][0]) + '->' + str(stateT[state][0]))
            print('')

        # print state info
        if prtstate != 0 or times == iters:
            print('< STATE INFO >')
            for i in range(rows*cols):
                temp = ''
                temp_ = ''
                if immedT[i][1] == '-':
                    temp = ' ------ '
                    temp_ = ' ------ '
                else:
                    temp = str(round(immedT[i][1], 6))
                    temp_ = str(round(stateT[i][1], 6))
                print(str(stateT[i][0]) + ' : ' + temp_ + (' '*(9-len(temp_))) + ' ( immed: ' + temp + (' '*(9-len(temp))) + ')')

    # return the neural network
    return (iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw)

# test
def test(data, iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw, actions):
    print('')
    print('******** TEST ********')
    
    # make table including information about each state
    (rows, cols, stateT, immedT) = initializeTable(data, 0)
    state = 0

    while state != rows*cols-1: # do until reaching goal
        
        prevState = state
        input_ = [float(int(state / cols)), float(int(state % cols))]

        # add actions            
        actList = makeActionList(state, actions, rows, cols)
        
        # forward propagation
        (hidden0Input, hidden0Output, hidden1Input, hidden1Output, hidden2Input, hidden2Output, outputInput, outputOutput) = DNN.forward([input_], 0, 0, -2, 0, iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw)

        # do BEST MOVE
        state = takeBest(stateT, actList, outputOutput, 999)
        print('state changed: ' + str(stateT[prevState][0]) + '->' + str(stateT[state][0]))
        
(data, actions, iters, h0Nn, h1Nn, h2Nn, prtinfo, prtstate, onehot) = getData()
(iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw) = DeepQlearning(data, 0.8, actions, iters, h0Nn, h1Nn, h2Nn, prtinfo, prtstate, onehot)
test(data, iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw, actions)
