import math
import random
import sys
import DNN as DNN

# read data
def getData():
    file = open('DeepQlearning.txt', 'r')
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
    onehot = int(x[7]) # if this value is not equal to 0, use one-hot DNN input

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
def makeActionList(stateCoor, actionT, actions, rows, cols):
    actList = []

    if actions == 4 or actions == 8:
        # find elements whose [0] is equal to stateCoor
        for i in range(len(actionT)):
            temp = actionT[i][0]
            if stateCoor == temp: actList.append([actionT[i][1], i])

        return actList
    
    else:
        print('[ERROR] Value of actions (number of next actions) must be 4 or 8.')
        sys.exit(1)

def isActionPossible(action, rows, cols, immedT):
    coor = action[0] # coordinates of action
    a = coor[0] # coordinate 1
    b = coor[1] # coordinate 2
    immedTIndex = a*cols+b # index of next state (immedT)
            
    if a < 0 or a >= rows or b < 0 or b >= cols or immedT[immedTIndex][1] == '-': # check if '-'
        return 0 # FALSE
    else:
        return 1 # TRUE

# take BEST action
# find reward value of each next action and find best among them
def takeBest(immedT, actList, outputOutput, times, rows, cols):
    if times < 30:
        maxIndex = -1 # index of maxValue
        maxValue = 0 # max value among actList[x][1] (reward of action)s
        
        for i in range(len(actList)):
            if isActionPossible(actList[i], rows, cols, immedT) == 0: continue # check if '-'
            
            if maxValue < actList[i][1]:
                maxIndex = i
                maxValue = actList[i][1]

        # take next action randomly, among best-reward-value actions
        bestAction = []
        for i in range(len(actList)):
            if isActionPossible(actList[i], rows, cols, immedT) == 0: continue # check if '-'
            
            if actList[i][1] == maxValue:
                bestAction.append(actList[i])
        nextNum = random.randint(0, len(bestAction)-1)
        state = bestAction[nextNum][0]

    else: # return state that value of outputOutput is the biggest
        maxOutput = 0
        maxIndex = 0
        for i in range(len(outputOutput)):
            if isActionPossible(actList[i], rows, cols, immedT) == 0: continue # check if '-'
            
            if outputOutput[i] > maxOutput:
                maxOutput = outputOutput[i]
                maxIndex = i
        state = actList[maxIndex][0]

    return state

# initialize table
def initializeTable(data, prt, actions):
    actionT = [] # state Q table
    immedT = [] # immediate reward table (collection of [state, immediate reward])
    
    rows = len(data) # number of rows in the array
    cols = len(data[0]) # number of columns in the array
    
    for i in range(rows):
        for j in range(cols):
            if data[i][j] == '-':
                immedT.append([[i, j], '-'])
            else:
                immedT.append([[i, j], round(float(data[i][j]), 1)])
                
                # UP
                if i > 0:
                    if data[i-1][j] != '-':
                        actionT.append([[i, j], [i-1, j], 0.0])
                    else: actionT.append([[i, j], [i-1, j], '-'])
                else: actionT.append([[i, j], [i-1, j], '-'])
                # DOWN
                if i < rows-1:
                    if data[i+1][j] != '-':
                        actionT.append([[i, j], [i+1, j], 0.0])
                    else: actionT.append([[i, j], [i+1, j], '-'])
                else: actionT.append([[i, j], [i+1, j], '-'])
                # LEFT
                if j > 0:
                    if data[i][j-1] != '-':
                        actionT.append([[i, j], [i, j-1], 0.0])
                    else: actionT.append([[i, j], [i, j-1], '-'])
                else: actionT.append([[i, j], [i, j-1], '-'])
                # RIGHT
                if j < cols-1:
                    if data[i][j+1] != '-':
                        actionT.append([[i, j], [i, j+1], 0.0])
                    else: actionT.append([[i, j], [i, j+1], '-'])
                else: actionT.append([[i, j], [i, j+1], '-'])

                # UP-LEFT, UP-RIGHT, DOWN-LEFT and DOWN-RIGHT
                if actions == 8:
                    # UP-LEFT
                    if i > 0 and j > 0:
                        if data[i-1][j-1] != '-':
                            actionT.append([[i, j], [i-1, j-1], 0.0])
                        else: actionT.append([[i, j], [i-1, j-1], '-'])
                    else: actionT.append([[i, j], [i-1, j-1], '-'])
                    # UP-RIGHT
                    if i > 0 and j < cols-1:
                        if data[i-1][j+1] != '-':
                            actionT.append([[i, j], [i-1, j+1], 0.0])
                        else: actionT.append([[i, j], [i-1, j+1], '-'])
                    else: actionT.append([[i, j], [i-1, j+1], '-'])
                    # DOWN-LEFT
                    if i < rows-1 and j > 0:
                        if data[i+1][j-1] != '-':
                            actionT.append([[i, j], [i+1, j-1], 0.0])
                        else: actionT.append([[i, j], [i+1, j-1], '-'])
                    else: actionT.append([[i, j], [i+1, j-1], '-'])
                    # DOWN-RIGHT
                    if i < rows-1 and j < cols-1:
                        if data[i+1][j+1] != '-':
                            actionT.append([[i, j], [i+1, j+1], 0.0])
                        else: actionT.append([[i, j], [i+1, j+1], '-'])
                    else: actionT.append([[i, j], [i+1, j+1], '-'])

    print('<Initial Data: Immediate Reward>')
    print(printArray(data, 1, 0))
    print('')

    if prt != 0:
        print('<Initial Action Reward Table>')
        for i in range(len(actionT)):
            print(actionT[i])
        print('')

    return (rows, cols, actionT, immedT)

# do Q Learning with Deep Neural Network
def DeepQlearning(data, lr, actions, iters, h0Nn, h1Nn, h2Nn, prtinfo, prtstate, onehot):
    # make table including information about each state
    (rows, cols, actionT, immedT) = initializeTable(data, 1, actions)

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
    iNn = len(actionT[0][0]) # input: each axis(x and y) value : value is 2
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
        while state != rows*cols-1:
            stateCoor = [int(state / cols), int(state % cols)] # coordinates of state
            prevState = state # previous state
            prevCoor = stateCoor # coordinates of previous state

            # input : state
            input_ = []
            if onehot != 0:
                for i in range(rows*cols):
                    if i == state: input_.append(1.0)
                    else: input_.append(0.0)
            else:
                input_ = [float(int(state / cols)), float(int(state % cols))]

            # output : result of each action (number of neurons == number of actions)
            # actions->4 : UP/DOWN/LEFT/RIGHT, actions->8 : UP/DOWN/LEFT/RIGHT/UP-LEFT/.../DOWN-RIGHT
            actList = [] # list of next actions (for indexing)
            output_ = [] # list of output (value of sigmoid(reward) for each next action)

            # add actions
            # actList: collection of [[Coor0, Coor1], actionT_index]
            actList = makeActionList(stateCoor, actionT, actions, rows, cols)

            # print info
            if prtinfo != 0:
                print('**** now state: ' + str(stateCoor) + ' next possible moves ****')
                for i in range(actions):
                    actIndex = actList[i][0][0]*rows + actList[i][0][1] # index of action (immedT)
                    
                    # index out of range
                    if actList[i][0][0] < 0 or actList[i][0][0] >= rows or actList[i][0][1] < 0 or actList[i][0][1] >= cols:
                        temp = '-'
                    # action is to go to the wall ('-')
                    elif immedT[actIndex][1] == '-':
                        temp = '-'
                    else:
                        temp = str(round(actionT[actList[i][1]][2], 6))
                    
                    print('next state   : ' + str(actionT[actList[i][1]][1]) + ' / reward: ' + temp)
                print('')

            # calculate reward for each action : (immediate reward) + lr * max{Q(s', a')}
            maxVal = 0 # max Q-value among next actions
            for i in range(actions):
                tempCoor = actList[i][0] # updated state if each action was taken
                tempIndex = tempCoor[0]*rows + tempCoor[1] # index of tempState (immedT)

                # check if '-' -> add 0 and continue
                if tempCoor[0] < 0 or tempCoor[0] >= rows or tempCoor[1] < 0 or tempCoor[1] >= cols or immedT[tempIndex][1] == '-':
                    output_.append(0.0)
                    continue

                # get immediate reward
                value = immedT[tempIndex][1]
                # get max reward
                tempActList = makeActionList(tempCoor, actionT, actions, rows, cols)
                maxReward = 0.0
                for j in range(len(tempActList)):
                    thisAction = tempActList[j] # each next action ([0]: coordinates, [1]: index of actionT)
                    
                    # check if '-'
                    if thisAction[0][0] < 0 or thisAction[0][0] >= rows: continue # out of range
                    if thisAction[0][1] < 0 or thisAction[0][1] >= cols: continue # out of range
                    if immedT[thisAction[0][0]*rows + thisAction[0][1]][1] == '-': continue # check if immediate reward is '-'
                    # update max reward
                    if actionT[thisAction[1]][2] > maxReward:
                        maxReward = actionT[thisAction[1]][2]
                value += lr * maxReward # value = (immed) + lr * (maxReward)
                
                if maxVal < value: maxVal = value
                
                actionT[actList[i][1]][2] = value
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
                nextNum = -1
                while 1:
                    nextNum = random.randint(0, len(actList)-1)
                    
                    coor = actList[nextNum][0] # coordinates of action
                    a = coor[0] # coordinate 1
                    b = coor[1] # coordinate 2
                    immedTIndex = a*cols+b # index of next state (immedT)

                    # decide nextNum (index of next action in actList)
                    if a >= 0 and a < rows and b >= 0 and b < cols and immedT[immedTIndex][1] != '-': break
                    
                newStateCoor = actList[nextNum][0] # update state
            # TAKE BEST ACTION
            else:
                print('state update : BEST')
                # find reward value of each next action and find best among them
                # if times>=30, consider neural network output
                newStateCoor = takeBest(immedT, actList, outputOutput, times, rows, cols)

            # update state
            state = newStateCoor[0]*rows + newStateCoor[1] 
            print('state changed: ' + str(prevCoor) + '->' + str(newStateCoor))
            print('')

        # print state info
        if prtstate != 0 or times == iters:
            print('< STATE INFO >')
            for i in range(rows*cols):
                temp = ''
                if immedT[i][1] == '-': temp = ' ------ '
                else: temp = str(round(immedT[i][1], 6))
                print(str(immedT[i][0]) + ' ( immed: ' + temp + (' '*(9-len(temp))) + ')')
            print('')

            print('< Action Reward Table >')
            for i in range(len(actionT)):
                temp = '-'
                if actionT[i][2] != '-': temp = str(round(actionT[i][2], 6))
                print(str(actionT[i][0]) + '->' + str(actionT[i][1]) + ' [ reward : ' + temp + ' ]')
            print('')

    # return the neural network
    return (iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw)

# test
def test(data, iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw, actions):
    print('')
    print('******** TEST ********')
    
    # make table including information about each state
    (rows, cols, actionT, immedT) = initializeTable(data, 1, actions)
    state = 0

    while state != rows*cols-1: # do until reaching goal

        stateCoor = [int(state / cols), int(state % cols)] # coordinates of state
        prevState = state
        input_ = [float(int(state / cols)), float(int(state % cols))]

        # add actions            
        actList = makeActionList(stateCoor, actionT, actions, rows, cols)
        
        # forward propagation
        (hidden0Input, hidden0Output, hidden1Input, hidden1Output, hidden2Input, hidden2Output, outputInput, outputOutput) = DNN.forward([input_], 0, 0, -2, 0, iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw)

        # do BEST MOVE
        newStateCoor = takeBest(immedT, actList, outputOutput, 999, rows, cols)
        state = newStateCoor[0]*rows + newStateCoor[1]
        print('DNN output   : ' + printArray([outputOutput], 6, 1))
        print('state changed: ' + str(stateCoor) + '->' + str(newStateCoor))
        print('')
        
(data, actions, iters, h0Nn, h1Nn, h2Nn, prtinfo, prtstate, onehot) = getData()
(iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw) = DeepQlearning(data, 0.8, actions, iters, h0Nn, h1Nn, h2Nn, prtinfo, prtstate, onehot)
test(data, iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw, actions)
