import os, imp
imp.load_source('DNN', os.path.join(os.path.dirname(__file__), "../DNN.py"))
import DNN
import sys
import math
import random

# print vector
def printVec(array, n):
    result = '['
    for i in range(len(array)):
        result += (' ' + str(round(array[i], n)) + ' ')
    return result + ']'

# print string
def printStr(array):
    result = ''
    for i in range(len(array)):
        result += (str(array[i]) + ' ')
    return result

# array copy
def arrayCopy(array):
    result = []
    for i in range(len(array)):
        result.append(array[i])
    return result

# expand array (for example: [2, 3, 1, 0, 3, 3] -> [1, 0, 0, 1, 0, -1, -1, 0, 0, 1, 0, 1])
def expandArray(array):
    result = []
    for i in range(len(array)):
        if array[i] == 0: # 0 -> [-1, 0]
            result.append(-1)
            result.append(0)
        elif array[i] == 1: # 1 -> [0, -1]
            result.append(0)
            result.append(-1)
        elif array[i] == 2: # 2 -> [1, 0]
            result.append(1)
            result.append(0)
        elif array[i] == 3: # 3 -> [0, 1]
            result.append(0)
            result.append(1)
    return result

# take action
def takeAction(board, point, inpVal):
    y = point[0]
    x = point[1]
    
    leftCell = board[y][x-1]
    upCell = board[y-1][x]
    rightCell = board[y][x+1]
    downCell = board[y+1][x]
    
    if board[y][x] == 'V': # if agent is on the goal cell
        board[y][x] = 'X'
    else: board[y][x] = '.'
                
    if inpVal == 0: # GO LEFT
        # if box is on leftCell, modify leftleftCell
        if leftCell == 'B' or leftCell == 'O':
            leftleftCell = board[y][x-2]
            if leftleftCell == '.': board[y][x-2] = 'B'
            elif leftleftCell == 'X': board[y][x-2] = 'O'

        # modify leftCell
        if leftCell == 'X' or leftCell == 'O': board[y][x-1] = 'V' # if leftCell is goal 
        else: board[y][x-1] = 'S'
                    
        point[1] -= 1 # modify the position of agent
                    
    elif inpVal == 1: # GO UP
        # if box is on upCell, modify upupCell
        if upCell == 'B' or upCell == 'O':
            upupCell = board[y-2][x]
            if upupCell == '.': board[y-2][x] = 'B'
            elif upupCell == 'X': board[y-2][x] = 'O'

        # modify upCell
        if upCell == 'X' or upCell == 'O': board[y-1][x] = 'V' # if upCell is goal 
        else: board[y-1][x] = 'S'
                    
        point[0] -= 1 # modify the position of agent

    elif inpVal == 2: # GO RIGHT
        # if box is on rightCell, modify rightrightCell
        if rightCell == 'B' or rightCell == 'O':
            rightrightCell = board[y][x+2]
            if rightrightCell == '.': board[y][x+2] = 'B'
            elif rightrightCell == 'X': board[y][x+2] = 'O'

        # modify rightCell
        if rightCell == 'X' or rightCell == 'O': board[y][x+1] = 'V' # if rightCell is goal 
        else: board[y][x+1] = 'S'
                    
        point[1] += 1 # modify the position of agent

    elif inpVal == 3: # GO DOWN
        # if box is on downCell, modify downdownCell
        if downCell == 'B' or downCell == 'O':
            downdownCell = board[y+2][x]
            if downdownCell == '.': board[y+2][x] = 'B'
            elif downdownCell == 'X': board[y+2][x] = 'O'

        # modify downCell
        if downCell == 'X' or downCell == 'O': board[y+1][x] = 'V' # if downCell is goal 
        else: board[y+1][x] = 'S'
                    
        point[0] += 1 # modify the position of agent

# SOCOBAN: #(wall) .(blank) S(agent point) B(box) X(goal) O(box on goal) V(agent on goal)
# play game (use input and make output)
def playGame(games, board, width, height, forwarding, lenInput, prt):
    for game in range(games): # play (N=games) games

        print(' ******** GAME ' + str(game) + ' ********')
        print('')
        inp = [] # agent input

        # store original board
        tempBoard = []
        for i in range(height):
            temp = []
            for j in range(width):
                temp.append(board[i][j])
            tempBoard.append(temp)

        # find the starting point
        point = [] # position of agent
        for i in range(height):
            broken = 0
            for j in range(width):
                if board[i][j] == 'S':
                    point = [i, j]
                    broken = 1
                    break
            if broken == 1: break

        # make environment array
        environment = []

        # choose the 'best' input (output value is maximum)
        if len(forwarding) > 0:
            maxInput = [] # input that returns maximum output value
            maxAgentInput = [] # agent input(action) that returns maximum output value
            maxOutput = 0 # max output
            
            # forward propagation (for each input)
            h0Nt = forwarding[0]
            h0Nw = forwarding[1]
            h1Nt = forwarding[2]
            h1Nw = forwarding[3]
            h2Nt = forwarding[4]
            h2Nw = forwarding[5]
            oNt = forwarding[6]
            oNw = forwarding[7]

            iNn = forwarding[8]
            h0Nn = forwarding[9]
            h1Nn = forwarding[10]
            h2Nn = forwarding[11]
            h3Nn = forwarding[12]

            # list of (agent input)-(output)
            ioList = []

            # make initial agent input array
            agentInput = [] # input made by agent (algorithm)
            for i in range(lenInput):
                temp = random.randint(0, 3)
                agentInput.append(temp)

            # find input that returns maximum output using Hill-Climbing Search
            while 1:
                # make list of neighbor input (only one element is different)
                neighborInput = []
                for i in range(lenInput):
                    for j in range(3): neighborInput.append(arrayCopy(agentInput))

                    agentValue = agentInput[i] # value at index i of agent input array
                    temp = [] # {0, 1, 2, 3} - agentValue
                    if agentValue == 0: temp = [1, 2, 3]
                    elif agentValue == 1: temp = [0, 2, 3]
                    elif agentValue == 2: temp = [0, 1, 3]
                    elif agentValue == 3: temp = [0, 1, 2]

                    # change value at index i to derive a neighbor input
                    neighborInput[i*3][i] = temp[0]
                    neighborInput[i*3+1][i] = temp[1]
                    neighborInput[i*3+2][i] = temp[2]

                # get DNN output using agent input
                inp_ = expandArray(agentInput) + environment # original agent input
                (hidden0Input, hidden0Output, hidden1Input, hidden1Output, hidden2Input, hidden2Output, outputInput, outputOutput) = DNN.forward([inp_], 0, 0, -2, 0, iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw)

                maxIndex = -1 # index of element in neighborInput that returns max output
                maxValue = outputOutput[0] # maximum output value
                maxInput = arrayCopy(agentInput)

                # find neighbor input that makes DNN output larger than the agent input
                for i in range(len(neighborInput)):
                    inp__ = expandArray(neighborInput[i]) + environment # each neighbor input
                    (hidden0Input_, hidden0Output_, hidden1Input_, hidden1Output_, hidden2Input_, hidden2Output_, outputInput_, outputOutput_) = DNN.forward([inp__], 0, 0, -2, 0, iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw)

                    if outputOutput_[0] > maxValue: # check if new max record
                        maxIndex = i
                        maxValue = outputOutput_[0]
                        maxInput = arrayCopy(neighborInput[i])

                # if there is no neighbor input that returns DNN output bigger than returned DNN output of original agent input, break
                if maxIndex == -1: break

                # update agent input as neighbor input that returns maximum DNN output
                agentInput = arrayCopy(maxInput)

            # decide action input as agent input that returns max output
            inp = arrayCopy(agentInput)

        # choose input randomly
        else:
            agentInput = [] # input made by agent (algorithm)
            for i in range(lenInput):
                temp = random.randint(0, 3)
                agentInput.append(temp)

            inp = arrayCopy(agentInput)

        # take action using input
        for i in range(lenInput):
            # make list of possible actions
            possible = [0, 0, 0, 0] # [LEFT, UP, RIGHT, DOWN] 1 if possible, 0 else

            leftCell = board[point[0]][point[1]-1]
            upCell = board[point[0]-1][point[1]]
            rightCell = board[point[0]][point[1]+1]
            downCell = board[point[0]+1][point[1]]
            
            # GO LEFT
            if leftCell == '.' or leftCell == 'X': possible[0] = 1
            elif leftCell == 'B' or leftCell == 'O':
                leftleftCell = board[point[0]][point[1]-2]
                if leftleftCell == '.' or leftleftCell == 'X': possible[0] = 1

            # GO UP     
            if upCell == '.' or upCell == 'X': possible[1] = 1
            elif upCell == 'B' or upCell == 'O':
                upupCell = board[point[0]-2][point[1]]
                if upupCell == '.' or upupCell == 'X': possible[1] = 1

            # GO RIGHT  
            if rightCell == '.' or rightCell == 'X': possible[2] = 1
            elif rightCell == 'B' or rightCell == 'O':
                rightrightCell = board[point[0]][point[1]+2]
                if rightrightCell == '.' or rightrightCell == 'X': possible[2] = 1

            # GO DOWN
            if downCell == '.' or downCell == 'X': possible[3] = 1
            elif downCell == 'B' or downCell == 'O':
                downdownCell = board[point[0]+2][point[1]]
                if downdownCell == '.' or downdownCell == 'X': possible[3] = 1

            # take action
            if possible[inp[i]] == 1: takeAction(board, point, inp[i])
            else: # RANDOM
                while 1:
                    a = random.randint(0, 3)
                    if possible[a] == 1: break
                takeAction(board, point, a)

            # print board
            if prt != 0 or i == lenInput-1:
                print('< BOARD after turn ' + str(i) + ' >')
                for j in range(len(board)):
                    print(printStr(board[j]))
                print('')

        # get result
        result = 0.0
        Xcoll = [] # collection of symbol 'X'
        Bcoll = [] # collection of symbol 'B'
        for i in range(height):
            for j in range(width):
                if board[i][j] == 'O': result += 2.0
                elif board[i][j] == 'X' : Xcoll.append([i, j])
                elif board[i][j] == 'B' : Bcoll.append([i, j])
        # for each box(B), find the closest 'X' using manhattan distance
        # add (1/minDistance) to result
        for i in range(len(Bcoll)):
            minDistance = height+width
            for j in range(len(Xcoll)):
                dist = abs(Bcoll[i][0] - Xcoll[j][0]) + abs(Bcoll[i][1] - Xcoll[j][1])
                if dist < minDistance: minDistance = dist
            result += 1 / minDistance

        # add input and output to DNN data
        input_.append(expandArray(inp) + environment)
        output_.append([DNN.sigmoid(result, 0)])

        # restore original board
        for i in range(height):
            for j in range(width):
                board[i][j] = tempBoard[i][j]

# 0. read file
f = open('DNN_socoban.txt', 'r')
read = f.readlines()

size = read[0].split('\n')[0].split(' ')
width = int(size[0]) # width of board
height = int(size[1]) # height of board

gameN = read[1].split('\n')[0].split(' ')
games = int(gameN[0]) # number of games before making DNN
aftergames = int(gameN[1]) # number of games after making DNN (for each stage)
stages = int(gameN[2]) # number of stages (repeat)

neurons = read[2].split('\n')[0].split(' ') # number of neurons in each hidden layer
h0Nn = int(neurons[0]) # number of neurons in hidden layer 0
h1Nn = int(neurons[1]) # number of neurons in hidden layer 1
h2Nn = int(neurons[2]) # number of neurons in hidden layer 2

lenInput = int(read[3]) # length of DNN input

prt = int(read[4]) # print?

board = [] # game board
for i in range(height):
    board.append(read[i+5].split('\n')[0].split(' '))

print(' **** INITIAL GAME BOARD **** ')
for i in range(height):
    print(printStr(board[i]))
print('')

f.close()

input_ = [] # DNN input
output_ = [] # DNN output

# 1. repeat playing game
playGame(games, board, width, height, [], lenInput, prt)
print('')

# 2. print result
for i in range(games):
    print('input: ' + printVec(input_[i], 4) + ', output: ' + printVec(output_[i], 6))

# 3. DNN learning
(iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw) = DNN.Backpropagation(input_, output_, h0Nn, h1Nn, h2Nn, 3.25, -2, input_[0], 0, 0)
print('')

# 4. keep playing game (for n stages)
for _ in range(stages):
    print(' ******** LEARNING: STAGE ' + str(_) + ' ********')
    print('')

    # 5. play game using result of DNN
    input_ = [] # DNN input
    output_ = [] # DNN output
    playGame(aftergames, board, width, height, [h0Nt, h0Nw, h1Nt, h1Nw, h2Nt, h2Nw, oNt, oNw, iNn, h0Nn, h1Nn, h2Nn, oNn], lenInput, prt)

    for i in range(aftergames):
        print('input: ' + printVec(input_[i], 4) + ', output: ' + printVec(output_[i], 6))

    # 6. learning from this stage
    (iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw) = DNN.Backpropagation(input_, output_, h0Nn, h1Nn, h2Nn, 3.25, -2, input_[0], 0, 0)
    print('')
