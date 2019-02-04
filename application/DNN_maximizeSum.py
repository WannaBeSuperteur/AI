import os, imp
imp.load_source('DNN', os.path.join(os.path.dirname(__file__), "../DNN.py"))
import DNN
import sys
import math
import random

# average
def avg(array):
    if len(array) == 0: return 0
    return sum(array, 0.0) / len(array)

# standard deviation
def sd(array):
    if len(array) < 2: return 1

    average = avg(array)
    sum0 = 0 # sum of (xi-avg)^2
    for i in range(len(array)):
        sum0 += (array[i] - average)*(array[i] - average)
    return math.sqrt(sum0 / len(array))

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
def takeAction(board, point, inpVal, result):
    y = point[0]
    x = point[1]

    # GO LEFT
    if inpVal == 0:
        leftCell = board[y][x-1]
        if leftCell != '.' and leftCell != '#': result += int(leftCell) # update result
        board[y][x-1] = 'S'
        point[1] -= 1 # modify the position of agent

    # GO UP
    elif inpVal == 1:
        upCell = board[y-1][x]
        if upCell != '.' and upCell != '#': result += int(upCell) # update result
        board[y-1][x] = 'S'
        point[0] -= 1 # modify the position of agent

    # GO RIGHT
    elif inpVal == 2:
        rightCell = board[y][x+1]
        if rightCell != '.' and rightCell != '#': result += int(rightCell) # update result
        board[y][x+1] = 'S'
        point[1] += 1 # modify the position of agent

    # GO DOWN
    elif inpVal == 3:
        downCell = board[y+1][x]
        if downCell != '.' and downCell != '#': result += int(downCell) # update result
        board[y+1][x] = 'S'
        point[0] += 1 # modify the position of agent

    board[y][x] = '.'
    return result

# MAXIMIZESUM: #(wall) .(blank) S(agent point) number(score item)
# play game (use input and make output)
def playGame(games, board, width, height, forwarding, lenInput, prt):

    result_ = []
    
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
        result = 0
        for i in range(lenInput):
            # make list of possible actions
            possible = [0, 0, 0, 0] # [LEFT, UP, RIGHT, DOWN] 1 if possible, 0 else
            
            # GO LEFT
            if point[1] > 0:
                leftCell = board[point[0]][point[1]-1]
                if leftCell != '#': possible[0] = 1

            # GO UP     
            if point[0] > 0:
                upCell = board[point[0]-1][point[1]]
                if upCell != '#': possible[1] = 1

            # GO RIGHT  
            if point[1] < width-1:
                rightCell = board[point[0]][point[1]+1]
                if rightCell != '#': possible[2] = 1

            # GO DOWN
            if point[0] < height-1:
                downCell = board[point[0]+1][point[1]]
                if downCell != '#': possible[3] = 1

            # take action
            if possible[inp[i]] == 1: result = takeAction(board, point, inp[i], result)
            else: # RANDOM
                a = 0
                for j in range(4):
                    if possible[j] == 1:
                        a = j
                        break
                result = takeAction(board, point, a, result)

            # print board
            if prt != 0 or i == lenInput-1:
                print('< BOARD after turn ' + str(i) + ' >')
                for j in range(len(board)):
                    print(printStr(board[j]))

        # get result
        print('result = ' + str(result))
        print('')

        # add input to DNN data
        input_.append(inp + environment)
        result_.append(result)

        # restore original board
        for i in range(height):
            for j in range(width):
                board[i][j] = tempBoard[i][j]

    # calculate average and SD of result values
    avgResult = avg(result_)
    sdResult = sd(result_)
    if sdResult == 0: sdResult = 1.0

    # add output to DNN data
    for game in range(games):
        output_.append([DNN.sigmoid((result_[game]-avgResult)/sdResult, 0)])

if __name__ == '__main__':
    # 0. read file
    f = open('DNN_maximizeSum.txt', 'r')
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
