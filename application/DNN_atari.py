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
def printVec(array, n, maxPrint):
    result = '['
    for i in range(min(len(array), maxPrint)):
        result += (' ' + str(round(array[i], n)) + ' ')
    if maxPrint < len(array): result += '... '
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

# take action
def takeAction(board, moveBar, result, barStart, barEnd, ballIndex, ballVector):

    # get width and height of board
    height = len(board)
    width = len(board[0])

    # update the board and if needed, modify ballVector
    newBallY = ballIndex[0] + ballVector[0] # new x coordinate of ball
    newBallX = ballIndex[1] + ballVector[1] # new y coordinate of ball

    # GAME OVER
    if ballIndex[0] >= len(board)-1: return (result, 'F', 'F') # GAME OVER
        
    # bar
    if newBallY == height-1 and ballIndex[1] >= barStart and ballIndex[1] < barEnd:
        ballVector[0] *= (-1) # reverse Y vector of ball (down -> up)

        # change X vector in accordance with location of ball related to bar
        ballLocOfBar = (ballIndex[1] + 0.5 - barStart)/(barEnd - barStart)
        if ballLocOfBar < 0.25: ballVector[1] = -1
        elif ballLocOfBar < 0.75: ballVector[1] = 0
        else: ballVector[1] = 1

        # if wall, modify X vector of ball
        if ballIndex[1] == 0: ballVector[1] = 1
        elif ballIndex[1] == width-1: ballVector[1] = -1
        
    # number
    elif (newBallX >= 0 and newBallX < width and newBallY >= 0 and newBallY < height) and (board[newBallY][newBallX] != '.' and board[newBallY][newBallX] != '-'):
        # change the direction of ball
        # DIRECT UP
        if ballVector[1] == 0:
            if board[newBallY][newBallX] != '.':
                ballVector[0] *= (-1)
        # DIAGONAL
        else:
            udlrBlock = 0 # check if there are blocks at UP/DOWN/LEFT/RIGHT cell
            if board[newBallY][ballIndex[1]] != '.':
                udlrBlock = 1
                ballVector[0] *= (-1)
            if board[ballIndex[0]][newBallX] != '.':
                udlrBlock = 1
                ballVector[1] *= (-1)

            if udlrBlock == 0 and board[newBallY][newBallX] != '.':
                ballVector[0] *= (-1)
                ballVector[1] *= (-1)

        # update result and board
        result += int(board[newBallY][newBallX])
        board[newBallY][newBallX] = '.'

    # wall (2)
    if ballVector[1] != 0: # leftmost or rightmost
        if ballIndex[1] == 0: ballVector[1] = 1 # leftmost
        elif ballIndex[1] == width-1: ballVector[1] = -1 # rightmost

        # if there is something at the changed cell, reverse Y vector
        newCell = board[ballIndex[0]+ballVector[0]][ballIndex[1]+ballVector[1]]
        if newCell != '.' and newCell != '-': ballVector[0] *= (-1)
        
    if ballIndex[0] == 0: # top of board
        ballVector[0] = 1

        # if there is something at the changed cell, reverse X vector
        newCell = board[ballIndex[0]+ballVector[0]][ballIndex[1]+ballVector[1]]
        if newCell != '.' and newCell != '-':
            # update result
            board[ballIndex[0]+ballVector[0]][ballIndex[1]+ballVector[1]] = '.'
            result += int(newCell)

            ballVector[1] *= (-1) # reverse X vector

    # move bar
    if moveBar == -1: # GO LEFT
        # move bar
        board[height-1][barStart-1] = '-'
        board[height-1][barEnd-1] = '.'
    elif moveBar == 1: # GO RIGHT
        # move bar
        board[height-1][barStart] = '.'
        board[height-1][barEnd] = '-'

    # update location of ball
    board[ballIndex[0]][ballIndex[1]] = '.'
    board[ballIndex[0]+ballVector[0]][ballIndex[1]+ballVector[1]] = 'O'
    ballIndex[0] += ballVector[0]
    ballIndex[1] += ballVector[1]

    # return
    return(result, ballIndex, ballVector)

# ATARI: O(ball), -(bar), number(block -> number indicates score)
# play game (use input and make output)
def playGame(games, board, width, height, forwarding, lenInput, prt):

    result_ = []
    
    for game in range(games): # play (N=games) games

        print(' ******** GAME ' + str(game) + ' ********')
        print('')
        inp = [] # agent input

        # find start and end index of bar
        barStart = -1
        barEnd = -1
        for i in range(width):
            if board[height-1][i] == '-':
                barStart = i
                break
        for i in range(barStart, width):
            if board[height-1][i] == '.':
                barEnd = i
                break

        # find ball
        ballIndex = [-1, -1]
        for i in range(height):
            broken = 0
            for j in range(width):
                if board[i][j] == 'O':
                    ballIndex = [i, j]
                    broken = 1
                    break
            if broken == 1: break
        ballVector = [-1, -1] # direction(vector) of ball moving

        # store original board
        tempBoard = []
        for i in range(height):
            temp = []
            for j in range(width):
                temp.append(board[i][j])
            tempBoard.append(temp)

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
                x = random.randint(0, 1)
                if x == 0: agentInput.append(-1)
                else: agentInput.append(1)

            # find input that returns maximum output using Hill-Climbing Search
            while 1:
                # make list of neighbor input (only one element is different)
                neighborInput = []
                for i in range(lenInput):
                    neighborInput.append(arrayCopy(agentInput))
                    
                    # change value at index i to derive a neighbor input
                    neighborInput[i][i] *= -1

                # get DNN output using agent input
                inp_ = agentInput + environment # original agent input
                (hidden0Input, hidden0Output, hidden1Input, hidden1Output, hidden2Input, hidden2Output, outputInput, outputOutput) = DNN.forward([inp_], 0, 0, -2, 0, iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw)

                maxIndex = -1 # index of element in neighborInput that returns max output
                maxValue = outputOutput[0] # maximum output value
                maxInput = arrayCopy(agentInput)

                # find neighbor input that makes DNN output larger than the agent input
                for i in range(len(neighborInput)):
                    inp__ = neighborInput[i] + environment # each neighbor input
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
                x = random.randint(0, 1)
                if x == 0: agentInput.append(-1)
                else: agentInput.append(1)

            inp = arrayCopy(agentInput)

        # take action using input
        result = 0
        for i in range(lenInput):

            # take action
            x = random.randint(0, 1)
            moveBar = -2
            # GO RIGHT
            if inp[i] == 1:
                if barEnd < width: moveBar = 1
                else: # RANDOM
                    if x == 0: moveBar = 0
                    else: moveBar = -1
            # GO LEFT
            elif inp[i] == -1:
                if barStart > 0: moveBar = -1
                else: # RANDOM
                    if x == 0: moveBar = 0
                    else: moveBar = 1
            # do action
            (result, ballIndex, ballVector) = takeAction(board, moveBar, result, barStart, barEnd, ballIndex, ballVector)
            if ballIndex == 'F':
                print('DNN_atari missed the ball and the final result score is ' + str(result) + '.')
                break
            # update barStart and barEnd         
            barStart += moveBar
            barEnd += moveBar

            # print board
            if prt != 0 or i == lenInput-1:
                print('< BOARD after turn ' + str(i) + ' > - result: ' + str(result))
                for j in range(len(board)):
                    print(printStr(board[j]))
                print('')

        # get result
        # DO NOTHING

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
    f = open('DNN_atari.txt', 'r')
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
    if prt > 0 or games * lenInput <= 2000:
        for i in range(games):
            print('input: ' + printVec(input_[i], 4, 20) + ', output: ' + printVec(output_[i], 6, 2147483647))

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
            print('input: ' + printVec(input_[i], 4, 20) + ', output: ' + printVec(output_[i], 6, 2147483647))

        # 6. learning from this stage
        (iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw) = DNN.Backpropagation(input_, output_, h0Nn, h1Nn, h2Nn, 3.25, -2, input_[0], 0, 0)
        print('')
