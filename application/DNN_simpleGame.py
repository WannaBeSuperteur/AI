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

# play game (use input and make output)
# Goal: find the point that maximizes distance from (0, 0) point to selected point
def playGame(games, board, width, height, forwarding):
    for game in range(games): # play (N=games) games

        print(' ******** GAME ' + str(game) + ' ********')
        print('')
        inp = [] # agent input

        # initialize board
        for i in range(height):
            for j in range(width):
                board[i][j] = '-'

        # make environment array
        #
        #
        #
        #
        
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

            # find input that returns maximum output
            for i in range(height):
                for j in range(width):
                    agentInput = [i, j] # input made by agent (algorithm)
                    environment = [] # array that represents environment

                    # DNN input
                    inp_ = agentInput + environment
                    
                    (hidden0Input, hidden0Output, hidden1Input, hidden1Output, hidden2Input, hidden2Output, outputInput, outputOutput) = DNN.forward([inp_], 0, 0, -2, 0, iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw)
                    if outputOutput[0] > maxOutput: # check if new max record
                        maxInput = inp_
                        maxAgentInput = agentInput
                        maxOutput = outputOutput[0]

                    ioList.append([agentInput, outputOutput[0]])

            # maxIoList: max output value among ioList
            maxIoList = []
            for i in range(len(ioList)):
                if abs(ioList[i][1] - maxOutput) < 0.00000001: maxIoList.append(ioList[i])

            # decide action input as agent input that returns max output
            inp = maxIoList[random.randint(0, len(maxIoList)-1)][0]

        # choose input randomly
        else:
            a = random.randint(0, height-1)
            b = random.randint(0, width-1)
            inp = [a, b] # decide action input randomly

        # take action using input, and get result
        board[inp[0]][inp[1]] = '#'
        for i in range(height): print(board[i])

        # playing game
        #
        #
        #
        #
        
        result = math.sqrt(inp[0]*inp[0] + inp[1]*inp[1])

        # add input and output to DNN data
        input_.append(inp + environment)
        output_.append([DNN.sigmoid(result, 0)])

# 0. read file
f = open('DNN_simpleGame.txt', 'r')
read = f.readlines()

size = read[0].split('\n')[0].split(' ')
width = int(size[0]) # width of board
height = int(size[1]) # height of board

gameN = read[1].split('\n')[0].split(' ')
games = int(gameN[0]) # number of games before making DNN
aftergames = int(gameN[1]) # number of games after making DNN (for each stage)
stages = int(gameN[2]) # number of stages (repeat)

neurons = read[2].split('\n')[0].split(' ') # number of neurons in each hidden layer

board = [] # game board
for i in range(height):
    board.append(read[i+3].split('\n')[0].split(' '))

f.close()

input_ = [] # DNN input
output_ = [] # DNN output

# 1. repeat playing game
playGame(games, board, width, height, [])
print('')

# 2. print result
for i in range(games):
    print('input: ' + printVec(input_[i], 4) + ', output: ' + printVec(output_[i], 6))

# 3. DNN learning
h0Nn = int(neurons[0])
h1Nn = int(neurons[1])
h2Nn = int(neurons[2])
(iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw) = DNN.Backpropagation(input_, output_, h0Nn, h1Nn, h2Nn, 3.25, -2, input_[0], 0, 0)
print('')

# 4. keep playing game (for n stages)
for _ in range(stages):
    print(' ******** LEARNING: STAGE ' + str(_) + ' ********')
    print('')

    # 5. play game using result of DNN
    input_ = [] # DNN input
    output_ = [] # DNN output
    playGame(aftergames, board, width, height, [h0Nt, h0Nw, h1Nt, h1Nw, h2Nt, h2Nw, oNt, oNw, iNn, h0Nn, h1Nn, h2Nn, oNn])

    for i in range(aftergames):
        print('input: ' + printVec(input_[i], 4) + ', output: ' + printVec(output_[i], 6))

    # 6. learning from this stage
    (iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw) = DNN.Backpropagation(input_, output_, h0Nn, h1Nn, h2Nn, 3.25, -2, input_[0], 0, 0)
    print('')
