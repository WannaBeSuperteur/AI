import os, imp
imp.load_source('GamePlaying', os.path.join(os.path.dirname(__file__), "../GamePlaying.py"))
imp.load_source('DNN', os.path.join(os.path.dirname(__file__), "../DNN.py"))
import GamePlaying
import DNN
import DNN_othello
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

# return R times of array (for example R=5: [2, 3, 4] -> [10, 15, 20])
def multiArray(array, value):
    result = []
    for i in range(len(array)):
        result.append(array[i] * value)
    return result

# get score of each player (can be not equal to count)
def getSco(board_, OorX):
    result = 0

    for i in range(len(board_)):
        for j in range(len(board_[0])):
            if board_[i][j] == OorX: result += DNNinput[i][j]
        
    return result

# check if victory (end of game)
def victory(board, turn, scoreFunc):
    size = len(board)
    
    # count blanks
    blanks = 0
    for i in range(size):
        for j in range(size):
            if board[i][j] == '-': blanks += 1

    # victory check
    if blanks == 0 or DNN_othello.getCount(board, 'O') == 0 or DNN_othello.getCount(board, 'X') == 0: return 1
    else: return 0

# othello
# play game (use input and make output)
def playGame(games, board, bSize, forwarding, lenInput, prt, DNNround):
    half = int(bSize/2)
    
    for game in range(games): # play (N=games) games

        print(' ******** GAME ' + str(game) + ' ********')
        inp = [] # agent input

        # initialize board
        board = [['-']*bSize for i in range(bSize)] # game board
        board[int(bSize/2)-1][int(bSize/2)-1] = 'O'
        board[int(bSize/2)-1][int(bSize/2)] = 'X'
        board[int(bSize/2)][int(bSize/2)-1] = 'X'
        board[int(bSize/2)][int(bSize/2)] = 'O'

        # store original board
        tempBoard = []
        for i in range(bSize):
            temp = []
            for j in range(bSize):
                temp.append(board[i][j])
            tempBoard.append(temp)

        # choose the 'best' input (output value is maximum)
        if DNNround > 0:
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
                temp = random.randint(0, 200)-100 # range: -100 ~ 100
                agentInput.append(temp)

            # find input that returns maximum output using Hill-Climbing Search
            while 1:
                # make list of neighbor input (only one element is different)
                neighborInput = []
                for i in range(lenInput):
                    # make neighbor inputs
                    if agentInput[i] >= -99: # -1 neighbor
                        neiInput = arrayCopy(agentInput)
                        neiInput[i] -= 1
                        neighborInput.append(arrayCopy(neiInput))
                    if agentInput[i] <= 99: # +1 neighbor
                        neiInput = arrayCopy(agentInput)
                        neiInput[i] += 1
                        neighborInput.append(arrayCopy(neiInput))

                # get DNN output of agent input
                inp_ = multiArray(agentInput, 0.01) # original agent input
                (hidden0Input, hidden0Output, hidden1Input, hidden1Output, hidden2Input, hidden2Output, outputInput, outputOutput) = DNN.forward([inp_], 0, 0, -2, 0, iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw)

                maxIndex = -1 # index of element in neighborInput that returns max output
                maxValue = outputOutput[0] # maximum output value
                maxInput = arrayCopy(agentInput)

                # find neighbor input that makes DNN output larger than the agent input
                for i in range(len(neighborInput)):
                    inp__ = multiArray(neighborInput[i], 0.01) # each neighbor input
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

            # add noise to input (decreasing noise)
            for i in range(len(inp)):
                inp[i] = int(inp[i] * DNNround/(random.random()+DNNround))

        # choose input randomly
        else:
            agentInput = [] # input made by agent (algorithm)
            for i in range(lenInput):
                temp = random.randint(0, 200)-100 # range: -100 ~ 100
                agentInput.append(temp)

            inp = arrayCopy(agentInput)

        # modify DNN input array
        if len(forwarding) >= 0:
            # 1. LEFT-UP about 1/8
            tempCount = 0
            for i in range(half):
                for j in range(i+1):
                    if i == j and i == half-1: continue # except for center cell
                    DNNinput[i][j] = inp[tempCount]
                    tempCount += 1

            # 2. LEFT-UP about 1/4
            for i in range(half):
                for j in range(i+1, half):
                    DNNinput[i][j] = DNNinput[j][i]

            # 3. EVERY CELL
            for i in range(half, bSize):
                for j in range(half):
                    DNNinput[i][j] = DNNinput[bSize-i-1][j]
            for i in range(bSize):
                for j in range(half, bSize):
                    DNNinput[i][j] = DNNinput[i][bSize-j-1]

        # print DNN input
        print(' **** DNN input array ****')
        for i in range(bSize):
            print(printVec(DNNinput[i], 4))
        print('')

        # take action using input
        # from DNN_othello.py in the same directory
        turns = 0
        drawturns = 0
        while(1):

            # make temp board (store previous board)
            for i in range(bSize):
                for j in range(bSize):
                    tempBoard[i][j] = board[i][j]
            
            # turn check
            if turns % 2 == 0: # turn of O: using Game Tree
                
                tree = GamePlaying.spanTree(board, 'O', bSize, getSco, DNN_othello.getVal, DNN_othello.checkCondi, DNN_othello.modifyBoard, victory)
                if len(tree) == 1: # no next move
                    print('no next move for player O')
                    drawturns += 1
                    turns += 1
                    if drawturns == 1: continue
                else: drawturns = 0
                board = GamePlaying.findAnswer(tree, 'O')
            else: # turn of X: random
                # make list of possible moves
                possible = [] # list of possible moves
                for i in range(bSize):
                    for j in range(bSize):
                        if DNN_othello.checkCondi(board, i, j, 'X') == 1:
                            possible.append([i, j])

                # MOVE!                        
                if len(possible) == 0: # no next move
                    print('no next move for player X')
                    drawturns += 1
                    turns += 1
                    if drawturns == 1: continue
                else: # there are some possible next moves
                    drawturns = 0
                    select = random.randint(0, len(possible)-1) # SELECT RANDOM
                    coor0 = possible[select][0]
                    coor1 = possible[select][1]
                    DNN_othello.modifyBoard(board, coor0, coor1, 'X')

            # print
            print('')
            print('board at turn ' + str(turns+1))
            print('O ' + str(DNN_othello.getCount(board, 'O')) + ' : X ' + str(DNN_othello.getCount(board, 'X')))
            for i in range(bSize):
                print(board[i])

            # count blanks
            blanks = 0
            for i in range(bSize):
                for j in range(bSize):
                    if board[i][j] == '-': blanks += 1

            # victory check
            if blanks == 0 or DNN_othello.getCount(board, 'O') == 0 or DNN_othello.getCount(board, 'X') == 0 or drawturns >= 2:
                print('FINAL RESULT: O ' + str(DNN_othello.getCount(board, 'O')) + ' : X ' + str(DNN_othello.getCount(board, 'X')))
                print('')
                
                check = DNN_othello.getCount(board, 'O') - DNN_othello.getCount(board, 'X')
                if check >= 1:
                    print('O victory')
                    break
                elif check <= -1:
                    print('X victory')
                    break
                else:
                    print('draw')
                    break

            turns += 1

        # get result
        Ocount = DNN_othello.getCount(board, 'O')
        Xcount = DNN_othello.getCount(board, 'X')
        result = (Ocount - Xcount)/(Ocount + Xcount)

        # add input and output to DNN data
        input_.append(multiArray(inp, 0.01))
        output_.append([DNN.sigmoid(result, 0)])

        # restore original board
        for i in range(bSize):
            for j in range(bSize):
                board[i][j] = tempBoard[i][j]

# 0. read file
f = open('DNN_othello2.txt', 'r')
read = f.readlines()

bSize = int(read[0]) # board size (width and height)
if bSize % 2 != 0:
    print('Board size must be an even number.')
    sys.exit(1)

gameN = read[1].split('\n')[0].split(' ')
games = int(gameN[0]) # number of games before making DNN
aftergames = int(gameN[1]) # number of games after making DNN (for each stage)
stages = int(gameN[2]) # number of stages (repeat)

neurons = read[2].split('\n')[0].split(' ') # number of neurons in each hidden layer
h0Nn = int(neurons[0]) # number of neurons in hidden layer 0
h1Nn = int(neurons[1]) # number of neurons in hidden layer 1
h2Nn = int(neurons[2]) # number of neurons in hidden layer 2

half = int(bSize/2)
lenInput = int(half*(half+1)/2 - 1) # length of DNN input

board = [['-']*bSize for i in range(bSize)] # game board
board[int(bSize/2)-1][int(bSize/2)-1] = 'O'
board[int(bSize/2)-1][int(bSize/2)] = 'X'
board[int(bSize/2)][int(bSize/2)-1] = 'X'
board[int(bSize/2)][int(bSize/2)] = 'O'

prt = int(read[3]) # print?

print(' **** INITIAL GAME BOARD **** ')
for i in range(bSize):
    print(printStr(board[i]))
print('')

f.close()

input_ = [] # DNN input
output_ = [] # DNN output

# 1. make DNN input array
DNNinput = [[0]*bSize for i in range(bSize)] # size: bSize x bSize

# 2. repeat playing game
playGame(games, board, bSize, [], lenInput, prt, -1)
print('')

# 3. print result
for i in range(games):
    print('input: ' + printVec(input_[i], 4) + ', output: ' + printVec(output_[i], 6))

# 4. DNN learning
(iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw) = DNN.Backpropagation(input_, output_, h0Nn, h1Nn, h2Nn, 3.25, -2, input_[0], 0, 0)
print('')

# 5. keep playing game (for n stages)
for DNNround in range(stages):
    print(' ******** LEARNING: STAGE ' + str(DNNround) + ' ********')
    print('')

    # 6. play game using result of DNN
    input_ = [] # DNN input
    output_ = [] # DNN output
    playGame(aftergames, board, bSize, [h0Nt, h0Nw, h1Nt, h1Nw, h2Nt, h2Nw, oNt, oNw, iNn, h0Nn, h1Nn, h2Nn, oNn], lenInput, prt, DNNround+1)

    for i in range(aftergames):
        print('input: ' + printVec(input_[i], 4) + ', output: ' + printVec(output_[i], 6))

    # 7. learning from this stage
    (iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw) = DNN.Backpropagation(input_, output_, h0Nn, h1Nn, h2Nn, 3.25, -2, input_[0], 0, 0)
    print('')
