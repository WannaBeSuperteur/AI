import os, imp
imp.load_source('GamePlaying', os.path.join(os.path.dirname(__file__), "../GamePlaying.py"))
imp.load_source('DNN', os.path.join(os.path.dirname(__file__), "../DNN.py"))
import GamePlaying
import DNN
import sys

# othello
MAXDEP = 4

# print vector
def printVec(array, n):
    result = '['
    for i in range(len(array)):
        result += (' ' + str(round(array[i], n)) + ' ')
    return result + ']'

# get count of each player
def getCount(board_, OorX):
    result = 0

    for i in range(len(board_)):
        for j in range(len(board_[0])):
            if board_[i][j] == OorX: result += 1
        
    return result

# get score of each player (can be not equal to count)
def getSco(board_, OorX):
    result = 0

    for i in range(len(board_)):
        for j in range(len(board_[0])):
            if board_[i][j] == OorX: result += 1
        
    return result

# get score of each player - using DNN (can be not equal to count)
def getScoUsingDNN(board_, OorX):
    result = 0

    for i in range(len(board_)):
        for j in range(len(board_[0])):
            if board_[i][j] == OorX: result += DNNarray[i][j]
        
    return result

# get score of player 1
def getVal(board_, scoreFunc):
    score_of_p1 = 0 # score of player 1 (n(O)-n(X))
    # O : Line(3) + 10*Line(2) + 100*Line(1)
    p1_score = scoreFunc(board_, 'O')
    # X : Line(3) + 10*Line(2) + 100*Line(1)
    p2_score = scoreFunc(board_, 'X')
    return p1_score-p2_score

# check condition
def checkCondi(board, coor0, coor1, turn):   
    a = coor0
    b = coor1
    aChange = [-1, -1, -1, 0, 1, 1, 1, 0] # LEFT-UP, UP, RIGHT-UP, ..., LEFT
    bChange = [-1, 0, 1, 1, 1, 0, -1, -1] # LEFT-UP, UP, RIGHT-UP, ..., LEFT

    if board[a][b] == 'O' or board[a][b] == 'X': return 0
    
    for i in range(8): # there are 8 conditions
        # initialize
        a = coor0
        b = coor1
        count = 0
        
        while 1:
            count += 1
            a += aChange[i]
            b += bChange[i]

            # if out of range -> break
            if a < 0 or b < 0 or a >= len(board) or b >= len(board[0]): break
            
            if board[a][b] == str(turn):
                if count >= 2: return 1
                break
            elif board[a][b] == '-': break

    return 0

# modify board after the turn
def modifyBoard(board, coor0, coor1, turn):
    board[coor0][coor1] = str(turn)

    a = coor0
    b = coor1
    aChange = [-1, -1, -1, 0, 1, 1, 1, 0] # LEFT-UP, UP, RIGHT-UP, ..., LEFT
    bChange = [-1, 0, 1, 1, 1, 0, -1, -1] # LEFT-UP, UP, RIGHT-UP, ..., LEFT
    
    for i in range(8): # there are 8 conditions
        # initialize
        a = coor0
        b = coor1
        count = 0
        reverseCoor = [] # list of coordinates where the stones are reversed
        
        while 1:
            count += 1
            a += aChange[i]
            b += bChange[i]
            reverseCoor.append([a, b])

            # if out of range -> break
            if a < 0 or b < 0 or a >= len(board) or b >= len(board[0]): break
            
            if board[a][b] == str(turn):
                if count >= 2:
                    # reverse stones
                    for j in range(len(reverseCoor)):
                        board[reverseCoor[j][0]][reverseCoor[j][1]] = str(turn)
                break
            elif board[a][b] == '-': break

# play game
def playGame(games, bSize, getSco_, getVal_, checkCondi_, modifyBoard_, getCount_):
    for game in range(games):
        print(' ******** GAME ' + str(game) + ' ********')
        print('')
        
        # 1-0. make input table
        Oput = [[0]*int(bSize/2) for i in range(int(bSize/2))]
        Xput = [[0]*int(bSize/2) for i in range(int(bSize/2))]
        
        # 1-1. make board
        board = [['-']*bSize for i in range(bSize)]
        for i in range(bSize):
            for j in range(bSize):
                board[i][j] = '-'

        board[int(bSize/2)-1][int(bSize/2)-1] = 'O'
        board[int(bSize/2)-1][int(bSize/2)] = 'X'
        board[int(bSize/2)][int(bSize/2)-1] = 'X'
        board[int(bSize/2)][int(bSize/2)] = 'O'

        tempBoard = [['-']*bSize for i in range(bSize)]

        # 1-2. print default board
        for i in range(bSize):
            print(board[i])

        # 1-3. playing game
        turns = 0
        drawturns = 0
        while(1):

            # make temp board (store previous board)
            for i in range(bSize):
                for j in range(bSize):
                    tempBoard[i][j] = board[i][j]
            
            # turn check
            if turns % 2 == 0: # turn of O
                tree = GamePlaying.spanTree(board, 'O', bSize, getSco_, getVal_, checkCondi_, modifyBoard_)
                if len(tree) == 1: # no next move
                    print('no next move for player O')
                    drawturns += 1
                    turns += 1
                    if drawturns == 1: continue
                else: drawturns = 0
                board = GamePlaying.findAnswer(tree, 'O')
            else: # turn of X
                tree = GamePlaying.spanTree(board, 'X', bSize, getSco_, getVal_, checkCondi_, modifyBoard_)
                if len(tree) == 1: # no next move
                    print('no next move for player X')
                    drawturns += 1
                    turns += 1
                    if drawturns == 1: continue
                else: drawturns = 0
                board = GamePlaying.findAnswer(tree, 'X')

            # add where the stone was put, to DNN input
            # find the index
            puti = -1 # i index of place where stone was put
            putj = -1 # j index of place where stone was put
            for i in range(bSize):
                broken = 0
                for j in range(bSize):
                    if tempBoard[i][j] == '-' and board[i][j] != '-':
                        broken = 1
                        puti = i
                        putj = j
                        break
                if broken == 1: break

            # add to Oput or Xput using the index
            if puti >= 0 and putj >= 0:
                imin = min(puti, bSize-1-puti)
                jmin = min(putj, bSize-1-putj)
                if turns % 2 == 0: # tuen of O
                    if imin > jmin: Oput[imin][jmin] += 1
                    else: Oput[jmin][imin] += 1
                else: # turn of X
                    if imin > jmin: Xput[imin][jmin] += 1
                    else: Xput[jmin][imin] += 1

            # print
            print('')
            print('board at turn ' + str(turns+1))
            print('O ' + str(getCount_(board, 'O')) + ' : X ' + str(getCount_(board, 'X')))
            for i in range(bSize):
                print(board[i])

            # count blanks
            blanks = 0
            for i in range(bSize):
                for j in range(bSize):
                    if board[i][j] == '-': blanks += 1

            # victory check
            if blanks == 0 or getCount_(board, 'O') == 0 or getCount_(board, 'X') == 0 or drawturns >= 2:
                print('FINAL RESULT: O ' + str(getCount_(board, 'O')) + ' : X ' + str(getCount_(board, 'X')))
                print('')
                
                check = getCount_(board, 'O') - getCount_(board, 'X')
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

        # 1-4. make input and output for this game
        bHalf = int(bSize/2)
        
        # make input
        temp = []
        for i in range(bHalf):
            for j in range(i+1):
                if Oput[i][j]+Xput[i][j] == 0: continue
                temp.append(Oput[i][j] / (Oput[i][j]+Xput[i][j]))
        # check if there are some element that has no data
        needs = int(bHalf*(bHalf+1)/2)-1
        if len(temp) != needs:
            print('need ' + str(needs) + ' data, but given ' + str(len(temp)) + ' data')
            print('')
            continue
        input_.append(temp)

        # make output (sigmoid(rate of O))
        Orate = (getCount_(board, 'O') - getCount_(board, 'X')) / (getCount_(board, 'O') + getCount_(board, 'X'))
        sigmOrate = DNN.sigmoid(Orate, 0)
        output_.append([sigmOrate])

        # 1-5. print result
        print('**** Oput ****')
        for i in range(len(Oput)): print(printVec(Oput[i], 6))
        print('**** Xput ****')
        for i in range(len(Xput)): print(printVec(Xput[i], 6))
        print('')
        print('input : ' + printVec(temp, 6))
        print('output: ' + printVec([sigmOrate], 6))
        print('')

# 0. read file
f = open('DNN_othello.txt', 'r')
read = f.readlines()
bSize = int(read[0]) # size of board

gameN = read[1].split('\n')[0].split(' ')
games = int(gameN[0]) # number of games before making DNN
aftergames = int(gameN[1]) # number of games after making DNN (for each stage)
stages = int(gameN[2]) # number of stages (repeat)

neurons = read[2].split('\n')[0].split(' ') # number of neurons in each hidden layer
f.close()

if bSize % 2 != 0:
    print('board size must be even number.')
    sys.exit(1)

input_ = [] # DNN input
output_ = [] # DNN output

# 1. repeat playing game
playGame(games, bSize, getSco, getVal, checkCondi, modifyBoard, getCount)

# 2. print result
for i in range(games):
    print('input: ' + printVec(input_[i], 4) + ', output: ' + printVec(output_[i], 6))

# 3. DNN learning
h0Nn = int(neurons[0])
h1Nn = int(neurons[1])
h2Nn = int(neurons[2])
(iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw) = DNN.Backpropagation(input_, output_, h0Nn, h1Nn, h2Nn, 3.25, -2, input_[0], 0, 0)
print('')

# 4. make DNN array (repeat)
for _ in range(stages):
    print(' ******** LEARNING: STAGE ' + str(stages) + ' ********')
    print('')
    
    DNNarray = [[0]*bSize for i in range(bSize)]
    for i in range(bSize):
        for j in range(bSize):
            coor0 = i
            coor1 = j

            # to match coordinate with DNN input
            if coor0 >= int(bSize/2): coor0 = (bSize-1)-coor0
            if coor1 >= int(bSize/2): coor1 = (bSize-1)-coor1
            if coor0 < coor1: (coor0, coor1) = (coor1, coor0)

            # make DNN input: {index->1.0, not index->0.5}
            inputIndex = coor0*(coor0+1)/2 + coor1
            input_ = []
            for k in range(iNn):
                if k == inputIndex: input_.append(1.0)
                else: input_.append(0.5)
            print('(' + str(i) + ',' + str(j) + ') -> DNN input = ' + printVec(input_, 4))

            # get output
            (hidden0Input, hidden0Output, hidden1Input, hidden1Output, hidden2Input, hidden2Output, outputInput, outputOutput) = DNN.forward([input_], 0, 0, -2, 0, iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw)
            DNNarray[i][j] = outputOutput[0] - 0.5
    print('')

    print('<DNN value array>')
    for i in range(bSize): # print
        print(printVec(DNNarray[i], 4))
    print('')

    # 5. play game using result of DNN
    input_ = [] # DNN input
    output_ = [] # DNN output
    playGame(aftergames, bSize, getScoUsingDNN, getVal, checkCondi, modifyBoard, getCount)

    for i in range(aftergames):
        print('input: ' + printVec(input_[i], 4) + ', output: ' + printVec(output_[i], 6))

    # 6. learning from this stage
    (iNn, h0Nn, h0Nt, h0Nw, h1Nn, h1Nt, h1Nw, h2Nn, h2Nt, h2Nw, oNn, oNt, oNw) = DNN.Backpropagation(input_, output_, h0Nn, h1Nn, h2Nn, 3.25, -2, input_[0], 0, 0)
    print('')
