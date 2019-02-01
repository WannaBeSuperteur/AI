import os, imp
imp.load_source('GamePlaying', os.path.join(os.path.dirname(__file__), "../GamePlaying.py"))
import GamePlaying

# othello
MAXDEP = 4

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

# check if victory (end of game)
def victory(board, turn, scoreFunc):
    size = len(board)
    
    # count blanks
    blanks = 0
    for i in range(size):
        for j in range(size):
            if board[i][j] == '-': blanks += 1

    # victory check
    if blanks == 0 or getCount(board, 'O') == 0 or getCount(board, 'X') == 0: return 1
    else: return 0

# 0. read file
f = open('GamePlaying_othello.txt', 'r')
read = f.readlines()
bSize = int(read[0]) # size of board
f.close()

# 1. make board
board = [['-']*bSize for i in range(bSize)]
for i in range(bSize):
    for j in range(bSize):
        board[i][j] = '-'

board[int(bSize/2)-1][int(bSize/2)-1] = 'O'
board[int(bSize/2)-1][int(bSize/2)] = 'X'
board[int(bSize/2)][int(bSize/2)-1] = 'X'
board[int(bSize/2)][int(bSize/2)] = 'O'

# 2. print default board
for i in range(bSize):
    print(board[i])

# 3. playing game
turns = 0
drawturns = 0
while(1):
    
    # turn check
    if turns % 2 == 0: # turn of O
        tree = GamePlaying.spanTree(board, 'O', bSize, getSco, getVal, checkCondi, modifyBoard, victory)
        if len(tree) == 1: # no next move
            print('no next move for player O')
            drawturns += 1
            turns += 1
            if drawturns == 1: continue
        else: drawturns = 0
        board = GamePlaying.findAnswer(tree, 'O')
    else: # turn of X
        tree = GamePlaying.spanTree(board, 'X', bSize, getSco, getVal, checkCondi, modifyBoard, victory)
        if len(tree) == 1: # no next move
            print('no next move for player X')
            drawturns += 1
            turns += 1
            if drawturns == 1: continue
        else: drawturns = 0
        board = GamePlaying.findAnswer(tree, 'X')

    # print
    print('')
    print('board at turn ' + str(turns+1))
    print('O ' + str(getCount(board, 'O')) + ' : X ' + str(getCount(board, 'X')))
    for i in range(bSize):
        print(board[i])

    # count blanks
    blanks = 0
    for i in range(bSize):
        for j in range(bSize):
            if board[i][j] == '-': blanks += 1

    # victory check
    if blanks == 0 or getCount(board, 'O') == 0 or getCount(board, 'X') == 0 or drawturns >= 2:
        print('FINAL RESULT: O ' + str(getCount(board, 'O')) + ' : X ' + str(getCount(board, 'X')))
        check = getCount(board, 'O') - getCount(board, 'X')
        
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
