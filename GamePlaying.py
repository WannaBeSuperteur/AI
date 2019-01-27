import random

# TIC-TAC-TOE
MAXDEP = 4

# get count of O or X
def getCount(board_, OorX, nums): # nums: list of coordinates (eg. [10, 11, 12])
    result = 0
    for i in range(len(nums)):
        if board_[int(nums[i]/10)][nums[i]%10] == OorX:
            result += 1

    if len(nums) == 3: # line -> 0->0, 1->1, 2->10, 3->100
        if result == 0: return 0
        elif result == 1: return 1
        elif result == 2: return 10
        elif result == 3: return 100
    else: # entire board -> count
        return result

# get score of each player
def getScore(board_, OorX):
    result = 0

    # order: victory -> count of 2/3 -> count of 1/3
    result += getCount(board_, OorX, [0, 1, 2])
    result += getCount(board_, OorX, [10, 11, 12])
    result += getCount(board_, OorX, [20, 21, 22])
    result += getCount(board_, OorX, [0, 10, 20])
    result += getCount(board_, OorX, [1, 11, 21])
    result += getCount(board_, OorX, [2, 12, 22])
    result += getCount(board_, OorX, [0, 11, 22])
    result += getCount(board_, OorX, [2, 11, 20])
    
    if result >= 100: # if made a line (victory) -> min (OorX)s
        result = 109 - getCount(board_, OorX, [0, 1, 2, 10, 11, 12, 20, 21, 22])
        
    return result

# get score of player 1
def getValue(board_, scoreFunc):
    score_of_p1 = 0 # score of player 1 (n(O)-n(X))
    # O : Line(3) + 10*Line(2) + 100*Line(1)
    p1_score = scoreFunc(board_, 'O')
    # X : Line(3) + 10*Line(2) + 100*Line(1)
    p2_score = scoreFunc(board_, 'X')
    return p1_score-p2_score

# check condition
def checkCondi(board, coor0, coor1, turn):
    if board[coor0][coor1] == '-': return 1
    else: return 0

# modify board after the turn
def modifyBoard(board, coor0, coor1, turn):
    board[coor0][coor1] = str(turn)

# span the tree with MAXDEPTH
def spanTree(board_, turn, bSize, scoreFunc, valueFunc, condiFunc, modiFunc):
    tree = [] # Game Search Tree (0=board, 1=value, 2=id, 3=parent, 4=depth)
    tree.append([board_, getValue(board_, scoreFunc), 0, -1, 0]) # append initial board
    new_id = 1 # ID of node of tree
    start_id = 0
    end_id = 1
    
    for i in range(MAXDEP):

        # temptree <- tree
        temptree = []
        for j in range(start_id, end_id): temptree.append(tree[j])
        
        start_id = new_id # mark: START

        # each node of temptree -> append child to tree
        for ii in range(len(temptree)):

            # get board and value of the node
            board1 = temptree[ii][0]
            id1 = temptree[ii][2]

            # if victory of O or X -> continue
            if scoreFunc(board1, 'O') >= 100: continue
            if scoreFunc(board1, 'X') >= 100: continue
            
            # find legal move for player
            for j in range(bSize):
                for k in range(bSize):
                    if condiFunc(board1, j, k, turn) != 0:

                        # make temp board
                        board2 = [['-']*bSize for ii in range(bSize)]
                        for l in range(bSize):
                            for m in range(bSize):
                                board2[l][m] = board1[l][m]
                        modiFunc(board2, j, k, turn)

                        # calculate score value
                        scoreVal = valueFunc(board2, scoreFunc)

                        # append to Game Search Tree
                        tree.append([board2, scoreVal, new_id, id1, i+1]) # append new board
                        new_id += 1

        end_id = new_id # mark: END

        # change turn
        if turn == 'O': turn = 'X'
        elif turn == 'X': turn = 'O'

    return tree # return GST

# find answer in the tree
def findAnswer(tree, turn):

    # copy tree
    tree_copy = []
    for i in range(len(tree)): tree_copy.append(tree[i])
    d = MAXDEP

    # UP
    while 1:
        # append node of depth d (MAXDEP ~ 1)
        tree0 = []
        for i in range(len(tree_copy)):
            if tree_copy[i][4] == d: tree0.append(tree_copy[i])
        
        # decide node to UP
        minval = 999
        maxval = -999
        final_id = -1 # node ID of final move (depth 1)
        last_parent = -1
        
        for i in range(len(tree0)):
            lenT = len(tree0)
            randElement = random.randint(0, lenT)

            # if different(new parent), initialize min/maxval
            if last_parent != tree0[i][3]:
                minval = 999
                maxval = -999
            
            # UP: MIN
            if (d % 2 == 0 and turn == 'O') or (d % 2 == 1 and turn == 'X'):
                val = tree0[i][1] # value of this node
                if val < minval or (val <= minval and i < randElement):
                    minval = val
                    if d == 1: final_id = tree0[i][2] # update final decision
                tree_copy[tree0[i][3]][1] = minval # update minval

            # UP: MAX
            else:
                val = tree0[i][1] # value of this node
                if val > maxval or (val >= maxval and i < randElement):
                    maxval = val
                    if d == 1: final_id = tree0[i][2] # update final decision
                tree_copy[tree0[i][3]][1] = maxval # update maxval

            last_parent = tree0[i][3] # check if different (if diff, initialize min/maxval)

        # update depth
        d -= 1
        if d == 0: break # break when reach root

    # decide final move
    board = tree_copy[final_id][0]
    return board

if __name__ == '__main__':
    # 0. make board
    bSize = 3
    board = [['-']*bSize for i in range(bSize)]
    for i in range(bSize):
        for j in range(bSize):
            board[i][j] = '-'

    # 1. print default board
    for i in range(bSize):
        print(board[i])

    # 2. playing game
    turns = 0
    while(1):
        # turn check
        if turns % 2 == 0:
            tree = spanTree(board, 'O', bSize, getScore, getValue, checkCondi, modifyBoard)
            board = findAnswer(tree, 'O')
        else:
            tree = spanTree(board, 'X', bSize, getScore, getValue, checkCondi, modifyBoard)
            board = findAnswer(tree, 'X')

        # print
        print('')
        print('board at turn ' + str(turns+1))
        for i in range(bSize):
            print(board[i])

        # victory check
        if getScore(board, 'O') >= 100:
            print('O victory')
            break
        if getScore(board, 'X') >= 100:
            print('X victory')
            break
        blanks = 0
        for i in range(bSize):
            for j in range(bSize):
                if board[i][j] == '-': blanks = blanks + 1
        if blanks == 0:
            print('draw')
            break

        turns += 1
