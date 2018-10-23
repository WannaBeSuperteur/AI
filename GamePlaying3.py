# 3-line for 4*4 board
MAXDEP = 3

# get count of A, B and C
def getCount(board_, ABC, nums, again): # nums: list of coordinates (eg. [10, 11, 12, 13])
    result = 0
    temp = 0
    for i in range(len(nums)):
        to_find = board_[int(nums[i]/10)][nums[i]%10] # value of the cell
        
        # if not continuous, initialize
        if len(nums) <= 4 and to_find != ABC:
            temp = 0
        if board_[int(nums[i]/10)][nums[i]%10] == ABC:
            temp += 1
            if temp > result: result = temp

    # convert result if 'nums' is a line
    if len(nums) <= 4: # line -> 0->0, 1->1, 2->20, 3->400
        if result == 2: result = 20
        elif result >= 3: result = 400

    # if victory of enemy -> result=0
    if again == 1:
        enemy1 = 0
        enemy2 = 0
        enemy3 = 0
        if ABC != 'A': enemy1 = getScore(board_, 'A', 0)
        if ABC != 'B': enemy2 = getScore(board_, 'B', 0)
        if ABC != 'C': enemy3 = getScore(board_, 'C', 0)

        if enemy1 >= 400 or enemy2 >= 400 or enemy3 >= 400:
            result = 0

    return result

# get score of each player
def getScore(board_, ABC, again):
    result = 0

    # order: victory -> count of 2/3 -> count of 1/3
    result += getCount(board_, ABC, [0, 1, 2, 3], again)
    result += getCount(board_, ABC, [10, 11, 12, 13], again)
    result += getCount(board_, ABC, [20, 21, 22, 23], again)
    result += getCount(board_, ABC, [30, 31, 32, 33], again)
    result += getCount(board_, ABC, [0, 10, 20, 30], again)
    result += getCount(board_, ABC, [1, 11, 21, 31], again)
    result += getCount(board_, ABC, [2, 12, 22, 32], again)
    result += getCount(board_, ABC, [3, 13, 23, 33], again)
    result += getCount(board_, ABC, [0, 11, 22, 33], again)
    result += getCount(board_, ABC, [3, 12, 21, 30], again)
    result += getCount(board_, ABC, [2, 11, 20], again)
    result += getCount(board_, ABC, [13, 22, 31], again)
    result += getCount(board_, ABC, [1, 12, 23], again)
    result += getCount(board_, ABC, [10, 21, 32], again)
    
    if result >= 400: # if made a line (victory) -> min (ABC)s and don't subtract max value of enemies
        result = 417 - getCount(board_, ABC, [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33], 0)
        
    return result

# get score of players
def getValue(board_, player_no): # player_no: A=0, B=1, C=2
    # Line(3) + 20*Line(2) + 400*Line(1)
    if player_no == 0: return getScore(board_, 'A', 1)
    elif player_no == 1: return getScore(board_, 'B', 1)
    elif player_no == 2: return getScore(board_, 'C', 1)

# span the tree with MAXDEPTH
def spanTree(board_, turn):
    tree = [] # Game Search Tree (0=board, 1=id, 2=parent, 3=depth, 4=valA, 5=valB, 6=valC)

    # append initial board
    tree.append([board_, 0, -1, 0, getValue(board_, 0), getValue(board_, 1), getValue(board_, 2)])
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
            id1 = temptree[ii][1]

            # if victory of A, B or C -> continue
            if getScore(board1, 'A', 0) >= 400: continue
            if getScore(board1, 'B', 0) >= 400: continue
            if getScore(board1, 'C', 0) >= 400: continue
            
            # find legal move for player
            for j in range(4):
                for k in range(4):
                    if board1[j][k] == '-':

                        # make temp board
                        board2 = [['-']*4 for ii in range(4)]
                        for l in range(4):
                            for m in range(4):
                                board2[l][m] = board1[l][m]
                        board2[j][k] = str(turn)

                        # calculate score value
                        valA = getValue(board2, 0)
                        valB = getValue(board2, 1)
                        valC = getValue(board2, 2)

                        # append to Game Search Tree
                        tree.append([board2, new_id, id1, i+1, valA, valB, valC]) # append new board
                        new_id += 1

        end_id = new_id # mark: END

        # change turn
        if turn == 'A': turn = 'B'
        elif turn == 'B': turn = 'C'
        elif turn == 'C': turn = 'A'

    print('spantree finished')
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
            if tree_copy[i][3] == d: tree0.append(tree_copy[i])
        
        # decide node to UP
        maxA = -5601
        maxB = -5601
        maxC = -5601
        final_id = -1 # node ID of final move (depth 1)
        last_parent = -1
        
        for i in range(len(tree0)):
            new_val = 0 # 1 if find new max val

            # if different(new parent), initialize min/maxval
            if last_parent != tree0[i][2]:
                maxA = -5601
                maxB = -5601
                maxC = -5601
            
            # UP: A-MAX
            if (d % 3 == 1 and turn == 'A') or (d % 3 == 2 and turn == 'C') or (d % 3 == 0 and turn == 'B'):
                val = tree0[i][4] # A value of this node
                if val > maxA:
                    maxA = val
                    if d == 1: final_id = tree0[i][1] # update final decision
                    new_val = 1                    

            # UP: B-MAX
            elif (d % 3 == 1 and turn == 'B') or (d % 3 == 2 and turn == 'A') or (d % 3 == 0 and turn == 'C'):
                val = tree0[i][5] # B value of this node
                if val > maxB:
                    maxB = val
                    if d == 1: final_id = tree0[i][1] # update final decision
                    new_val = 1

            # UP: C-MAX
            else:
                val = tree0[i][6] # C value of this node
                if val > maxC:
                    maxC = val
                    if d == 1: final_id = tree0[i][1] # update final decision
                    new_val = 1

            # update value of A, B and C
            if new_val == 1:
                tree_copy[tree0[i][2]][4] = tree0[i][4] # A
                tree_copy[tree0[i][2]][5] = tree0[i][5] # B
                tree_copy[tree0[i][2]][6] = tree0[i][6] # C
            
            last_parent = tree0[i][2] # check if different (if diff, initialize val of A, B and C)

        # update depth
        d -= 1
        if d == 0: break # break when reach root
        
    # decide final move
    board = tree_copy[final_id][0]
    return board

# 0. make board
board = [['-']*4 for i in range(4)]
for i in range(4):
    for j in range(4):
        board[i][j] = '-'

# 1. print default board
for i in range(4):
    print(board[i])

# 2. playing game
turns = 0
while(1):
    # turn check
    if turns % 3 == 0:
        tree = spanTree(board, 'A')
        board = findAnswer(tree, 'A')
    elif turns % 3 == 1:
        tree = spanTree(board, 'B')
        board = findAnswer(tree, 'B')
    elif turns % 3 == 2:
        tree = spanTree(board, 'C')
        board = findAnswer(tree, 'C')

    # print
    print('')
    print('board at turn ' + str(turns+1))
    for i in range(4):
        print(board[i])

    # victory check
    scoreA = getScore(board, 'A', 0)
    scoreB = getScore(board, 'B', 0)
    scoreC = getScore(board, 'C', 0)
    if getScore(board, 'A', 0) >= 400:
        print('A victory (' + str(scoreA) + ')')
        break
    if getScore(board, 'B', 0) >= 400:
        print('B victory (' + str(scoreB) + ')')
        break
    if getScore(board, 'C', 0) >= 400:
        print('C victory (' + str(scoreC) + ')')
        break

    # draw check
    blanks = 0
    for i in range(4):
        for j in range(4):
            if board[i][j] == '-': blanks = blanks + 1
    if blanks == 0:
        print('draw')
        break

    turns += 1
