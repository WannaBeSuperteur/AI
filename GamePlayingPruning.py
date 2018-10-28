import time
# GERM GAME (NxN)
N = 6

# get score of each player
def getScore(board_, OorX):
    result = 0
    for i in range(N):
        for j in range(N):
            if board_[i][j] == OorX: result += 1
    return result

# get score of player 1
def getValue(board_):
    score_of_p1 = 0
    p1_score = getScore(board_, 'O')
    p2_score = getScore(board_, 'X')
    return p1_score-p2_score

# span the tree with MAXDEPTH
def spanTree(board_, turn, depth):
    tree = [] # Game Search Tree (0=board, 1=value, 2=id, 3=parent, 4=depth,
              # 5=alpha, 6=beta, 7=child start id, 8=child end id)
    tree.append([board_, getValue(board_), 0, -1, 0, -999, 999, -1, -1]) # append initial board
    new_id = 1 # ID of node of tree
    start_id = 0
    end_id = 1
    
    for i in range(depth):

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
            if getScore(board1, 'O') + getScore(board1, 'X') == N*N: continue
            if getScore(board1, 'O') == 0: continue
            if getScore(board1, 'X') == 0: continue
            
            # find legal move for player
            tree[id1][7] = new_id
            for j in range(N):
                for k in range(N):
                    # count (around 8 cells)
                    count = 0
                    for l in range(j-1, j+2):
                        for m in range(k-1, k+2):
                            if l < 0 or l >= N or m < 0 or m >= N: continue
                            if board1[l][m] == str(turn): count = 1

                    # for each legal cell
                    if count > 0 and board1[j][k] == '-':

                        # make temp board
                        board2 = [['-']*N for ii in range(N)]
                        for l in range(N):
                            for m in range(N):
                                board2[l][m] = board1[l][m]

                        # update temp board
                        board2[j][k] = str(turn)
                        for l in range(j-1, j+2):
                            for m in range(k-1, k+2):
                                if l < 0 or l >= N or m < 0 or m >= N: continue
                                if str(turn) == 'O' and board2[l][m] == 'X':
                                    board2[l][m] = 'O'
                                if str(turn) == 'X' and board2[l][m] == 'O':
                                    board2[l][m] = 'X'

                        # calculate score value
                        scoreVal = getValue(board2)

                        # append to Game Search Tree
                        tree.append([board2, scoreVal, new_id, id1, i+1, -999, 999, -1, -1]) # append new board
                        new_id += 1
            tree[id1][8] = new_id

        end_id = new_id # mark: END

        # change turn
        if turn == 'O': turn = 'X'
        elif turn == 'X': turn = 'O'

    return tree # return GST

# find answer in the tree
def findAnswer(tree, turn, depth, p):

    if len(tree) == 1: return tree[0][0]

    # copy tree
    tree_copy = []
    for i in range(len(tree)): tree_copy.append(tree[i])
    d = depth

    final_id = -1 # node ID of final move (depth 1)
    now_id = 0

    # node visit check
    visited = []
    for i in range(len(tree)): visited.append(0)

    # reassign value for nonleaf
    for i in range(len(tree)):
        if tree_copy[i][7] != tree_copy[i][8]:
            if (tree_copy[i][4] % 2 == 0 and turn == 'O') or (tree_copy[i][4] % 2 == 1 and turn == 'X'): tree_copy[i][1] = -999 # max node
            else: tree_copy[i][1] = 999 # min node

    count = 0
    while 1:
        count += 1
        d = tree_copy[now_id][4] # update depth
        visited[now_id] += 1

        # if alpha > beta, pruning
        pruning = 0
        if tree_copy[now_id][5] > tree_copy[now_id][6]:
            if p != 0:
                pruning = 1
                for i in range(tree_copy[now_id][7], tree_copy[now_id][8]): visited[i] = 1
            # print('pruning(' + str(now_id) + ')')

        # if not root node, UP max or min value
        if (now_id != 0 and tree_copy[now_id][7] == tree_copy[now_id][8]) or visited[now_id] >= 2:
            val = tree_copy[now_id][1] # value of this node

            start = tree_copy[now_id][7]
            end = tree_copy[now_id][8]
            leafnode = 0
            visited_allchild = 1
            if start == end: leafnode = 1
            elif visited[end-1] == 0: visited_allchild = 0

            # UP VALUE, alpha or beta only when leaf node or visited all child
            if leafnode == 1 or visited_allchild == 1:
                
                # UP: MIN (this is MAX node)
                if (d % 2 == 0 and turn == 'O') or (d % 2 == 1 and turn == 'X'):
                    # UP VALUE
                    if val < tree_copy[tree_copy[now_id][3]][1] and val > -999:                                            
                        tree_copy[tree_copy[now_id][3]][1] = val
                        # print(str(tree_copy[now_id][3]) + ' -> ★ = ' + str(val))
                    # UP beta
                    if val < tree_copy[tree_copy[now_id][3]][6] and val > -999:
                        tree_copy[tree_copy[now_id][3]][6] = val
                        # print(str(tree_copy[now_id][3]) + ' -> beta _= ' + str(val))
                        
                # UP: MAX (this is MIN node)
                else:
                    # UP VALUE
                    if val > tree_copy[tree_copy[now_id][3]][1] and val < 999:                        
                        tree_copy[tree_copy[now_id][3]][1] = val
                        # print(str(tree_copy[now_id][3]) + ' -> ★ = ' + str(val))                           
                    # UP alpha
                    if val > tree_copy[tree_copy[now_id][3]][5] and val < 999:
                        tree_copy[tree_copy[now_id][3]][5] = val
                        # print(str(tree_copy[now_id][3]) + ' -> alpha_= ' + str(val))
                        
        # if non-leaf node, DOWN alpha and beta value
        for i in range(tree_copy[now_id][7], tree_copy[now_id][8]):
            if tree_copy[i][7] == tree_copy[i][8]: continue # do not DOWN for leaf node
            if visited[i] > 0: continue # do not DOWN for visited node
            
            if tree_copy[i][5] < tree_copy[now_id][5]: # alpha value
                tree_copy[i][5] = tree_copy[now_id][5]
                # print(str(i) + ' -> alpha = ' + str(tree_copy[i][5]))
            if tree_copy[i][6] > tree_copy[now_id][6]: # beta value
                tree_copy[i][6] = tree_copy[now_id][6]
                # print(str(i) + ' -> beta  = ' + str(tree_copy[i][6]))

        # find next node (update now_id)
        if tree_copy[now_id][7] == tree_copy[now_id][8] or pruning == 1: now_id = tree_copy[now_id][3]
        else:
            all_visited = 1
            # for each child
            for i in range(tree_copy[now_id][7], tree_copy[now_id][8]):
                if visited[i] == 0:
                    now_id = i
                    all_visited = 0
                    break
            # if all child visited, go to parent
            if all_visited == 1:
                if now_id == 0: break
                now_id = tree_copy[now_id][3]

    # decide final move
    if turn == 'O': # maximize value
        fvalue = -999
        for i in range(tree_copy[0][7], tree_copy[0][8]):
            if tree_copy[i][1] > fvalue:
                fvalue = tree_copy[i][1]
                final_id = i
    else: # minimize value
        fvalue = 999
        for i in range(tree_copy[0][7], tree_copy[0][8]):
            if tree_copy[i][1] < fvalue:
                fvalue = tree_copy[i][1]
                final_id = i

    #for i in range(len(tree_copy)):
    #    print(tree_copy[i])

    board = tree_copy[final_id][0]
    return (board, count)

# 0. make board
board = [['-']*N for i in range(N)]
for i in range(N):
    for j in range(N):
        board[i][j] = '-'
board[0][0] = 'O'
board[5][5] = 'X'

# 1. print default board
for i in range(N):
    print(board[i])

# 2. playing game
turns = 0
counting = 0
Oname = 'Goddess'
Xname = 'Man'
Odepth = 4 # tree depth of O
Xdepth = 4 # tree depth of X
p = 1 # pruning?
start_time = time.time()
while(1):
    # turn check
    tree = []
    if turns % 2 == 0:
        tree = spanTree(board, 'O', Odepth)
        if len(tree) == 1:
            turns += 1
            continue
        (board, count) = findAnswer(tree, 'O', Odepth, p)
    else:
        tree = spanTree(board, 'X', Xdepth)
        if len(tree) == 1:
            turns += 1
            continue
        (board, count) = findAnswer(tree, 'X', Xdepth, p)
    counting += count

    # print
    print('')
    print('board at turn ' + str(turns+1))
    for i in range(N):
        print(board[i])
    print('<' + Oname + '> = ' + str(getScore(board, 'O')) + ', <' + Xname + '> = ' + str(getScore(board, 'X')))

    # zero score check
    if getScore(board, 'O') == 0:
        print('[' + Xname + '] victory')
        break
    if getScore(board, 'X') == 0:
        print('[' + Oname + '] victory')
        break

    # victory check (when there is no blank)
    blanks = 0
    for i in range(N):
        for j in range(N):
            if board[i][j] == '-': blanks = blanks + 1
    if blanks == 0:
        if getValue(board) > 0: print('[' + Oname + '] victory')
        elif getValue(board) == 0: print('draw')
        else: print('[' + Xname + '] victory')
        break

    turns += 1
end_time = time.time()
print('total loop: ' + str(counting))
print('total time: ' + str(end_time - start_time))
