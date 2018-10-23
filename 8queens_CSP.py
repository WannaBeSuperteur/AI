import random
num_of_cons = []

def make_conboard(board):
    conboard = [[0]*N for i in range(N)]
    for i in range(N):
        for j in range(N):
            if board[i][j] == 1:
                # check row and column
                for k in range(N):
                    conboard[i][k] = 1
                    conboard[k][j] = 1

                # check diagonal
                for k in range(-N+1, N):
                    if i+k >= 0 and i+k < N and j+k >= 0 and j+k < N: conboard[i+k][j+k] = 1
                for k in range(-N+1, N):
                    if i-k >= 0 and i-k < N and j+k >= 0 and j+k < N: conboard[i-k][j+k] = 1

    # get count
    count = 0
    for i in range(N):
        for j in range(N):
            if conboard[i][j] == 1: count += 1

    return (conboard, count)

def find_queens(board, row, col, N):
    result = 0

    # check row
    for i in range(N):
        if board[row][i] == 1 and i != col:
            return 1

    # check diagonal
    for i in range(-N+1, N):
        if row+i >= N or col+i >= N or row+i < 0 or col+i < 0:
            continue
        if board[row+i][col+i] == 1 and i != 0:
            return 1
    for i in range(-N+1, N):
        if row-i >= N or col+i >= N or row-i < 0 or col+i < 0:
            continue
        if board[row-i][col+i] == 1 and i != 0:
            return 1

    return 0

def print_board(board, N, count, num_of_cons):
    print('')
    print('<BOARD ' + str(count) + '>')

    for i in range(N):
        to_be_printed = ''
        for j in range(N):
            to_be_printed += ' ' + str(board[i][j])
        if len(num_of_cons) > 0:
            to_be_printed += ' | ' + num_of_cons[i]
        print(to_be_printed)
    return 0

# solve N-queens problem
# 1. most-constrained
# if option=0 -> random, else -> least_constraining
def most_constrained(N, option):
    board = [[0]*N for i in range(N)]
    if option == 0: print('***** MOST-CONSTRAINED WITH RANDOM *****')
    else: print('***** MOST-CONSTRAINED WITH LEAST-CONSTRAINING *****')
    
    variable_set = [] # value of each variable (-1 if not assigned)
    for i in range(N):
        variable_set.append(-1)
    
    for times in range(N): # (number of columns) times
        num_of_cons = []
        least_legal_values = N+1 # least among number of legal values
        least_legal_column = -1 # index of column whose number of legal values is the least

        # print line
        line = ''
        for i in range(2*N): line += '-'
        print(line)

        # decide column to be set
        decided = 0 # if finished but this value is still 0, return error
        testlist = ''
        for column in range(N): # for each column
        
            # pass if variable was already set
            if variable_set[column] >= 0:
                testlist += ' *'
                continue

            # for each cell in the column
            test = N # number of legal rows (check each row)
            for cell in range(N):
                result = find_queens(board, cell, column, N)
                if result > 0: test -= 1

            # update least_legal information
            if test < least_legal_values and test > 0:
                least_legal_values = test
                least_legal_column = column
                decided = 1

            # add to testlist
            if test < 10: testlist += ' ' + str(test)
            else: testlist += str(test)

        if decided == 0:
            print('there is no legal column')
            return 0

        # set column if legal (random, iterative)
        if option == 0: # RANDOM
            while 1:
                # find appropriate row and update board
                a = random.randint(0, N-1)
                if find_queens(board, a, least_legal_column, N) == 0:
                    variable_set[least_legal_column] = a
                    board[a][least_legal_column] = 1
                    break
        else: # LEAST-CONSTRAINING

            # decide value(index: row)
            least_con_value = N*N+1
            least_con_row = -1 # row index to get the least number of constraints
            for i in range(N): # for each row

                # if violate the constraint, continue
                if find_queens(board, i, least_legal_column, N) > 0:
                    num_of_cons.append('*')
                    continue

                # check the number of constraints (set each row to 1 and make con-board)
                board[i][least_legal_column] = 1
                (conboard, count) = make_conboard(board)
                board[i][least_legal_column] = 0
                
                if count < least_con_value: # using count instead of (count+count_init)
                    least_con_value = count
                    least_con_row = i

                num_of_cons.append(str(count))

            # set board
            variable_set[least_legal_column] = least_con_row
            board[least_con_row][least_legal_column] = 1

        # print the board
        print(testlist + ' <- number of legal vals')
        mark = ''
        for i in range(2*N-1):
            if i == 2*least_legal_column+1: mark += 'T'
            else: mark += ' '
        print(mark)
        print_board(board, N, times+1, num_of_cons)

    return 0

# 2. forward-checking
def forward_checking(N, option, allfind):
    numofsol = 0
    
    if N > 16:
        print('N is too big')
        return
    
    board = [[0]*N for i in range(N)]
    print('***** FORWARD CHECKING *****')
    
    variable_set = [] # value of each variable (-1 if not assigned)
    for i in range(N):
        variable_set.append(-1)

    # DFS
    stack = [] # boards
    substack = [] # row index of column 0, 1, ...
    sol_found = 0 # if found solution, set to 1
    while 1:
        # locate queens and add to stack
        if len(substack) < N: # check the length of substack
            for i in range(N-1, -1, -1):
                substack.append(i)
                if option == 3: print('')
                if option == 1 or option == 3: print('stack:   ' + str(stack))
                if option == 2 or option == 3: print('checking ' + str(substack))
                
                # check violation
                violate = 0
                board = [[0]*N for i in range(N)]
                for j in range(len(substack)): # set board (substack[j]=row index)
                    board[substack[j]][j] = 1
                    if find_queens(board, substack[j], j, N) > 0:
                        board[substack[j]][j] = 0
                        violate = 1
                        break

                # check no solution for a column
                for j in range(N):
                    placed = 0
                    sols = 0
                    for k in range(N):
                        # if placed, continue
                        if board[k][j] == 1:
                            placed = 1
                            break

                        # no solution?
                        if find_queens(board, k, j, N) == 0:
                            sols = 1
                            break

                    # if there is a row that no queen placed and no solution -> violate
                    if placed == 0 and sols == 0:
                        if option > 0: print('no legal value for column ' + str(j))
                        violate = 1
                        break

                # append if not violate
                if violate == 0:
                    new_substack = []
                    for k in range(len(substack)): new_substack.append(substack[k])
                    stack.append(new_substack)

                    # if length is N, it means found solution -> if not allfind, break
                    if len(substack) == N:
                        numofsol += 1
                        sol_found = 1
                        if allfind == 0: break
                        print('solution ' + str(numofsol) + ': ' + str(new_substack))

                substack.pop(len(substack)-1)

        # if length of stack is 0, finish
        if len(stack) == 0 or (allfind == 0 and sol_found > 0): break

        # pop from stack
        substack = stack.pop(len(stack)-1)
    
    # set and print the board
    for i in range(len(substack)): # set board (substack[j]=row index)
        board[substack[i]][i] = 1
    num_of_cons = []

    if allfind == 0:
        print_board(board, N, 0, num_of_cons)
        if sol_found > 0: print('solution found')
        else: print('there is no solution for N=' + str(N))
    else: print(str(numofsol) + ' solutions found')

# most_constrained(6, 0)
# most_constrained(6, 1)
forward_checking(11, 0, 1)
