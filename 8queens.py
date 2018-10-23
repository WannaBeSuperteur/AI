import random

# read file
# file = open('queen.txt', 'r')
# read = file.readlines()
# file.close
# read_split = read[0].split(' ')

N = 15
success = 0
alltry = 0
boardcount = 0
while alltry < 3000:
    count = 0
    print('<No.' + str(alltry+1) + '> / success=' + str(success) + ' / board_count=' + str(boardcount))

    board = [[0]*N for i in range(N)]
    boards = []

    queenindex = []
    for i in range(N):
        a = random.randint(0, N-1)
        queenindex.append(a)
        board[a][i] = 1

    print(queenindex)
    print('')

    # iterative
    while 1:
        
        if count >= 1:
            before_minval = minval
        else:
            before_minval = 1000
        count += 1
        boardcount += 1

        evaluate = [[0]*N for i in range(N)]
        minval = 1000
        opti_row = -1
        opti_col = -1

        # find the minimum solution
        for i in range(N):

            # find the queen in column i
            for j in range(N):
                if board[j][i] == 1:
                    index = j
                    break

            # move the queen and evaluat
            for j in range(N):

                # make temporary board
                temp_board = [[0]*N for q in range(N)]
                for k in range(N):
                    for l in range(N):
                        temp_board[k][l] = board[k][l]
                
                # if queen exist, continue
                if j == index:
                    evaluate[j][i] = minval
                    continue

                # move queen on temporary board
                temp_board[index][i] = 0
                temp_board[j][i] = 1
                value = 0

                # evaluate
                # 0. each row
                for k in range(N):
                    queens = 0
                    for l in range(N):
                        if temp_board[k][l] == 1: queens += 1
                    if queens > 1: value += (queens-1)

                # 1. diagonal
                # (10-01)(20-11-02)(30-21-12-03)...(70-61-52-...-25-16-07)
                for k in range(1, N):
                    queens = 0
                    for l in range(k+1):
                        if temp_board[k-l][l] == 1:
                            queens += 1
                    if queens > 1: value += (queens-1)
                # (71-62-53-...-26-17)(72-63-54-45-36-27)(73-64-55-46-37)...(76-67)
                for k in range(1, N-1):
                    queens = 0
                    for l in range(N-k):
                        if temp_board[N-1-l][k+l] == 1:
                            queens += 1
                    if queens > 1: value += (queens-1)
                # symmetric(left-right)
                for k in range(1, N):
                    queens = 0
                    for l in range(k+1):
                        if temp_board[k-l][N-1-l] == 1:
                            queens += 1
                    if queens > 1: value += (queens-1)
                for k in range(1, N-1):
                    queens = 0
                    for l in range(N-k):
                        if temp_board[N-1-l][N-1-k-l] == 1:
                            queens += 1
                    if queens > 1: value += (queens-1)

                evaluate[j][i] = value
                    
        # print each row
        if count == 1:
            print('        <BOARD ' + str(count-1) + ': INITIAL>')
        else:
            print('        <BOARD ' + str(count-1) + ': ' + str(before_minval) + '>')
        for k in range(N):
            queenrow = '        '
            for l in range(N):
                if board[k][l] == 1:
                    queenrow += '<> '
                else:
                    if evaluate[k][l] >= 10:
                        queenrow += str(evaluate[k][l]) + ' '
                    else:
                        queenrow += ' ' + str(evaluate[k][l]) + ' '
            print(queenrow)
        print('')

        # find the min evaluated
        for i in range(N):        
            for j in range(N):
                if minval > evaluate[i][j] and board[i][j] == 0:
                    minval = evaluate[i][j]
                    opti_row = i
                    opti_col = j

        # move queen
        for i in range(N):
            if board[i][opti_col] == 1:
                board[i][opti_col] = 0
                board[opti_row][opti_col] = 1
                break

        # save the board
        temp = [[0]*N for q in range(N)]
        for i in range(N):
            for j in range(N):
                temp[i][j] = board[i][j]
        boards.append(temp)

        # find same board
        same_exist = 0
        for i in range(len(boards)-2):
            same = 1
            for j in range(N):
                for k in range(N):
                    if boards[i][j][k] != board[j][k]:
                        same = 0
                        break
                if same == 0: break
            if same == 1:
                same_exist = 1
                break

        if before_minval == 0: success += 1
        if before_minval == 0 or same_exist == 1:
            boardcount -= 1
            break
    alltry += 1

print('try=' + str(alltry) + ', success=' + str(success) + ', rate=' + str(int(10000*success/alltry)/100) + '%')
print('boardcount=' + str(boardcount) + ', avg_boardcount=' + str(int((boardcount/alltry)*100)/100))
