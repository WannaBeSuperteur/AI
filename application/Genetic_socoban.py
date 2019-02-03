import os, imp
imp.load_source('Genetic', os.path.join(os.path.dirname(__file__), "../Genetic.py"))
import Genetic
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

# take action
def takeAction(board, point, inpVal):
    y = point[0]
    x = point[1]
    
    leftCell = board[y][x-1]
    upCell = board[y-1][x]
    rightCell = board[y][x+1]
    downCell = board[y+1][x]
    
    if board[y][x] == 'V': # if agent is on the goal cell
        board[y][x] = 'X'
    else: board[y][x] = '.'
                
    if inpVal == 0: # GO LEFT
        # if box is on leftCell, modify leftleftCell
        if leftCell == 'B' or leftCell == 'O':
            leftleftCell = board[y][x-2]
            if leftleftCell == '.': board[y][x-2] = 'B'
            elif leftleftCell == 'X': board[y][x-2] = 'O'

        # modify leftCell
        if leftCell == 'X' or leftCell == 'O': board[y][x-1] = 'V' # if leftCell is goal 
        else: board[y][x-1] = 'S'
                    
        point[1] -= 1 # modify the position of agent
                    
    elif inpVal == 1: # GO UP
        # if box is on upCell, modify upupCell
        if upCell == 'B' or upCell == 'O':
            upupCell = board[y-2][x]
            if upupCell == '.': board[y-2][x] = 'B'
            elif upupCell == 'X': board[y-2][x] = 'O'

        # modify upCell
        if upCell == 'X' or upCell == 'O': board[y-1][x] = 'V' # if upCell is goal 
        else: board[y-1][x] = 'S'
                    
        point[0] -= 1 # modify the position of agent

    elif inpVal == 2: # GO RIGHT
        # if box is on rightCell, modify rightrightCell
        if rightCell == 'B' or rightCell == 'O':
            rightrightCell = board[y][x+2]
            if rightrightCell == '.': board[y][x+2] = 'B'
            elif rightrightCell == 'X': board[y][x+2] = 'O'

        # modify rightCell
        if rightCell == 'X' or rightCell == 'O': board[y][x+1] = 'V' # if rightCell is goal 
        else: board[y][x+1] = 'S'
                    
        point[1] += 1 # modify the position of agent

    elif inpVal == 3: # GO DOWN
        # if box is on downCell, modify downdownCell
        if downCell == 'B' or downCell == 'O':
            downdownCell = board[y+2][x]
            if downdownCell == '.': board[y+2][x] = 'B'
            elif downdownCell == 'X': board[y+2][x] = 'O'

        # modify downCell
        if downCell == 'X' or downCell == 'O': board[y+1][x] = 'V' # if downCell is goal 
        else: board[y+1][x] = 'S'
                    
        point[0] += 1 # modify the position of agent

    # evaluate and return result
    result = 0.0
    Xcoll = [] # collection of symbol 'X'
    Bcoll = [] # collection of symbol 'B'
    for i in range(height):
        for j in range(width):
            if board[i][j] == 'O': result += 2.0
            elif board[i][j] == 'X' : Xcoll.append([i, j])
            elif board[i][j] == 'B' : Bcoll.append([i, j])
    # for each box(B), find the closest 'X' using manhattan distance
    # add (1/minDistance) to result
    for i in range(len(Bcoll)):
        minDistance = height+width
        for j in range(len(Xcoll)):
            dist = abs(Bcoll[i][0] - Xcoll[j][0]) + abs(Bcoll[i][1] - Xcoll[j][1])
            if dist < minDistance: minDistance = dist
        result += 1 / minDistance
        
    return result

# restore board
def restoreBoard(board, original):
    for i in range(len(board)):
        for j in range(len(board[0])):
            board[i][j] = original[i][j]

# evaluation function
def evaluate(board, inp, etc):
    result = 0
    height = len(board)
    width = len(board[0])

    # store original board
    tempBoard = []
    for i in range(height):
        temp = []
        for j in range(width):
            temp.append(board[i][j])
        tempBoard.append(temp)

    # find point
    point = [] # position of agent
    for i in range(height):
        broken = 0
        for j in range(width):
            if board[i][j] == 'S':
                point = [i, j]
                broken = 1
                break
        if broken == 1: break
    
    # get result
    for i in range(len(inp)):
        
        # check if possible -> if impossible, immediately return result
        possible = [0, 0, 0, 0] # [LEFT, UP, RIGHT, DOWN] 1 if possible, 0 else

        leftCell = board[point[0]][point[1]-1]
        upCell = board[point[0]-1][point[1]]
        rightCell = board[point[0]][point[1]+1]
        downCell = board[point[0]+1][point[1]]
        
        # GO LEFT
        if leftCell == '.' or leftCell == 'X': possible[0] = 1
        elif leftCell == 'B' or leftCell == 'O':
            leftleftCell = board[point[0]][point[1]-2]
            if leftleftCell == '.' or leftleftCell == 'X': possible[0] = 1

        # GO UP     
        if upCell == '.' or upCell == 'X': possible[1] = 1
        elif upCell == 'B' or upCell == 'O':
            upupCell = board[point[0]-2][point[1]]
            if upupCell == '.' or upupCell == 'X': possible[1] = 1

        # GO RIGHT  
        if rightCell == '.' or rightCell == 'X': possible[2] = 1
        elif rightCell == 'B' or rightCell == 'O':
            rightrightCell = board[point[0]][point[1]+2]
            if rightrightCell == '.' or rightrightCell == 'X': possible[2] = 1

        # GO DOWN
        if downCell == '.' or downCell == 'X': possible[3] = 1
        elif downCell == 'B' or downCell == 'O':
            downdownCell = board[point[0]+2][point[1]]
            if downdownCell == '.' or downdownCell == 'X': possible[3] = 1

        # if impossible, immediately return
        if possible[inp[i]] == 0:        
            restoreBoard(board, tempBoard) # restore original board         
            return result

        # take action
        result = takeAction(board, point, inp[i])

        # print board
        if len(etc) > 0:
            print(' **** after turn ' + str(i) + ' ****')
            for j in range(height):
                print(board[j])
            print('')
 
    restoreBoard(board, tempBoard) # restore original board
    return result

# SOCOBAN: #(wall) .(blank) S(agent point) B(box) X(goal) O(box on goal) V(agent on goal)
# play game (use input and make output)
def playGame(board, width, height, numArray, arrayLen, p, iters, games, prt):

    geneticResult = []

    # perform genetic algorithm for (game) times
    for game in range(games):
        print('******** performing ' + str(game) + ' ********')
        
        # using genetic algorithm, find the best input (called bestmix)
        valueList = [0, 1, 2, 3]
        bestmix = Genetic.Genetic(board, evaluate, numArray, arrayLen, p, valueList, iters, prt)

        # print the best way
        print('')
        bestmixValue = evaluate(board, bestmix, [0])
        print('')

        geneticResult.append([bestmix, bestmixValue])

    # find best choice and print it
    print('genetic results:')
    for i in range(len(geneticResult)):
        print('seq: ' + str(geneticResult[i][0]))
        print('res: ' + str(geneticResult[i][1]))
    print('')

    geneticResult.sort(key=lambda x:x[1])

    # perform the best input sequence
    bestSeq = geneticResult[len(geneticResult)-1][0]
    print('perform BEST among genetic results:')
    print('seq: ' + str(bestSeq))
    print('')
    evaluate(board, bestSeq, [0])

# 0. read file
f = open('Genetic_socoban.txt', 'r')
read = f.readlines()

config = read[0].split('\n')[0].split(' ')
numArray = int(config[0]) # number of arraies
arrayLen = int(config[1]) # length of each array
p = float(config[2]) # probability of mutation
iters = int(config[3]) # number of iterations
prt = int(config[4]) # print?

games = int(read[1].split('\n')[0]) # number of iterations of performing genetic algorithm

board = [] # game board
for i in range(2, len(read)):
    board.append(read[i].split('\n')[0].split(' '))

print(' **** INITIAL GAME BOARD **** ')
for i in range(len(board)):
    print(printStr(board[i]))
print('')

f.close()

input_ = [] # DNN input
output_ = [] # DNN output

# play game
height = len(board)
width = len(board[0])
playGame(board, width, height, numArray, arrayLen, p, iters, games, prt)
print('')
