import os, imp
imp.load_source('Genetic', os.path.join(os.path.dirname(__file__), "../Genetic.py"))
import Genetic
import sys
import math
import random
import DNN_maximizeSum as maxSum

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
    result = 0
    
    # GO LEFT
    if inpVal == 0:
        leftCell = board[y][x-1]
        if leftCell != '.' and leftCell != '#': result = int(leftCell) # update result
        board[y][x-1] = 'S'
        point[1] -= 1 # modify the position of agent

    # GO UP
    elif inpVal == 1:
        upCell = board[y-1][x]
        if upCell != '.' and upCell != '#': result = int(upCell) # update result
        board[y-1][x] = 'S'
        point[0] -= 1 # modify the position of agent

    # GO RIGHT
    elif inpVal == 2:
        rightCell = board[y][x+1]
        if rightCell != '.' and rightCell != '#': result = int(rightCell) # update result
        board[y][x+1] = 'S'
        point[1] += 1 # modify the position of agent

    # GO DOWN
    elif inpVal == 3:
        downCell = board[y+1][x]
        if downCell != '.' and downCell != '#': result = int(downCell) # update result
        board[y+1][x] = 'S'
        point[0] += 1 # modify the position of agent

    board[y][x] = '.'
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

        # GO LEFT
        if point[1] > 0:
            leftCell = board[point[0]][point[1]-1]
            if leftCell != '#': possible[0] = 1

        # GO UP     
        if point[0] > 0:
            upCell = board[point[0]-1][point[1]]
            if upCell != '#': possible[1] = 1

        # GO RIGHT  
        if point[1] < width-1:
            rightCell = board[point[0]][point[1]+1]
            if rightCell != '#': possible[2] = 1

        # GO DOWN
        if point[0] < height-1:
            downCell = board[point[0]+1][point[1]]
            if downCell != '#': possible[3] = 1

        # if impossible, immediately return
        if possible[inp[i]] == 0:        
            restoreBoard(board, tempBoard) # restore original board         
            return result

        # take action
        result = maxSum.takeAction(board, point, inp[i], result)

        # print board
        if len(etc) > 0:
            print(' **** after turn ' + str(i) + ' ****')
            for j in range(height):
                print(board[j])
            print('')
 
    restoreBoard(board, tempBoard) # restore original board
    return result

# MAXIMIZESUM: #(wall) .(blank) S(agent point) number(score item)
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
f = open('Genetic_maximizeSum.txt', 'r')
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
