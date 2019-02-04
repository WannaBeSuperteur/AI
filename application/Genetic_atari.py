import os, imp
imp.load_source('Genetic', os.path.join(os.path.dirname(__file__), "../Genetic.py"))
import Genetic
import DNN_atari as atari
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

    # find start and end index of bar
    barStart = -1
    barEnd = -1
    for i in range(width):
        if board[height-1][i] == '-':
            barStart = i
            break
    for i in range(barStart, width):
        if board[height-1][i] == '.':
            barEnd = i
            break

    # find ball
    ballIndex = [-1, -1]
    for i in range(height):
        broken = 0
        for j in range(width):
            if board[i][j] == 'O':
                ballIndex = [i, j]
                broken = 1
                break
        if broken == 1: break
    ballVector = [-1, -1] # direction(vector) of ball moving
    
    # get result
    for i in range(len(inp)):

        # take action
        moveBar = 0
        # GO RIGHT
        if inp[i] == 2:
            if barEnd < width: moveBar = 1
            else: moveBar = 0
        # GO LEFT
        elif inp[i] == 0:
            if barStart > 0: moveBar = -1
            else: moveBar = 0
        # if input value is 1, do nothing
        # do action
        
        (result, ballIndex, ballVector) = atari.takeAction(board, moveBar, result, barStart, barEnd, ballIndex, ballVector)
        barStart += moveBar
        barEnd += moveBar

        # game over
        if ballIndex == 'F': break

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
        valueList = [0, 1, 2]
        bestmix = Genetic.Genetic(board, evaluate, numArray, arrayLen, p, valueList, iters, prt)

        # print the best way
        print('')
        print(bestmix)
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
f = open('Genetic_atari.txt', 'r')
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
