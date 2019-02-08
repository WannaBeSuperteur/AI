import os, imp
imp.load_source('Astar', os.path.join(os.path.dirname(__file__), "../AstarSearch.py"))
import Astar
import sys
import math
import random

# read data from file
def getData():
    # get data
    file = open('Astar_8puzzle.txt', 'r')
    read = file.readlines()
    file.close

    config = read[0].split('\n')[0].split(' ')
    prt = int(config[0]) # print?
    timeSec = float(config[1]) # time (seconds)

    # read maze data
    data = []
    for i in range(1, len(read)):
        row = read[i].split('\n')[0].split(' ')
        data.append(row)
        
    return (data, prt, timeSec)

# find value in the array
def findArray(value, array):
    for i in range(len(array)):
        for j in range(len(array[0])):
            if array[i][j] == value: return [i, j]
    return [-1, -1]

# distance function
def distance_(data, etc):
    goal = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '.']]

    Sum = 0 # sum of distance of each number in 'goal' and 'data' array
    list_ = ['1', '2', '3', '4', '5', '6', '7', '8']
    for i in range(len(list_)):
        inData = findArray(list_[i], data)
        inGoal = findArray(list_[i], goal)
        Sum += abs(inData[0]-inGoal[0]) + abs(inData[1]-inGoal[1])

    return Sum

# total cost function
def cost_(data, totalCost):
    return totalCost + 1

# copy board
def boardCopy(board):
    result = []
    for i in range(len(board)):
        temp = []
        for j in range(len(board[0])):
            temp.append(board[i][j])
        result.append(temp)
    return result

# find next move
def next_(data, currPoint, distFunc, costFunc, totalCost, Astar, Id, newId):
    
    # find '.' from array 'currPoint'
    blank = 0
    for i in range(len(currPoint)):
        broken = 0
        for j in range(len(currPoint[0])):
            if currPoint[i][j] == '.':
                broken = 1
                blank = i*3 + j
                break
        if broken == 1: break

    result = [] # list of next moves
    newCost = costFunc(data, totalCost) # accumulated cost after move
    bY = int(blank / 3) # Y index of blank
    bX = blank % 3 # X index of blank

    # find next moves
    if blank % 3 > 0: # change with LEFT
        newB = boardCopy(currPoint)
        newB[bY][bX-1], newB[bY][bX] = newB[bY][bX], newB[bY][bX-1]
        dist = distFunc(newB, [])

        result.append([newB, Astar(newCost, dist), newCost, newId, Id])
        newId += 1

    if blank % 3 < 2: # change with RIGHT
        newB = boardCopy(currPoint)
        newB[bY][bX+1], newB[bY][bX] = newB[bY][bX], newB[bY][bX+1]
        dist = distFunc(newB, [])

        result.append([newB, Astar(newCost, dist), newCost, newId, Id])
        newId += 1

    if blank >= 3: # change with UP
        newB = boardCopy(currPoint)
        newB[bY-1][bX], newB[bY][bX] = newB[bY][bX], newB[bY-1][bX]
        dist = distFunc(newB, [])

        result.append([newB, Astar(newCost, dist), newCost, newId, Id])
        newId += 1

    if blank < 6: # change with DOWN
        newB = boardCopy(currPoint)
        newB[bY+1][bX], newB[bY][bX] = newB[bY][bX], newB[bY+1][bX]
        dist = distFunc(newB, [])

        result.append([newB, Astar(newCost, dist), newCost, newId, Id])
        newId += 1
        
    return (newId, result)

# move
def move_(data, nextPoint, count):
    # update board
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j] = nextPoint[i][j]

    # print board
    print(' **** board after turn ' + str(count) + ' ****')
    for i in range(len(data)):
        print(data[i])

# A* result function
def Astar_(cost, distance):
    return cost + distance # simple sum

(data, prt, timeSec) = getData()
Astar.AstarSearch(data, data, distance_, cost_, next_, move_, Astar_, prt, timeSec)
