import math
import random
import time

# read data from file
def getData():
    # get data
    file = open('AstarSearch.txt', 'r')
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

# find point and goal
def pointAndGoal(data):
    point = []
    goal = []
    
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] == 'P': point = [i, j]
            elif data[i][j] == 'G': goal = [i, j]

    return (point, goal)

# find next moves
def next_(data, currPoint, distFunc, costFunc, totalCost, Astar, Id, newId):
    (point, goal) = pointAndGoal(data)

    y = currPoint[0]
    x = currPoint[1]
    height = len(data)
    width = len(data[0])

    result = [] # list of next moves
    newCost = costFunc(data, totalCost) # accumulated cost after move
    
    if x > 0: # LEFT
        dist = distFunc([y, x-1], goal) # distance      
        if data[y][x-1] != '#':
            result.append([[y, x-1], Astar(newCost, dist), newCost, newId, Id])
            newId += 1
            
    if x < width-1: # RIGHT
        dist = distFunc([y, x+1], goal) # distance
        if data[y][x+1] != '#':
            result.append([[y, x+1], Astar(newCost, dist), newCost, newId, Id])
            newId += 1
            
    if y > 0: # UP
        dist = distFunc([y-1, x], goal) # distance
        if data[y-1][x] != '#':
            result.append([[y-1, x], Astar(newCost, dist), newCost, newId, Id])
            newId += 1
            
    if y < height-1: # DOWN
        dist = distFunc([y+1, x], goal) # distance
        if data[y+1][x] != '#':
            result.append([[y+1, x], Astar(newCost, dist), newCost, newId, Id])
            newId += 1

    return (newId, result)

# evaluation function (distance)
def distance_(point, goal):
    return abs(point[0]-goal[0]) + abs(point[1]-goal[1])

# evaluation function (accumulated cost)
def cost_(data, totalCost):
    return totalCost + 1

# move
def move_(data, nextPoint, count):
    (point, goal) = pointAndGoal(data)
    y = point[0]
    x = point[1]
    yN = nextPoint[0]
    xN = nextPoint[1]

    # update board
    data[y][x] = '.'
    data[yN][xN] = 'P'

    # print board
    print(' **** board after turn ' + str(count) + ' ****')
    for i in range(len(data)):
        print(data[i])

# A* result function
def Astar_(cost, distance):
    return cost + distance # simple sum

# number to string
def nToS(num, n):
    s = str(num)
    return ' '*(n-len(s)) + s

# A* search algorithm
def AstarSearch(startPoint, data, distFunc, costFunc, findNextMove, moveFunc, Astar, prt, timeSec):
    queue = [] # [0]: state after move, [1]: A* result, [2]: cost, [3]: ID [4]: ID of predecessor
    totalCost = 0 # accumulated cost
    
    Id = 0 # id of this element
    newId = 1 # id of element (will be enqueued)

    # tree of elements have been enqueued in the queue
    tree = [[startPoint, 2147483647, 0, 0, -1]] # initialize: information about the start point
    currPoint = startPoint # current point

    # do not search again if finished
    finishList = []

    # get current time
    startTime = time.time()
    timeover = 0

    # search until there is no element in the priority queue or reach the goal
    while 1:
        broken = 0

        # update queue
        (newId, nextMoves) = findNextMove(data, currPoint, distFunc, costFunc, totalCost, Astar, Id, newId) # find next moves
        queue += nextMoves # append nextMoves in the priority queue
        tree += nextMoves # also in the tree
        if len(queue) == 0: break # break if the priority queue is empty
        queue.sort(key=lambda x:x[1]) # sort by cost

        # print queue
        if prt != 0:
            prtResult = ''
            for i in range(min(len(queue), 4)):
                prtResult += (str(queue[i][0]) + '(A*:' + nToS(queue[i][1], 3) + ', cost:' + nToS(queue[i][2], 3) + ') (id: ' + nToS(queue[i][3], 3) + '/' + nToS(queue[i][4], 3) + ')  ')
            if len(queue) > 10: prtResult += '...'
            print('queue: ' + prtResult)

        # break if reached the goal
        if queue[0][1] == queue[0][2]: broken = 1

        # if the point is in finishList, do not update and just pop
        while any (x == queue[0][0] for x in finishList):
            queue.pop(0)
            if len(queue) == 0: break

        # next move (to search)
        currPoint = queue[0][0] # update current point
        totalCost = queue[0][2] # update cost
        Id = queue[0][3] # update ID of current point
        preId = queue[0][4] # ID of predecessor of current point
        queue.pop(0) # pop best move from the priority queue

        # if there is no point that predecessor point is X (in the priority queue), append X in finishList
        preIdCount = 0
        for i in range(len(queue)):
            if queue[i][4] == preId: preIdCount += 1
        if preIdCount == 0: finishList.append(tree[preId][0])

        if broken == 1: break

        # time check
        currTime = time.time()
        if currTime - startTime > timeSec:
            timeover = 1
            break
        
    print('')

    # find the goal element
    goalId = 0 # ID of goal element (also index of goal element in the tree)
    if timeover == 0:
        for i in range(len(tree)):
            if tree[i][1] == tree[i][2]: # if A* result and cost is same -> distance is 0, goal
                goalId = i
                break
    else: # if timeover -> find element that has minimum distance
        minDist = tree[0][1]-tree[0][2]
        minDistId = 0
        for i in range(len(tree)):
            thisDist = tree[i][1] - tree[i][2]
            if thisDist < minDist:
                minDist = thisDist
                minDistId = i
        
        # consider minimum distance element as the goal
        goalId = minDistId
        print('time over : nearest goal is ' + str(tree[goalId]))

    # backtracking from goal element to start point
    point = tree[goalId][0]
    pointList = []
    currId = goalId # id of current element
    while 1:
        pointList.append(point)
        if currId == 0: # reached the start element
            pointList.reverse()
            break
        
        predId = tree[currId][4] # id of predecessor
        point = tree[predId][0] # go to predecessor point
        currId = tree[predId][3] # update ID to ID of predecessor

    print('list of point: ' + str(pointList))
    print('')

    # move
    for i in range(len(pointList)):
        point = pointList[i]
        moveFunc(data, point, i+1)
        print('')

if __name__ == '__main__':
    (data, prt, timeSec) = getData()
    AstarSearch([0, 1], data, distance_, cost_, next_, move_, Astar_, prt, timeSec)
