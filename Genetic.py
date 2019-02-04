import math
import random

# read data from file
def getData():
    # get data
    file = open('Genetic.txt', 'r')
    read = file.readlines()
    file.close

    # read configuration data
    config = read[0].split('\n')[0].split(' ')
    numArray = int(config[0])
    arrayLen = int(config[1])
    p = float(config[2])
    iters = int(config[3])
    prt = int(config[4])

    # read maze data
    data = []
    for i in range(1, len(read)):
        row = read[i].split('\n')[0].split(' ')
        data.append(row)
        
    return (data, numArray, arrayLen, p, iters, prt)

# evaluation function
def evaluation(data, array, etc):
    steps = 0
    height = len(data)
    width = len(data[0])
    point = [0, 1]
    goal = [height-1, width-2]

    # search
    while point != goal and steps < len(array):
        # possible next points (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        possible = []

        # UP
        if point[0] >= 1:
            if data[point[0]-1][point[1]] == '.': possible.append(1)
            else: possible.append(0)
        else: possible.append(0)

        # DOWN
        if point[0] < height-1:
            if data[point[0]+1][point[1]] == '.': possible.append(1)
            else: possible.append(0)
        else: possible.append(0)

        # LEFT
        if point[1] >= 1:
            if data[point[0]][point[1]-1] == '.': possible.append(1)
            else: possible.append(0)
        else: possible.append(0)

        # RIGHT
        if point[1] < width-1:
            if data[point[0]][point[1]+1] == '.': possible.append(1)
            else: possible.append(0)
        else: possible.append(0)
        
        # decide next point in accordance with each element in the array
        # if possible
        if possible[array[steps]] == 1:
            if array[steps] == 0: point = [point[0]-1, point[1]] # UP
            elif array[steps] == 1: point = [point[0]+1, point[1]] # DOWN
            elif array[steps] == 2: point = [point[0], point[1]-1] # LEFT
            elif array[steps] == 3: point = [point[0], point[1]+1] # RIGHT

        # if impossible, return manhattan distance between current point and goal
        else: return -(abs(point[0]-goal[0])+abs(point[1]-goal[1]))

        steps += 1

    # return manhattan distance between current point and goal
    return -(abs(point[0]-goal[0])+abs(point[1]-goal[1]))

# perform genetic algorithm
def Genetic(data, evalFunc, numArray, arrayLen, p, valueList, iters, prt):

    # 1. random initialize
    arraies = []
    bestmix = []
    for i in range(numArray):
        temp = []
        for j in range(arrayLen):
            temp.append(valueList[random.randint(0, len(valueList)-1)])
        arraies.append(temp)

    # iteration
    for eachIter in range(iters):
        if prt >= 1: print('< iteration ' + str(eachIter) + ' >')
        
        # 2. evaluate array
        evals = []
        for i in range(numArray):
            evals.append(evalFunc(data, arraies[i], []))

        # 3. print array
        if prt >= 1:
            for i in range(numArray):
                print(str(arraies[i]) + ' -> eval: ' + str(evals[i]))

        # 4. find the best two and mix them: call 'bestmix'
        # find the best two arraies
        indexArray = []
        for i in range(numArray):
            indexArray.append([i, evals[i]])
        indexArray.sort(key=lambda x:x[1])

        best0Index = indexArray[numArray-1][0] # index of 1st best array
        best1Index = indexArray[numArray-2][0] # index of 2nd best array

        # mix them
        best0 = []
        best1 = []
        for i in range(int(arrayLen/2)): best0.append(arraies[best0Index][i])
        for i in range(int(arrayLen/2), arrayLen): best1.append(arraies[best1Index][i])

        # make 'bestmix': half best0, half best1
        bestmix = best0 + best1

        # 5. update arraies: add random noise to 'bestmix'
        for i in range(numArray):
            for j in range(arrayLen):
                if random.random() >= p: arraies[i][j] = bestmix[j]
                else: arraies[i][j] = valueList[random.randint(0, len(valueList)-1)] # for probability p, make a mutation

        if prt >= 1:
            print('bestmix: ' + str(bestmix[0:50]) + ' -> eval: ' + str(evalFunc(data, bestmix, [])))
            print('')
        elif prt == 0:
            print('iter ' + str(eachIter) + ' bestmix: ' + str(bestmix[0:50]) + ' -> eval: ' + str(evalFunc(data, bestmix, [])))

    if prt == 0: print('')
    print('FINAL bestmix: ' + str(bestmix) + '-> eval: ' + str(evalFunc(data, bestmix, [])))
    return bestmix

if __name__ == '__main__':
    (data, numArray, arrayLen, p, iters, prt) = getData()
    Genetic(data, evaluation, numArray, arrayLen, p, [0, 1, 2, 3], iters, prt)
