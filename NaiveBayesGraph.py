import math
import NaiveBayesPackage as NBP

# [string array] -> [number array]
def toNumber(arr):
    for j in range(len(arr)):
        arr[j] = float(arr[j])
        if arr[j] % 1.0 == 0.0: arr[j] = int(arr[j])
    return arr

# read data from file
def getData():
    # get data
    file = open('NaiveBayesGraph.txt', 'r')
    read = file.readlines()
    file.close

    graph = [[]] # for index 0
    probs = [[]] # for index 0
    query = []
    
    # first N lines -> read probabilities
    for i in range(int(read[0])):
        row0 = read[i+1].split('\n')[0].split('/')[0].split(' ')
        row0 = toNumber(row0)
        graph.append(row0)
        row1 = read[i+1].split('\n')[0].split('/')[1].split(' ')
        row1 = toNumber(row1)
        probs.append(row1)

    # next lines until the end -> read simple prob or cond prob
    for i in range(len(read)-int(read[0])-1):
        row = read[i+int(read[0])+1].split('\n')[0].split(' ')
        query.append(row)

    print('Graph: ' + str(graph))
    return (graph, probs, query)

(graph, probs, query) = getData()

# answer each query
for i in range(len(query)):
    queryCopy = []
    for j in range(len(query[i])): queryCopy.append(query[i][j])
    print('')
    print('Query ' + str(i) + ': ' + str(query[i]))
    
    # simple probability
    # P(x) = P(y)P(x|y)+P(~y)P(x|~y) = P(0)P(y|0)P(x|y)+P(~0)P(~y|~0)P(x|~y) (if 0-y-x tree)
    if len(query[i]) == 1:
        prob = NBP.simpleProb(toNumber(query[i][0].split('^')), graph, probs, 1)
        print('Query ' + str(queryCopy) + ' -> ' + str(round(prob, 9)))

    # conditional probability
    # P(y|x) = P(x^y)/P(x) = P(0)P(x^y|0)/P(0)P(x) = P(0)P(x|0)P(y|0)/P(x) (0-x, 0-y tree)
    elif len(query[i]) == 2:
        arr0 = toNumber(query[i][0].split('^')) # left of |
        arr1 = toNumber(query[i][1].split('^')) # right of |
        prob = NBP.condiProb(arr0, arr1, graph, probs, 1)
        print('Query ' + str(queryCopy) + ' -> ' + str(round(prob, 9)))
