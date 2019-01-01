import math

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

# calculate simple probability
def simpleProb(iArr, graph, probs):
    pre = 0 # are there predecessors?
    additional = [] # additional nodes
    
    # 0. add all predecessors
    i = 0
    while i < len(iArr):
        node = abs(int(iArr[i]))
        
        # add predecessors of the node
        for j in range(1, len(graph[node])):
            toAppend = graph[node][j]
            if not any(x == toAppend for x in iArr) and not any(x == -toAppend for x in iArr):
                iArr.append(toAppend)
                additional.append(toAppend)
                pre = 1
        i += 1
            
    print('iArr: ' + str(iArr))

    # 1. calculate probability
    # there is no additional predecessor
    if pre == 0:
        # there is only one node
        if len(iArr) == 1:
            if iArr[0] >= 0: return probs[iArr[0]][0]
            else: return 1-probs[-iArr[0]][0]

        # there are 2 or more nodes
        else:
            prob = 1.0
            for i in range(len(iArr)):
                index = iArr[i]
                pres = len(graph[abs(index)])-1 # number of predecessors
                probIndex = 0 # index of array probs

                # modify probIndex if predecessor is FALSE
                for j in range(1, pres+1):
                    if not any(x == graph[abs(index)][j] for x in iArr):
                        probIndex += pow(2, pres-j)

                # multiply probability
                if index >= 0: prob *= probs[index][probIndex]
                else: prob *= 1-probs[-index][probIndex]

            return prob

    # there are 1 or more additional predecessors
    else:
        prob = 0.0
        addStart = len(iArr)-len(additional) # start index of additions in iArr
        
        for i in range(pow(2, len(additional))):          
            # make copy of iArr
            iArrCopy = []
            for j in range(len(iArr)): iArrCopy.append(iArr[j])

            # modify iArrCopy
            for j in range(addStart, len(iArr)):
                if int(i / pow(2, len(iArr)-1-j)) % 2 == 1: iArrCopy[j] *= -1

            # add the case of probability
            prob += simpleProb(iArrCopy, graph, probs)

        return prob

# calculate conditional probability
def condiProb(iArr0, iArr1, graph, probs):
    iArr2 = iArr0 + iArr1 # merge
    # P(A|B) = P(A, B)/P(B)
    return simpleProb(iArr2, graph, probs) / simpleProb(iArr1, graph, probs)

# answer each query
for i in range(len(query)):
    queryCopy = []
    for j in range(len(query[i])): queryCopy.append(query[i][j])
    print('')
    print('Query ' + str(i) + ': ' + str(query[i]))
    
    # simple probability
    # P(x) = P(y)P(x|y)+P(~y)P(x|~y) = P(0)P(y|0)P(x|y)+P(~0)P(~y|~0)P(x|~y) (if 0-y-x tree)
    if len(query[i]) == 1:
        prob = simpleProb(toNumber(query[i][0].split('^')), graph, probs)
        print('Query ' + str(queryCopy) + ' -> ' + str(round(prob, 9)))

    # conditional probability
    # P(y|x) = P(x^y)/P(x) = P(0)P(x^y|0)/P(0)P(x) = P(0)P(x|0)P(y|0)/P(x) (0-x, 0-y tree)
    elif len(query[i]) == 2:
        arr0 = toNumber(query[i][0].split('^')) # left of |
        arr1 = toNumber(query[i][1].split('^')) # right of |
        prob = condiProb(arr0, arr1, graph, probs)
        print('Query ' + str(queryCopy) + ' -> ' + str(round(prob, 9)))
