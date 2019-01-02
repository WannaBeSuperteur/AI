import math
import random
import NaiveBayesPackage as NBP

# read data from file
def getData():
    # get data
    file = open('EMNaiveBayes.txt', 'r')
    read = file.readlines()
    file.close

    data = [] # training data
    count = [] # count data
    cols = 0 # number of columns
    for i in range(len(read)):
        row = read[i].split(' ')
        cols = len(row)-1
        row[cols] = row[cols].split('\n')[0]

        # append
        count.append(row[cols])
        row.pop(cols)
        data.append(row)
        
    return (data, count, cols)

# make random value between 0 and 1
def makeRanVal(times):
    value = 0
    for i in range(times):
        test = random.randint(0, 1)
        if test == 1: value += 1/times
    return round(value, 4)

# print
def pr(prob, probH, probNotH, prob0):
    prob_ = round(prob, 4)
    probH_ = []
    probNotH_ = []
    prob0_ = []
    for i in range(len(probH)): probH_.append(round(probH[i], 6))
    for i in range(len(probNotH)): probNotH_.append(round(probNotH[i], 6))
    for i in range(len(prob0)): prob0_.append(round(prob0[i], 6))
    
    print('P(H) = ' + str(prob_))
    print('P(X|H) = ' + str(probH_))
    print('P(~X|H) = ' + str(probNotH_))
    print('P(H|A=a, B=b, ...) = ' + str(prob0_))

# EM algorithm (with Naive Bayes)
def EMwithNB(data, count, cols, n):
    
    prob = makeRanVal(100) # P(H)
    probH = [] # P(A|H), P(B|H), ...
    probNotH = [] # P(A|~H), P(B|~H), ...
    prob0 = [] # P(H|A=a, B=b, ...)
    for i in range(len(data)): prob0.append(0)
    sumCount = 0 # sum of count
    for i in range(len(count)): sumCount += int(count[i])

    # initialize probabilities
    for i in range(cols):
        probH.append(makeRanVal(100))
        probNotH.append(makeRanVal(100))

    print('INIT: ')
    pr(prob, probH, probNotH, prob0)

    # to calculate probabilities
    graph = [[], [1]]
    for i in range(cols):
        graph.append([i+2, 1])

    print(graph)

    # iterate: n times
    for i in range(n):
        # update probs
        probs = [[], [prob]]
        for j in range(cols):
            probs.append([probH[j], probNotH[j]])
        
        # calculate probabilities P(H|A=a, B=b, ...)
        # using prob, probH and probNotH
        for j in range(len(data)):
            arr0 = [1]
            arr1 = []
            for k in range(cols):
                if data[j][k] == '1': arr1.append(k+2)
                else: arr1.append(-k-2)
            prob0[j] = NBP.condiProb(arr0, arr1, graph, probs, 0)

        # update prob, probH and probNotH using data
        # P(H)
        Sum = 0
        for j in range(len(data)):
            Sum += float(count[j]) * float(prob0[j])
        prob = Sum/sumCount

        # P(X|H) and P(X|~H)
        # using P(X|H) = P(X, H)/P(H) -> calculate P(X, H)
        for j in range(cols):
            Sum1 = 0 # Sum of X=1 and H=1
            Sum0 = 0 # Sum of X=1 and H=0
            for k in range(len(data)):
                if data[k][j] == '1':
                    Sum1 += float(count[k]) * float(prob0[k])
                    Sum0 += float(count[k]) * (1-float(prob0[k]))
            PXH = Sum1/sumCount # P(X, H)
            PNXH = Sum0/sumCount # P(~X, H)
            probH[j] = PXH/prob
            probNotH[j] = PNXH/(1-prob)

        # print
        print('')
        print('ROUND ' + str(i+1))
        pr(prob, probH, probNotH, prob0)

(data, count, cols) = getData()

EMwithNB(data, count, cols, 50)
