# calculate simple probability
def simpleProb(iArr, graph, probs, pr):
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
            
    if pr != 0: print('iArr: ' + str(iArr))

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
            prob += simpleProb(iArrCopy, graph, probs, pr)

        return prob

# calculate conditional probability
def condiProb(iArr0, iArr1, graph, probs, pr):
    iArr2 = iArr0 + iArr1 # merge
    # P(A|B) = P(A, B)/P(B)
    return simpleProb(iArr2, graph, probs, pr) / simpleProb(iArr1, graph, probs, pr)
