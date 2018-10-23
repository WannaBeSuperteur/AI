import math

# Get count of each item (in array data, attribute No.coln)
def getCounts(data, coln):
    countlist = [] # [index][0] = item name, [index][1] = count
    for i in range(len(data)):
        # check if the item already exists
        find_item = 0
        index = -1 # index of item in countlist
        for j in range(len(countlist)):
            if countlist[j][0] == data[i][coln]: # if the item exists
                find_item = 1
                index = j
                break

        # if not exist in majority -> add
        if find_item == 0: countlist.append([data[i][coln], 1])
        # if exist
        else: countlist[index][1] = countlist[index][1] + 1

    return countlist

# Get Entropy
def getEntropy(countlist):
    valuelist = []
    total = 0
    for i in range(len(countlist)):
        valuelist.append(countlist[i][1])
        total += countlist[i][1]

    # evaluate entropy
    entropy = 0
    for i in range(len(valuelist)):
        p_i = countlist[i][1] / total
        entropy += (-1)*p_i*math.log(p_i, 2)
    return entropy

# Get Gini
def getGini(countlist):
    valuelist = []
    total = 0
    for i in range(len(countlist)):
        valuelist.append(countlist[i][1])
        total += countlist[i][1]

    # evaluate Gini
    Gini = 0
    for i in range(len(valuelist)):
        p_i = countlist[i][1] / total
        Gini += p_i*p_i
    return Gini
        
def Decision(option):
    decisiontree = []
    final_decisiontree = []
    filter_column = [] # consider if value of filter_column is filter_value
    filter_value = []
    
    # get data
    file = open('DecisionData.txt', 'r')
    read = file.readlines()
    file.close

    data = [] # training data
    cols = 0 # number of columns (attribute)
    for i in range(len(read)-1):
        row = read[i].split(' ')
        cols = len(row)
        row[len(row)-1] = row[len(row)-1].split('\n')[0]
        data.append(row)
    targetcol = int(read[len(read)-1]) # target attribute

    # making decision tree
    while 1:
        
        # filtering
        filtered_data = []
        for i in range(len(data)):
            # filter using filter_column and filter_value
            filtered = 0
            for j in range(len(filter_column)):
                if data[i][int(filter_column[j])] != filter_value[j]:
                    filtered = 1
                    break
            if filtered == 1: continue

            # get data
            temp = []
            for j in range(len(data[i])):
                temp.append(data[i][j])
            filtered_data.append(temp)

        # print filter
        toprint = []
        for i in range(len(filter_column)):
            toprint.append(filter_column[i])
            toprint.append(filter_value[i])
        print('')
        print('★filter: ' + str(toprint))

        # check if pure (leaf node)
        countdata = getCounts(filtered_data, targetcol)
        nonleaf = 1
        if len(countdata) == 1: # pure data -> leaf node
            print('▷Target Col is PURE:' + str(countdata[0]))
            print('▶PURE -> LEAF')
            print('')
            
            temp = []
            for i in range(len(filter_column)):
                temp.append(filter_column[i])
                temp.append(filter_value[i])
            temp += '*' # mark leaf node
            temp.append(countdata[0][0]) # value
            decisiontree.append(temp)
            final_decisiontree.append(temp)
            print('→tree:' + str(final_decisiontree))
            nonleaf = 0 # if leaf node, set nonleaf to 0

        # if leaf node, do not calculate entropy or Gini
        if nonleaf == 1:
            
            # option 0: using entropy
            # evaluate entropy of target attribute
            if option == 0:
                Sentropy = getEntropy(countdata) # Entropy(S)
                print('▷Target Col: ' + str(countdata))
                print('▷SEntropy  : ' + str(round(Sentropy, 4)))

                # evaluate entropy for each attribute
                gainlist = []
                for i in range(cols):
                    Gain = Sentropy
                    
                    # pass if the attribute was checked
                    if any(x == str(i) for x in filter_column) or i == targetcol: continue

                    # get classes of column i
                    class_of_coli = getCounts(filtered_data, i)

                    # evaluate entropy for each class of column i
                    for j in range(len(class_of_coli)):
                        coli_data = [] # targetcol data of each class of column i (eg. Gender-Male)
                        for x in range(len(filtered_data)):
                            if filtered_data[x][i] == class_of_coli[j][0]:
                                coli_data.append(filtered_data[x][targetcol])
                        coli_count = getCounts(coli_data, 0)
                        Gain -= (len(coli_data)*getEntropy(coli_count)/len(filtered_data))
                        print('<' + str(i) + '> Entropy of (' + class_of_coli[j][0] + '): ' + str(round(getEntropy(coli_count), 4)))

                    gainlist.append([i, Gain])
                    print('<' + str(i) + '> Gain          : ' + str(round(Gain, 4)))

            # option else: using Gini
            else:
                SGini = getGini(countdata) # Gini(S)
                print('SGini: ' + str(round(SGini, 4)))

                # evaluate Gini for each attribute
                gainlist = []
                for i in range(cols):
                    Gain = -SGini
                    
                    # pass if the attribute was checked
                    if any(x == str(i) for x in filter_column) or i == targetcol: continue

                    # get classes of column i
                    class_of_coli = getCounts(filtered_data, i)

                    # evaluate Gini for each class of column i
                    for j in range(len(class_of_coli)):
                        coli_data = [] # targetcol data of each class of column i (eg. Gender-Male)
                        for x in range(len(filtered_data)):
                            if filtered_data[x][i] == class_of_coli[j][0]: # filter(column i = each class)
                                coli_data.append(filtered_data[x][targetcol])
                        coli_count = getCounts(coli_data, 0)
                        Gain += (len(coli_data)*getGini(coli_count)/len(filtered_data)) # weight
                        print('<' + str(i) + '> Gini of (' + class_of_coli[j][0] + '): ' + str(round(getGini(coli_count), 4)))

                    gainlist.append([i, Gain])
                    print('<' + str(i) + '> Gain       : ' + str(round(Gain, 4)))

            # sort and get the largest Gain value
            gainlist.sort(key=lambda x:x[1])
            split_no = gainlist[len(gainlist)-1][0] # split by this attribute
            print('▷decide column to split: ' + str(split_no))
            print('▶SPLIT -> NONLEAF')
            print('')

            # make decision tree
            to_split = getCounts(filtered_data, split_no) # [index][0] = item name, [index][1] = count
            for i in range(len(to_split)):
                temp = []
                for j in range(len(filter_column)):
                    temp += filter_column[j]
                    temp += filter_value[j]
                temp += str(split_no) # column index
                temp.append(to_split[i][0]) # value
                decisiontree.append(temp)
                final_decisiontree.append(temp)
                print('→tree:' + str(final_decisiontree))

        # find next node of decision tree (BFS)
        # modify filter_column and filter_value
        while 1:
            
            # if decision tree is empty, return final decision tree
            if len(decisiontree) == 0:
                # only return leaf nodes
                result = []
                for i in range(len(final_decisiontree)):
                    if any(x == '*' for x in final_decisiontree[i]):
                        result.append(final_decisiontree[i])
                return result

            # if there are nodes in decision tree
            filter_column = []
            filter_value = []
            nextnode = decisiontree[0]
            leafnode = 0
            for i in range(int(len(nextnode)/2)):
                filter_column.append(nextnode[i*2])
                filter_value.append(nextnode[i*2+1])

                if nextnode[i*2] == '*': # if leaf node
                    leafnode = 1
                    break

            # find non-leaf node in decision tree -> if leaf node, pass
            decisiontree.pop(0)
            if leafnode == 0: break

def getAnswer(query, decisiontree):
    for i in range(1, len(query)): # deepening is iterative
        count = 0
        answer = ''
        # get answer from each item in decision tree
        for j in range(len(decisiontree)):

            # pass if not equal
            equal = 1
            for k in range(i):
                if query[int(decisiontree[j][2*k])] != decisiontree[j][2*k+1]:
                    equal = 0
                    break
            if equal == 0: continue

            # get answer if equal
            count += 1
            answer = decisiontree[j][len(decisiontree[j])-1]
            
        if count == 1: return answer

    return 'I can''t answer. T_T'

dt = Decision(0)
print('')
print('FINAL DECISION TREE')
print(dt)
print(getAnswer(['16', '15', '14', '13'], dt))
