import math

def str_(a):
    if a < 10: return '  ' + str(a)
    elif a < 100: return ' ' + str(a)
    else: return str(a)

# average
def avg(array):
    if len(array) == 0: return 0
    return sum(array, 0.0) / len(array)

# standard deviation
def sd(array):
    if len(array) < 2: return 1

    average = avg(array)
    sum0 = 0 # sum of (xi-avg)^2
    for i in range(len(array)):
        sum0 += (array[i] - average)*(array[i] - average)
    return math.sqrt(sum0 / len(array))

# main: kNN Algorithm
def kNN():
    # get data
    file = open('kNN.txt', 'r')
    read = file.readlines()
    file.close

    data = [] # training data
    cols = 0 # number of columns (attribute and answer)
    for i in range(len(read)-2):
        row = read[i].split(' ')
        cols = len(row)
        row[len(row)-1] = row[len(row)-1].split('\n')[0]
        data.append(row)
    query = read[len(read)-2].split(' ') # query
    query[len(query)-1] = query[len(query)-1].split('\n')[0]
    k = int(read[len(read)-1].split(' ')[0]) # number of neighbors
    option = int(read[len(read)-1].split(' ')[1]) # option

    # get each attribute
    attribute_data = [] # [attribute][n] = data of attribute, index n
    for i in range(cols-1): # for each attribute
        temp = []
        for j in range(len(data)): # for each row
            temp.append(float(data[j][i]))
        attribute_data.append(temp)

    # get AVG and SD of each attribute
    avg_list = []
    sd_list = []
    for i in range(cols-1):
        avg_list.append(avg(attribute_data[i]))
        sd_list.append(sd(attribute_data[i]))

    # normalize and get distance
    distance = [] # for each item, 0=id and 1=distance (for each row)
    for i in range(len(data)): # for each row
        tempsum = 0
        for j in range(cols-1): # for each attribute
            data0 = (float(query[j]) - avg_list[j]) / sd_list[j] # normalization of query
            data1 = (float(data[i][j]) - avg_list[j]) / sd_list[j] # normalization of row i
            tempsum += (data0 - data1) * (data0 - data1)
        distance.append([i, math.sqrt(tempsum)]) # store the distance

    # get k nearest neightbors
    print('<Distance Data>')
    for i in range(len(distance)):
        print(str_(distance[i][0]) + ': ' + str(round(distance[i][1], 6)) + ' (answer = ' + str(data[distance[i][0]][cols-1]) + ')')
    distance.sort(key=lambda x:x[1])
    answer = [] # answers from k nearest neighbors
    print('')
    print('<' + str(k) + ' nearest neighbors>')
    for i in range(k):
        print(str_(distance[i][0]) + ': ' + str(round(distance[i][1], 6)) + ' (answer = ' + str(data[distance[i][0]][cols-1]) + ')')
        answer.append(data[distance[i][0]][cols-1])

    # get majority of answer(0: type, 1: score)
    majority = []
    for i in range(len(answer)):
        # check if the item already exists
        find_answer = 0
        index = -1 # index of item in majority
        for j in range(len(majority)):
            if majority[j][0] == answer[i]:
                find_answer = 1
                index = j
                break

        # if not exist in majority -> add
        if find_answer == 0:
            if option == 0: majority.append([answer[i], 1])
            elif abs(distance[i][1]) < 0.01: majority.append([answer[i], 100])
            else: majority.append([answer[i], 1/distance[i][1]])
        # if exist
        else:
            if option == 0: majority[index][1] = majority[index][1] + 1
            elif abs(distance[i][1]) < 0.01: majority.append([answer[i], 100])
            else: majority[index][1] = majority[index][1] + 1/distance[i][1]

    print('')
    print('<Average and SD>')
    x = 'avg: '
    for i in range(cols-1): x += str(round(avg_list[i], 6)) + ' | '
    print(x)
    x = 'sd : '
    for i in range(cols-1): x += str(round(sd_list[i], 6)) + ' | '
    print(x)
    print('')
    print('<Majority>')
    for i in range(len(majority)):
        print(str(majority[i][0]) + ': ' + str(round(majority[i][1], 6)))
    print('')
    majority.sort(key=lambda x:x[1])
    print('answer for query=' + str(query) + ' is ' + str(majority[len(majority)-1][0]))

kNN()
