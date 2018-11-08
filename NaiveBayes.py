import math

# read data from file
def getData():
    # get data
    file = open('NaiveBayes.txt', 'r')
    read = file.readlines()
    file.close

    data = [] # training data
    cols = 0 # number of columns (attribute and answer)
    for i in range(len(read)-1):
        row = read[i].split(' ')
        cols = len(row)
        row[len(row)-1] = row[len(row)-1].split('\n')[0]
        data.append(row)

    query = read[len(read)-1].split(' ')

    return (data, cols, query)

# get data
(data, cols, query) = getData()
cols -= 1 # except for 1st column

# get probability
total_Y = 0
prob_Y = 0
total_X_Y = [] # number of X where Y is TRUE
total_X_notY = [] # number of X where Y is FALSE
for i in range(cols):
    total_X_Y.append(0.0)
    total_X_notY.append(0.0)
    
prob_X = [[0.0]*2 for i in range(cols)]
for i in range(len(data)):
    if data[i][0] == '1':
        total_Y += 1
        for j in range(cols):
            if data[i][j+1] == '1': total_X_Y[j] += 1
    else:
        for j in range(cols):
            if data[i][j+1] == '1': total_X_notY[j] += 1

prob_Y = total_Y / len(data)
for i in range(cols):
    prob_X[i][0] = total_X_Y[i] / total_Y # calculate P(Xi|Y)
    prob_X[i][1] = total_X_notY[i] / (len(data)-total_Y) # calculate P(Xi|~Y)
print('P(Y) = ' + str(prob_Y))
for i in range(cols):
    print('P(X' + str(i) + '|Y) = ' + str(prob_X[i][0]) + ', P(X' + str(i) + '|~Y) = ' + str(prob_X[i][1]))

# prediction
print('query: ' + str(query))
prob1 = prob_Y # P(Y)*P(X1|Y)*P(X2|Y)*...*P(Xn|Y)
prob2 = 1-prob_Y # P(~Y)*P(X1|~Y)*P(X2|~Y)*...*P(Xn|~Y)
for i in range(cols):
    if query[i] == '1':
        prob1 *= prob_X[i][0] # P(Xi|Y)
        prob2 *= prob_X[i][1] # P(Xi|~Y)
    else:
        prob1 *= 1-prob_X[i][0] # P(~Xi|Y)
        prob2 *= 1-prob_X[i][1] # P(~Xi|~Y)
print('Probability for Y: ' + str(prob1/(prob1+prob2)))
print('Probability for not Y: ' + str(prob2/(prob1+prob2)))
if prob1 > prob2: print ('result: Y')
else: print('result: not Y')
