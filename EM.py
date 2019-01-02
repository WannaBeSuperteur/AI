import math

# read data from file
def getData():
    # get data
    file = open('EM.txt', 'r')
    read = file.readlines()
    file.close

    data = [] # training data
    cols = 0 # number of columns
    for i in range(len(read)):
        row = read[i].split(' ')
        cols = len(row)
        row[len(row)-1] = row[len(row)-1].split('\n')[0]
        data.append(row)

    # make initialized matrix
    x = pow(2, cols)
    matrix = []
    for i in range(x): matrix.append(1/x)

    return (data, cols, matrix)

#except for not corresponding elements of matrix
def exceptNotCorresp(index, data, cols, j):
    test = 1
    for m in range(cols):
        value = int(index/pow(2, cols-1-m)) % 2
        if data[j][m] == '0' and value == 1:
            test = 0
            break
        if data[j][m] == '1' and value == 0:
            test = 0
            break
    return test

#EM algorithm
def EM(data, cols, matrix, n):

    # print initial data
    print('')
    print('init data  : ')
    for j in range(len(data)): print(data[j])

    # store original data for iteration
    oData = []
    for j in range(len(data)):
        row = []
        for k in range(len(data[j])):
            row.append(data[j][k])
        oData.append(row)
    
    # iterate: n times
    for i in range(n):
        leng = len(data) # number of rows
        
        # store current matrix
        mStore = []
        for j in range(len(matrix)): mStore.append(matrix[j])

        # all element of matrix -> 0
        for j in range(len(matrix)): matrix[j] = 0

        # calculate hidden and update matrix
        for j in range(leng):
            
            # if no hidden variable
            if not any(x == 'H' for x in data[j]):
                index = 0
                for k in range(cols):
                    index += int(data[j][k]) * pow(2, cols-1-k)
                matrix[index] += 1/leng
            
            # if hidden
            else:
                # get probability of hidden variables
                for k in range(cols):
                    if data[j][k] == 'H':                     

                        # get probability
                        rate_of_0 = 0.0
                        rate_of_1 = 0.0
                        for index in range(len(matrix)):
                            # except for not corresponding elements of matrix
                            test = exceptNotCorresp(index, data, cols, j)
                            if test == 0: continue
                            
                            # add rate of 0 and 1
                            # val: value of column k correspond to index(0 or 1)
                            val = int(index/pow(2, cols-1-k)) % 2
                            if val == 0: rate_of_0 += mStore[index]
                            else: rate_of_1 += mStore[index]

                        # update data
                        data[j][k] = rate_of_1/(rate_of_0+rate_of_1)

                # update matrix
                for index in range(len(matrix)):
                    # except for not corresponding elements of matrix
                    test = exceptNotCorresp(index, data, cols, j)
                    if test == 0: continue

                    addVal = 1/leng # value to add to matrix
                    for k in range(cols):
                        # val: value of column k correspond to index(0 or 1)
                        val = int(index/pow(2, cols-1-k)) % 2
                        if val == 1: addVal *= float(data[j][k])
                        else: addVal *= 1-float(data[j][k])
                    matrix[index] += addVal

        # print result
        print('')
        print('ROUND ' + str(i+1))
        print('curr data  : ')
        for j in range(len(data)):
            for k in range(len(data[j])):
                data[j][k] = round(float(data[j][k]), 6)
            print(data[j])
        for j in range(len(matrix)): matrix[j] = round(matrix[j], 6)
        print('prev matrix: ' + str(mStore))
        print('curr matrix: ' + str(matrix))

        # update mStore and data(data: reset to original)
        for j in range(len(mStore)): mStore[j] = matrix[j]
        for j in range(len(oData)):
            for k in range(len(oData[j])):
                data[j][k] = oData[j][k]
            
    return matrix

(data, cols, matrix) = getData()

EM(data, cols, matrix, 8)
