import math
import random

# read data from file
def getData():
    # get data
    file = open('Hierarchical.txt', 'r')
    read = file.readlines()
    file.close

    data = [] # training data
    cols = 0 # number of columns (attribute and answer)
    for i in range(len(read)):
        row = read[i].split(' ')
        cols = len(row)
        row[len(row)-1] = row[len(row)-1].split('\n')[0]
        data.append(row)

    return data

# main: Hierarchical Clustering
def Hierarchical(data, K, M):
    N = len(data)
    clusters = []
    for i in range(N): clusters.append([i])

    # 1. make matrix
    matrix = [[0]*N for i in range(N)]
    for i in range(N):
        for j in range(N):
            distance = 0
            for k in range(M):
                distance += pow(int(data[i][k]) - int(data[j][k]), 2)
            distance = math.sqrt(distance)
            matrix[i][j] = distance

    # merge until K clusters
    while 1:

        # print matrix
        print('clusters: ' + str(clusters))
        print('matrix : distance of cluster')
        NC = len(clusters) # number of clusters
        matrix2 = [[0]*NC for i in range(NC)]
        for i in range(NC):
            for j in range(NC):
                matrix2[i][j] = int(matrix[i][j])
        for i in range(NC):
            temp = ''
            for j in range(NC):
                value = matrix2[i][j]
                if value < 10: temp += ' ' + str(value) + ' '
                else: temp += str(value) + ' '
            print('| ' + temp + '|')
        
        # 2. find the max similar(min distance) index (x, y)
        mindist = 99999
        merge1 = -1 # to merge
        merge2 = -1 # to merge
        
        for i in range(NC):
            for j in range(NC):
                if i == j: continue
                if matrix[i][j] < mindist:
                    mindist = matrix[i][j]
                    merge1 = i
                    merge2 = j

        print('to merge: ' + str(clusters[merge1]) + ', ' + str(clusters[merge2]))
        print('')

        # 3. make temporary matrix to calculate new similarity(distance)
        tm = [[0]*NC for i in range(NC)]
        for i in range(NC):
            for j in range(NC):
                tm[i][j] = matrix[i][j]

        # 4. merge cluster x, y
        new_cluster = []
        for i in range(len(clusters[merge1])):
            new_cluster.append(clusters[merge1][i])
        for i in range(len(clusters[merge2])):
            new_cluster.append(clusters[merge2][i])
        
        if merge1 > merge2: # x > y
            matrix.pop(merge1) # 1. remove x-th row of matrix
            matrix.pop(merge2) # 2. remove y-th row of matrix
            for i in range(len(matrix)):
                matrix[i].pop(merge1) # 3. remove x-th column of matrix
                matrix[i].pop(merge2) # 4. remove y-th column of matrix
            clusters.pop(merge1) # 5. remove x-th member of cluster
            clusters.pop(merge2) # 6. remove y-th member of cluster
        else: # x <= y
            matrix.pop(merge2) # 1. remove y-th row of matrix
            matrix.pop(merge1) # 2. remove x-th row of matrix
            for i in range(len(matrix)):
                matrix[i].pop(merge2) # 3. remove y-th column of matrix
                matrix[i].pop(merge1) # 4. remove x-th column of matrix
            clusters.pop(merge2) # 5. remove y-th member of cluster
            clusters.pop(merge1) # 6. remove x-th member of cluster

        # update matrix for merged cluster
        clusters.append(new_cluster)
        for i in range(NC-2): matrix[i].append(0)
        new_row = []
        for i in range(NC-1): new_row.append(0)
        matrix.append(new_row)
        
        NC -= 1

        # 5. if K clusters -> break
        if NC == K: break

        # 6. calculate sim(AB, X) = max(sim(A, X), sim(B, X)) for each member X
        #    -> using distance(AB, X) = min(distance(A, X), distance(B, X))
        #    -> update matrix
        last_i = 0
        for i in range(NC-1):
            if last_i == merge1: last_i += 1
            if last_i == merge2: last_i += 1
            mindist = min(tm[merge1][last_i], tm[merge2][last_i])
            matrix[NC-1][i] = mindist
            matrix[i][NC-1] = mindist
            last_i += 1
        
    return clusters

data = getData()
result = Hierarchical(data, 1, 2)
print(result)
