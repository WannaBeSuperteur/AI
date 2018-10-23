import math
import random

# read data from file
def getData():
    # get data
    file = open('KMeans.txt', 'r')
    read = file.readlines()
    file.close

    data = [] # training data
    cols = 0 # number of columns (attribute and answer)
    for i in range(len(read)-2):
        row = read[i].split(' ')
        cols = len(row)
        row[len(row)-1] = row[len(row)-1].split('\n')[0]
        data.append(row)

    return data

# main: K-means Clustering
# K=num of clusters, I=max iterations, M=dimension
# Time Complexity=O(KIMN) where N is number of vectors
def KMeans(data, K, I, M):
    N = len(data)
    clusters = []
    for i in range(K): clusters.append([])

    # 0. get MAX and MIN value of each dimension
    maxs = []
    for i in range(M): maxs.append(-10000)
    mins = []
    for i in range(M): mins.append(10000)
    for i in range(N):
        for j in range(M):
            if int(data[i][j]) > maxs[j]: maxs[j] = int(data[i][j])
            if int(data[i][j]) < mins[j]: mins[j] = int(data[i][j])
    
    # 1. make K random initial centroids
    centroid = [[0]*M for i in range(K)]
    for i in range(K):
        for j in range(M):
            a = random.randint(0, maxs[j]-mins[j])
            centroid[i][j] = mins[j]+a

    # repeat 2 and 4 until 3 or I iterations
    for ii in range(I):
        # 2. assign vector to each cluster
        # Time: O(KNM) because O(KN) vector-centroid distance calculating
        #       O(M) for calculating each vector distance

        new_clusters = []
        for i in range(K): new_clusters.append([])

        # for each vector, calculate distance -> O(KNM)
        for i in range(N):
            min_distance = 10000
            closest_cluster = -1

            # for each cluster, calculate distance
            for j in range(K):
                # don't calculate if the cluster has no member
                if ii > 0 and len(clusters[j]) == 0: continue
                
                distance = 0
                for k in range(M):
                    distance += pow(centroid[j][k] - int(data[i][k]), 2)
                distance = math.sqrt(distance)

                if min_distance > distance:
                    closest_cluster = j
                    min_distance = distance

            # assign vector
            new_clusters[closest_cluster].append(i)

        # print result
        print('')
        print(str(ii+1) + ' iteration result')
        for i in range(K):
            if len(new_clusters[i]) == 0: continue
            centroid_int = []
            for j in range(len(centroid[i])):
                centroid_int.append(int(centroid[i][j]))
            print('last centroid of cluster ' + str(i) + ': ' + str(centroid_int))
        for i in range(K):
            for j in range(len(new_clusters[i])):
                print(str(new_clusters[i][j]) + '(' + str(i) + '): ' + str(data[new_clusters[i][j]]))
        
        # 3. if cluster not changed -> break
        if ii > 0:
            cluster_changed = 0
            for i in range(K):
                # if length is different -> changed
                if len(clusters[i]) != len(new_clusters[i]):
                    cluster_changed = 1
                    break
                # if member info is different -> changed
                for j in range(len(clusters[i])):
                    if clusters[i][j] != new_clusters[i][j]:
                        cluster_changed = 1
                        break
                if cluster_changed == 1: break
            if cluster_changed == 0: break
            
        # update clusters
        clusters = []
        for i in range(len(new_clusters)):
            clusters.append(new_clusters[i])

        # 4. re-calculate K centroids
        # Time: O(NM) - average of N data for M dimensions
        for i in range(K):
            # if no member in a cluster
            if len(clusters[i]) == 0: continue

            # calculate and reset centroid
            for j in range(M):
                centroid[i][j] = 0
            for j in range(len(clusters[i])):
                for k in range(M):
                    centroid[i][k] += int(data[clusters[i][j]][k])
            for j in range(M):
                centroid[i][j] /= len(clusters[i])
        
    return clusters

data = getData()
result = KMeans(data, 4, 50, 4)
