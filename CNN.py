import math
import random

# read data from file
def getData():
    # get data
    file = open('CNN.txt', 'r')
    read = file.readlines()
    file.close

    info = read[0].split('\n')[0].split(' ')
    rows = int(info[0]) # number of rows of each image
    cols = int(info[1]) # number of columns of each image
    images = int(info[2]) # number of images
    filtersize = int(info[3]) # number of rows and columns of each filter
    filters = int(info[4]) # number of filters
    prt = int(info[5]) # print?

    # data -> imgdata -> symbol and (content -> each row -> each pixel)
    data = []

    # read input and output
    for i in range(0, images):
        imgdata = [] # data of each image

        # read symbol of each image
        symbol = read[i*(rows+1)+1].split('\n')[0]
        imgdata.append(symbol)

        # read content of each image
        content = []
        for j in range(0, rows):
            content.append(read[i*(rows+1)+j+2].split('\n')[0])
        imgdata.append(content)

        data.append(imgdata)

    # read test image
    testimgdata = []
    
    symbol = read[images*(rows+1)+1].split('\n')[0] # symbol of test image
    testimgdata.append(symbol)
    
    testimgcontent = [] # content of test image
    for i in range(0, rows):
        testimgcontent.append(read[images*(rows+1)+i+2].split('\n')[0])
    testimgdata.append(testimgcontent)

    return (data, rows, cols, images, filtersize, filters, testimgdata, prt)

# . or # to 0 or 1
def toVal(value):
    if value == '.': return -1
    elif value == '#': return 1
    else: return value

# convolution
def convolutionImageFilter(data, rows, cols, img, i, filtersize, flts):
    temp = []
    for j in range(rows-filtersize+1):
        temp_ = []
        for k in range(cols-filtersize+1):

            # unit: each element of convolution result
            sumProduct = 0
            for l in range(filtersize):
                for m in range(filtersize):
                    # using Sumproduct = Sum(ImageData * Filter)
                    sumProduct += round(toVal(data[img][1][j+l][k+m]) * toVal(flts[i][l][m]), 2)
            elements = filtersize*filtersize # number of elements in each filter
            temp_.append(sumProduct/elements)
                    
        temp.append(temp_)
    return temp

# convolution -> return convolution result array
def convolution(data, rows, cols, images, filtersize, filters, flts, prt):
    convol = [] # result of convolution 0
    for img in range(images):
        convolOfImg = [] # convolution result of each image
        if prt != 0: print('** Name of image: ' + str(data[img][0]) + ' **')
        convolOfImg.append(data[img][0])

        # unit: each image
        imgData = []
        for i in range(filters):
            if prt != 0: print('< Filter ' + str(i) + ' >')

            # unit: each filter
            temp = convolutionImageFilter(data, rows, cols, img, i, filtersize, flts)
            imgData.append(temp)

            # print result
            if prt != 0:
                for j in range(len(temp)):
                    printStr = '['
                    for k in range(len(temp[0])):
                        printStr += (' ' + str(round(temp[j][k], 2)) + ' ')
                    print(str(printStr) + ' ]')
                print('')
            
        convolOfImg.append(imgData)
        convol.append(convolOfImg)

    return convol

# ReLU of convolution result -> return ReLU result
def ReLU(convol, prt):

    # len(convol) = number of images
    # len(convol[0][1]) = number of filters
    # len(convol[0][1][0]) = number of rows in convolution result
    # len(convol[0][1][0][0]) = number of columns in convolution result
    
    ReLU = [] # result of ReLU
    for img in range(len(convol)):
        ReLUOfImg = [] # ReLU result of each image
        if prt != 0: print('** Name of image: ' + str(convol[img][0]) + ' **')
        ReLUOfImg.append(convol[img][0])

        # unit: each image
        imgData = []
        for i in range(len(convol[0][1])):
            if prt != 0: print('< Filter ' + str(i) + ' >')

            # unit: each filter
            temp = []
            for j in range(len(convol[0][1][0])):
                temp_ = []
                for k in range(len(convol[0][1][0][0])):

                    # ReLU using max(value, 0) function
                    temp_.append(max(convol[img][1][i][j][k], 0))
                    
                temp.append(temp_)
            imgData.append(temp)

            # print result
            if prt != 0:
                for j in range(len(temp)):
                    printStr = '['
                    for k in range(len(temp[0])):
                        printStr += (' ' + str(round(temp[j][k], 2)) + ' ')
                    print(str(printStr) + ' ]')
                print('')
        ReLUOfImg.append(imgData)
            
        ReLU.append(ReLUOfImg)
        
    return ReLU

# Pooling of ReLU result -> return Pooling result
def pooling(ReLU, poolFiltSize, stride, option, prt):

    # len(ReLU) = number of images
    # len(ReLU[0][1]) = number of filters
    # len(ReLU[0][1][0]) = number of rows in ReLU result
    # len(ReLU[0][1][0][0]) = number of columns in ReLU result
    # stride : stride of pooling filter
    # option : 0=max, other=average
    # poolFiltSize : number of rows and columns in pooling filter (same)

    Pooling = [] # result of pooling
    for img in range(len(ReLU)):
        PoolingOfImg = [] # pooling result of each image
        if prt != 0: print('** Name of image: ' + str(ReLU[img][0]) + ' **')
        PoolingOfImg.append(ReLU[img][0])

        # unit: each image
        poolingData = []
        for i in range(len(ReLU[0][1])):

            temp = [] # will store result of pooling
            rown = 0 # row number of top left cells of pooling filter
            coln = 0 # column number of top left cells of pooling filter
            
            while rown < len(ReLU[0][1][0]):
                temp_ = []
                coln = 0
                
                while coln < len(ReLU[0][1][0][0]):

                    # decide value of each element in pooling result
                    poolingValue = 0
                    for j in range(poolFiltSize):

                        # if out of range, break
                        if rown+j >= len(ReLU[0][1][0]): break
                        
                        for k in range(poolFiltSize):

                            # if out of range, break
                            if coln+k >= len(ReLU[0][1][0]): break
                            
                            # max pooling
                            if option == 0:
                                if poolingValue < ReLU[img][1][i][rown+j][coln+k]:
                                    poolingValue = ReLU[img][1][i][rown+j][coln+k]

                            # average pooling
                            else:
                                poolingValue += ReLU[img][1][i][rown+j][coln+k]

                    # average pooling
                    if option != 0: poolingValue /= (poolFiltSize*poolFiltSize)
                    temp_.append(poolingValue)
                    coln += stride
                    
                temp.append(temp_)
                rown += stride

            poolingData.append(temp)
                
            # print result
            if prt != 0:
                for j in range(len(temp)):
                    printStr = '['
                    for k in range(len(temp[0])):
                        printStr += (' ' + str(round(temp[j][k], 2)) + ' ')
                    print(str(printStr) + ' ]')
                print('')
        PoolingOfImg.append(poolingData)
            
        Pooling.append(PoolingOfImg)
        
    return Pooling

# make filters for CNN
def makeFilters(data, rows, cols, images, filtersize, filters, prt):
    # generate filters
    flts = [] # filter array
    i = 0
    while i < filters:
        temp = []
        imgn = random.randint(0, images-1) # decide image
        rown = random.randint(0, rows-filtersize) # decide row
        coln = random.randint(0, cols-filtersize) # decide column

        count = 0 # count of # (must be at least 5)

        # make filter for each image
        for j in range(filtersize):
            temp_ = []
            for k in range(filtersize):
                temp_.append(data[imgn][1][rown+j][coln+k])
                if data[imgn][1][rown+j][coln+k] == '#': count += 1
            temp.append(temp_)

        # if count is less than half of elements
        if count < filtersize*filtersize/2: continue
        
        flts.append(temp)
        i += 1

    # print filters
    print('******** FILTERS ********')
    for i in range(filters):
        print('Filter ' + str(i) + ':')
        for j in range(filtersize):
            print(flts[i][j])
        print('')

    return flts

# CNN
def CNN(data, rows, cols, images, filtersize, filters, flts, prt):

    # data : image data
    # rows : number of rows in each image
    # cols : number of columns in each image
    # filtersize : number of rows and columns in each filter (same)
    # filters : number of filters
    # testimgdata : image for testing CNN (each test image)

    # convolution 0
    if prt != 0: print('******** CONVOLUTION 0 RESULT ********')
    convol0 = convolution(data, rows, cols, images, filtersize, filters, flts, prt)

    # ReLU 0
    if prt != 0: 
        print('')
        print('******** RELU 0 RESULT ********')
    ReLU0 = ReLU(convol0, prt)

    # pooling 0
    # filter size = 2, stride = 2 and max pooling(option=0)
    if prt != 0: 
        print('')
        print('******** POOLING 0 RESULT ********')
    pooling0 = pooling(ReLU0, 2, 2, 0, prt)
    
    # convolution 1
    # convolution of pooling 0 result, using only flts[0] filter
    if prt != 0: 
        print('')
        print('******** CONVOLUTION 1 RESULT ********')

    # make 'pooling0' available for 'convolution' function
    pooling0Data = []
    for i in range(images):
        for j in range(filters):
            temp = []
            temp.append(str(pooling0[i][0]) + '-filter-' + str(j))
            temp.append(pooling0[i][1][j])
            pooling0Data.append(temp)

    # using only filter 0 when doing 'convolution 1'
    newFilter = [[[1, -1], [-1, 1]]]
    convol1 = convolution(pooling0Data, math.floor((rows-1)/2), math.floor((cols-1)/2), images*filters, 2, 1, newFilter, prt)

    # ReLU 1
    if prt != 0:
        print('')
        print('******** RELU 1 RESULT ********')
    ReLU1 = ReLU(convol1, prt)

    # pooling 1
    # filter size = 2, stride = 2 and max pooling(option=0)
    if prt != 0:
        print('')
        print('******** POOLING 1 RESULT ********')
    pooling1 = pooling(ReLU1, 2, 2, 0, prt)

    # result array
    result = []
    for i in range(images):
        temp = []
        for j in range(i*filters, (i+1)*filters):
            for k in range(len(pooling1[0][1][0])):
                for l in range(len(pooling1[0][1][0][0])):
                    temp.append(pooling1[j][1][0][k][l])
        result.append(temp)

    # print result array
    if prt != 0:
        print('******** RESULT ARRAY ********')
        print('')
        for i in range(len(result)):
            print('< image symbol: ' + str(data[i][0]) + ' >')
            printStr = '['
            for j in range(len(result[0])):
                printStr += (' ' + str(round(result[i][j], 2)) + ' ')
            print(str(printStr) + ' ]')

    return result

def CNNmain():
    # ConvolWidth : number of columns in each convolution result
    # ConvolHeight : number of rows in each convolution result
    # resultArrays : convolution result of each image data (N images),
    #                (length of [0], [1], ..., [N-1]) = ConvolWidth * ConvolHeight * Filters
    # testArray : convolution result of test image data (1 image),
    #             (length of [0]) = ConvolWidth * ConvolHeight * Filters
    
    (data, rows, cols, images, filtersize, filters, testimgdata, prt) = getData()

    print('######## making CNN array ########')
    print('')

    # make filters
    flts = makeFilters(data, rows, cols, images, filtersize, filters, prt)
    # CNN using the filter - data image
    resultArrays = CNN(data, rows, cols, images, filtersize, filters, flts, prt)

    # modify resultArrays -> sum = 1
    for i in range(len(resultArrays)):
        Sum = 0
        for j in range(len(resultArrays[0])):
            Sum += resultArrays[i][j]
        for j in range(len(resultArrays[0])):
            resultArrays[i][j] /= Sum

    if prt != 0: print('')
    print('######## test image ########')
    print('')

    # CNN using the filter - test image
    testArray = CNN([testimgdata], rows, cols, 1, filtersize, filters, flts, prt)

    # return final prediction
    maxVal = 0
    maxIndex = -1

    # print modified resultArrays
    if prt != 0:
        print('')
        print('**** modified result array ****')
        print('')
        for i in range(len(resultArrays)):
            print('< image symbol: ' + str(data[i][0]) + ' >')
            printStr = '['
            for j in range(len(resultArrays[0])):
                printStr += (' ' + str(round(resultArrays[i][j], 2)) + ' ')
            print(str(printStr) + ' ]')
            print('')

    # make list of sumProduct
    listOfSumProduct = [] # list of symbol and sumproduct
    for i in range(images):
        sumProduct = 0
        for j in range(len(testArray[0])):
            sumProduct += resultArrays[i][j] * testArray[0][j]
        listOfSumProduct.append([data[i][0], sumProduct])

    # print list of sumProduct
    for i in range(len(listOfSumProduct)):
        print('sumProduct of image ' + str(listOfSumProduct[i][0]) + ': ' + str(round(listOfSumProduct[i][1], 6)))

    # group for each symbol
    listOfSumProduct.sort(key=lambda x:x[0])

    groupedListOfSumProduct = [] # info about each symbol group
    count = 0 # count of each symbol
    Sum = 0 # sum of sumProduct of each symbol

    for i in range(len(listOfSumProduct)):
        count += 1
        Sum += listOfSumProduct[i][1]
        symbol = listOfSumProduct[i][0] # symbol of item
        
        if i < len(listOfSumProduct)-1:
            if listOfSumProduct[i+1][0] == symbol: continue

        # execute if new symbol
        groupedListOfSumProduct.append([symbol, Sum/count])
        count = 0
        Sum = 0

    # find max average group
    maxGroup = 0
    maxVal = 0
    for i in range(len(groupedListOfSumProduct)):
        if groupedListOfSumProduct[i][1] > maxVal:
            maxVal = groupedListOfSumProduct[i][1]
            maxGroup = i

    # print grouped list of sumProduct
    print('')
    for i in range(len(groupedListOfSumProduct)):
        print('avg sumProduct of symbol ' + str(groupedListOfSumProduct[i][0]) + ': ' + str(round(groupedListOfSumProduct[i][1], 6)))

    print('')
    print('final prediction of ' + str(testimgdata[0]) + ': ' + str(groupedListOfSumProduct[maxGroup][0]))

if __name__ == '__main__':
    CNNmain()
