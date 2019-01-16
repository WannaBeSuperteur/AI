import CNN as CNN

# read data from file
def getData():
    file = open('CNNimage.txt', 'r')
    read = file.readlines() # read[0]: information about images
    fns = read[1].split('\n')[0].split(' ') # image file names
    symbols = read[2].split(' ') # symbols of image
    file.close()
    return (read[0], fns, symbols)

# write image data
def fileWrite(imginfo, fns, symbols):
    file = open('CNN.txt', 'w')
    file.write(imginfo)

    info = imginfo.split(' ')
    imageHeight = int(info[0]) # height of image
    imageWidth = int(info[1]) # width of image
    imageCount = int(info[2]) # number of images except for last(test) image

    # write information
    for i in range(imageCount+1):
        file.write(symbols[i] + '\n') # write symbol

        # write content of each image
        imgf = open(fns[i], 'rb')
        imgf.seek(57)
        content = str(imgf.read(3*imageHeight*imageWidth)).split('\\x')

        # write to file
        for j in range(imageHeight-1, -1, -1):
            for k in range(imageWidth):
                if content[3*(j*imageHeight+k)] >= '8': file.write('.')
                else: file.write('#')
            file.write('\n')
        
        imgf.close
                
    file.close

# main
(imginfo, fns, symbols) = getData()
fileWrite(imginfo, fns, symbols)
CNN.CNNmain()
