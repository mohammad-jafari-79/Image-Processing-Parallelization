import cv2 as cv
import matplotlib.pyplot as plt
import time

def image_processor(array):
    for i in range(0, array.shape[0]):
        for j in range(0, array.shape[1]):
            array[i][j] = array[i][j] + 50
            if(array[i][j]>255):
                array[i][j]=255
    plt.imshow(array,cmap='gray')

if __name__ == '__main__':        
    startTime = time.time()      
    catImage = cv.imread('./cat.jpg')
    catImage = cv.cvtColor(catImage, cv.COLOR_BGR2GRAY)
    image_processor(catImage)
    print('that took: {} seconds'.format(time.time() - startTime))