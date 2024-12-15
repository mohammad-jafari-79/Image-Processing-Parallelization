import cv2 as cv
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import numpy as np

def image_processor(array, sliceNumber, return_dict):
    for i in range(0, array.shape[0]):
        for j in range(0, array.shape[1]):
            array[i][j] = array[i][j] + 50
            if (array[i][j] > 255):
                array[i][j] = 255
    return_dict[sliceNumber] = array

if __name__ == '__main__':
    startTime = time.time()
    manager = mp.Manager()
    return_dict = manager.dict()
    catImage = cv.imread('./cat.jpg')
    brightenedImage = np.zeros((catImage.shape[0], catImage.shape[1]))
    catImage = cv.cvtColor(catImage, cv.COLOR_BGR2GRAY)
    height, width = catImage.shape

    processes = []
    for i in range(0, 4):
        p = mp.Process(target=image_processor, args=(
            catImage[:][int(i*height/4):int((i+1)*height/4)], i, return_dict))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    brightenedImage = return_dict[0]
    for i in range(1, len(return_dict.values())):
        brightenedImage = np.concatenate((brightenedImage, return_dict[i]))

    grayPic = plt.imshow(brightenedImage, cmap='gray')
    print('that took: {} seconds'.format(time.time() - startTime))
    plt.show()