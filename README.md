# Image-Processing-Parallelization
A Python project demonstrating image brightness enhancement using both single-core and multi-core parallel processing techniques for faster execution.

# Image Processing with Multiprocessing and Single Processing

This project demonstrates two methods of image processing: one using **Multiprocessing** and the other using **Single Processing**. Both approaches involve increasing the brightness of an image by adding 50 units to each pixel, with the constraint that pixel values do not exceed 255. The main difference between the two methods is the parallel execution of the task using multiple processor cores in the **Multiprocessing** method.

## Libraries Used

- **cv2** (OpenCV): Used for reading and converting images between color spaces.
- **Matplotlib**: Used for displaying processed images.
- **Multiprocessing**: Used for distributing the image processing task across multiple CPU cores.
- **Numpy**: Used for creating and manipulating arrays of image pixels.
- **Time**: Used for calculating and displaying the execution time.

## Multiprocessing Version

### Explanation

In the **Multiprocessing** approach, the image is divided into four slices, and each slice is processed in parallel using a separate process. This method uses the Python `multiprocessing` library to assign one process to each slice. The results are then combined after all processes have finished.

### Workflow

1. The start time of the program is recorded using the `time` library.
2. The image is read and converted to grayscale using OpenCV (`cv2`).
3. The image dimensions (height and width) are stored.
4. Four processes are created to handle one quarter of the image each.
5. The processed slices are combined into a final image.
6. The execution time is calculated and displayed.
7. The final processed image is displayed.

### Code

```python
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
```

## Single Processing Version

### Explanation

The **Single Processing** method processes the entire image using a single CPU core, without dividing the task into multiple processes. This is a simpler approach that doesn’t require the use of the `multiprocessing` or `numpy` libraries.

### Workflow

1. The start time of the program is recorded using the `time` library.
2. The image is read and converted to grayscale using OpenCV (`cv2`).
3. The `image_processor()` function is called to process the entire image.
4. The processed image is displayed, and the execution time is printed.

### Code

```python
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
```

## Test Results

### Processor 1: Intel Core i5-10400F
- **Cores**: 6
- **Threads**: 12
- **Base Clock / Boost Clock**: 2.9 GHz / 4.3 GHz

#### Parallel Processing Execution Time:
- 9.35 seconds
- 8.97 seconds

#### Single-Core Processing Execution Time:
- 25.53 seconds
- 23.55 seconds

### Processor 2: Intel Core i7-4600M
- **Cores**: 2
- **Threads**: 4
- **Base Clock / Boost Clock**: 2.9 GHz / 3.6 GHz

#### Parallel Processing Execution Time:
- 15.93 seconds
- 16.04 seconds

#### Single-Core Processing Execution Time:
- 29.22 seconds
- 29.27 seconds

## Conclusion

The results show that using **Multiprocessing** significantly reduces the execution time compared to **Single Processing**, especially with a multi-core processor. For instance, on the Intel Core i5-10400F, the parallel approach completes the task in about 9 seconds, while the single-core approach takes around 25 seconds. Similarly, on the Intel Core i7-4600M, the parallel approach finishes in around 15 seconds, while single-core processing takes approximately 29 seconds.

This demonstrates the efficiency of parallel processing for computationally intensive tasks. Leveraging multiple CPU cores can lead to faster execution times, especially when the task can be divided into independent sub-tasks. This approach is particularly beneficial for processors with more cores and threads.

## Future Recommendations

It is recommended that future applications be designed to leverage **parallel processing** where possible, especially when dealing with tasks that can be split into independent operations. This can lead to faster execution times and improved resource utilization, helping to reduce power consumption and increase overall performance.
