import numpy as np
from skimage import transform, color, exposure
import cv2
import matplotlib.pyplot as plt
import csv
from skimage import io
import warnings

"""
Used to perfom pre processing on the images
"""
def preprocess_image(input_image, image_size):
    #Image Histogram equalization - Used to improve the contrast of the image
    hsv_image = color.rgb2hsv(input_image)
    
    #Normalize each channel separately
    hsv_image[:, :, 2] = exposure.equalize_hist(hsv_image[:, :, 2])
    
    #Convert HSV image back to RGB
    input_image = color.hsv2rgb(hsv_image)
    
    #Crop image around center of the image.
    smaller_side = min(input_image.shape[:-1])
    center = input_image.shape[0]//2, input_image.shape[1]//2
    input_image = input_image[center[0]-smaller_side//2:center[0]+smaller_side//2, center[1]-smaller_side//2:center[1]+smaller_side//2, :]

    #Resizing the image
    input_image = transform.resize(input_image, (image_size, image_size))

    #roll color axis to axis 0
    input_image = np.rollaxis(input_image, -1)

    return input_image


"""
Plot the input image
"""
def display_image(input_image):
    print("Plotting the image..")
    rolled_axis_image = np.rollaxis(input_image, -1)
    second_rolled_axis_img = np.rollaxis(rolled_axis_image, -1)
    plt.imshow(second_rolled_axis_img)
    plt.show()



# The German Traffic Sign Recognition Benchmark
# sample code for reading the traffic sign images and the
# corresponding labels
#Reference - http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset


"""
This method is used for - 
1. Preprocess train images from directory.
2. Store preprocessed train images to a list
3. Read train image class ids/labels from every CSV file with the traffic sign images.
4. Store these train images in a list 
5. Store the train images' class ids/labels to a list

"""

def readPreprocessedTrainTrafficSigns(rootpath, image_size):
    images = [] 
    labels = [] 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in range(0,43):
            prefix = rootpath + '/' + format(c, '05d') + '/' 
            gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') 
            gtReader = csv.reader(gtFile, delimiter=';') 
            next(gtReader) 
            for row in gtReader:
                images.append(preprocess_image(io.imread(prefix + row[0]), image_size))    
                labels.append(int(row[7])) 
            gtFile.close()
    return images, labels


"""
This method is used for - 
1. Preprocess test images from directory.
2. Store preprocessed test images to a list
3. Read test image class ids/labels from GT-final_test.csv
4. Store these test images in a list
5. Store these test images' class ids/labels to a list

"""

def readPreprocessedTestTrafficSigns(rootpath, image_size):
    images = [] 
    labels = [] 
    prefix = rootpath + '/'
    gtFile = open(prefix + 'GT-final_test.csv')
    gtReader = csv.reader(gtFile, delimiter=';') 
    next(gtReader) 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for row in gtReader:
            images.append(preprocess_image(io.imread(prefix + row[0]), image_size))          
            labels.append(int(row[7])) 
    gtFile.close()
    return images, labels
