import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from helpers.preprocessImage import display_image

"""
Printing details of a random image -
1. Printing the image array
2. Printing the image shape
3. Printing the total number of rows and columns in the image
4. Printing the label for the image

We also plot the processed image
"""

def randomImageDetails(random_number, images, labels_arr):

    print("\nThe image {} is: ".format(random_number))
    print(images[random_number])

    print(("\nImage {} shape is: ").format(random_number))
    print(images[random_number].shape)

    print("\nIn image {}: Total number of rows: {} Total number of columns: {}".format(random_number, len(images[random_number][1]), len(images[random_number][2])))

    print("\nThe label for image {} is: {}".format(random_number, labels_arr[random_number]))

    display_image(images[random_number])