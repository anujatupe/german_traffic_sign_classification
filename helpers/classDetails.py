import matplotlib.pyplot as plt
import numpy as np

"""
This method is used to visualize the class IDs and their frequencies.
It also shows the mapping between the class IDs and labels.
For example - Class ID 0 has the class label Speed Limit 20.
This method prints the class with the highest frequency and the class 
with the lowest frequency.
"""

def datasetDetails(labels_arr, total_classes, class_labels_list):
    labels_int_arr = [int(l) for l in labels_arr]
    plt.hist(labels_int_arr, np.arange(min(labels_int_arr), max(labels_int_arr) + 1, 1), rwidth=0.5)
    plt.xlabel('Traffic Sign Class IDs')
    plt.ylabel('Total Count')
    plt.show()

    labels_frequency = []
    
    for l in range(total_classes):
        labels_frequency.insert(l, labels_int_arr.count(l))

    max_frequency = max(labels_frequency)
    min_frequency = min(labels_frequency)

    print("Class {} has the highest frequency and the frequency is {}".format(labels_frequency.index(max_frequency), max_frequency))
    print("Class {} has the lowest frequency and the frequency is {}".format(labels_frequency.index(min_frequency), min_frequency))
    
    print("******* Printing Class ID, Class Label and Frequency Mapping *******")
    
    for i in range(total_classes):
        print("Class ID {} - {} - Frequency {}".format(i, class_labels_list[i], labels_frequency[i]))

    