import csv

"""
This method reads the class_labels.csv files and loads it in memory.
The class_labels.csv file has the mapping between class IDs and class labels.
For example - Class ID 0 is for Speed Limit 20.
"""

def loadClassLabelsMapAndList(rootpath):
    #class_id_labels_map = {} 
    class_id_labels_list = []
    prefix = rootpath + '/'
    class_labels_File = open(prefix + 'class_labels.csv')
    class_labels_Reader = csv.reader(class_labels_File, delimiter=';') 
    next(class_labels_Reader) 
    for row in class_labels_Reader:
        #class_id_labels_map[int(row[0])] = row[1]
        class_id_labels_list.append(row[1])
    class_labels_File.close()
    return class_id_labels_list
    #return class_id_labels_map, class_id_labels_list