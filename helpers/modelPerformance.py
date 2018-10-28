
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

"""
This method is used to show how well the model is doing.
It prints the precision, recall, fscore of the model.
It prints out the classification report of the model.
It prints out the confusion matrix for the model.
It also plots the confusion matrix.

"""
def show_models_performance(test_labels, y_pred, total_classes):

    lenet_test_accuracy = np.sum(y_pred == test_labels)/np.size(y_pred)
    print("\nLenet Model accuracy for the test dataset: {}".format(lenet_test_accuracy))


    precision, recall, fscore, support = precision_recall_fscore_support(test_labels, y_pred, average='micro')
    print("\nWhen Average is micro - Calculate metrics globally by counting the total true positives, false negatives and false positives.")
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))

    precision, recall, fscore, support = precision_recall_fscore_support(test_labels, y_pred, average='macro')
    print("\nWhen Average is macro - Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.")
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))

    precision, recall, fscore, support = precision_recall_fscore_support(test_labels, y_pred, average='weighted')
    print("\nWhen Average is weighted - Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.")
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))


    print("\nClassification Report")
    print(classification_report(test_labels, y_pred, digits=5))


    #Find the confusion matrix
    conf_matrix = confusion_matrix(test_labels, y_pred)

    #Printing Confusion Matrix 
    print("\nConfusion Matrix")
    print(conf_matrix)

    #Plotting Confusion Matrix. 
    conf_matrix_dataframe = pd.DataFrame(conf_matrix, index = [class_num for class_num in range(total_classes)], columns = [class_num for class_num in range(total_classes)])
    plt.figure(figsize = (40,28))
    sn.set(font_scale=2)
    sn.heatmap(conf_matrix_dataframe, annot=True, annot_kws={"size": 24})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()