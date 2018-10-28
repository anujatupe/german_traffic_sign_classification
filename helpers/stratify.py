from sklearn.model_selection import StratifiedShuffleSplit

"""
This method is used split the dataset into train and validation sets.
We stratify the data using the StratifiedShuffleSplit sklearn method.
"""
def get_stratified_dataset(images, labels):
    
    stratified_dataset = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, validation_index in stratified_dataset.split(images, labels):
        X_train, X_validation = images[train_index], images[validation_index]
        y_train, y_validation = labels[train_index], labels[validation_index]
    return X_train, y_train, X_validation, y_validation
    