from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout

"""
This is a modified Lenet-5 model.
Changed the number of filters in the first convolutional layer from 6 to 12
Changed the number of filters in the second convolutional layer from 16 to 32
Added a dropout layer after the flatten layer.
"""
def get_modified_lenet_model(input_image_size, total_classes):
    model = Sequential()    
    model.add(Conv2D(12, (5,5), activation="relu", input_shape=(3, input_image_size, input_image_size)))
    model.add(MaxPooling2D(pool_size=2, strides = 2))
    model.add(Conv2D(32, (5,5), input_shape=(6, 14, 14), activation="relu"))
    model.add(MaxPooling2D(pool_size=2, strides = 2))
    model.add(Flatten())
    #Added dropout layer 0.5
    model.add(Dropout(0.5))
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(total_classes, activation='softmax'))
    return model 