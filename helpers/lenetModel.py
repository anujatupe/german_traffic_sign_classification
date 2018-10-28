from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten


"""
    This method returns the Lenet-5 Model
"""
def get_lenet_model(input_image_size, total_classes):
    model = Sequential()    
    model.add(Conv2D(6, (5,5), activation="relu", input_shape=(3, input_image_size, input_image_size)))
    model.add(MaxPooling2D(pool_size=2, strides = 2))
    model.add(Conv2D(16, (5,5), input_shape=(6, 14, 14), activation="relu"))
    model.add(MaxPooling2D(pool_size=2, strides = 2))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(total_classes, activation='softmax'))
    return model 