from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dense, Activation
from keras import backend as K
K.set_image_dim_ordering('th')

"""
 This is a simple CNN model.
"""
def get_simple_cnn_model(image_size):
    simple_model = Sequential()
    simple_model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(3, image_size, image_size)))
    simple_model.add(MaxPooling2D(pool_size=2, strides=2))
    simple_model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
    simple_model.add(MaxPooling2D(pool_size=2, strides=2))
    simple_model.add(Conv2D(filters=256, kernel_size=2, padding='same', activation='relu'))
    simple_model.add(MaxPooling2D(pool_size=2, strides=2))
    simple_model.add(GlobalAveragePooling2D())
    simple_model.add(Dense(43, activation='softmax'))
    simple_model.summary()

    return simple_model

