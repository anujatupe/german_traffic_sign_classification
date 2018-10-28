import glob
from IPython.core.display import display, Image 
from helpers.preprocessImage import preprocess_image
import numpy as np
from skimage import io
import warnings

"""
This method prints the class label for the incoming image
"""
def detect_image_type_lenet_model(input_cnn_model, input_img_path, class_labels_list, IMAGE_SIZE):
    predicted_sign = predict_traffic_sign_lenet_model(input_cnn_model, input_img_path, class_labels_list, IMAGE_SIZE)
    print("The predicted traffic sign is " + predicted_sign)
    return predicted_sign

""" 
This method predicts and returns the class label for the incoming image
"""
def predict_traffic_sign_lenet_model(input_cnn_model, input_img_path, class_labels_list, IMAGE_SIZE):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image_to_be_recognised = preprocess_image(io.imread(input_img_path), IMAGE_SIZE)
        predicted_label = input_cnn_model.predict(np.array([image_to_be_recognised]))
    return class_labels_list[np.argmax(predicted_label)]