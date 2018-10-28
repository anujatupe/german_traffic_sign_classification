******** PROJECT STRUCTURE  *****

I used the AWS credits provided by Udacity to run my project on AWS EC2 Instance. The AMI that I chose for my project is - Deep Learning AMI with Source Code (CUDA 8, Ubuntu). 

I followed the instructions provided in Lesson 2 of the Deep Learning module of the Machine Learning Nanodegree program. This helped me setup my EC2 instance where I could start my jupyter notebook and access it on my machine.

Classroom link for Lesson 2 of the Deep Learning module of the MLND program - https://classroom.udacity.com/nanodegrees/nd009t/parts/0ac87c1d-350a-417b-93c8-392dbf9cb8c2/modules/f5e5f204-ae46-415c-bd9b-a160eecf044f/lessons/29df00d8-01c2-4995-92fa-a4afd020be90/concepts/a20383fd-1e14-4311-824b-3d7981d99dee

Since, the dataset is big, we have mentioned instructions below on how to download it and place it in the project folder.


******** HOW TO MAKE THE PROJECT STRUCTURE? ********

1. Unzip final_capstone.zip. This will create a final_capstone folder - unzip final_capstone.zip

2. Change directory to final_capstone folder - cd final_capstone

3. Download the training and testing dataset and arrange it using the below instructions
	a. Download the zip files for training - wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip
	b. Download the zip files for testing - wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip

4. Unzip GTSRB_Final_Training_Images.zip, this will unzip contents in a folder named GTSRB - unzip GTSRB_Final_Training_Images.zip

5. Unzip  GTSRB_Final_Test_Images.zip, this will unzip contents in a folder named GTSRB - unzip GTSRB_Final_Test_Images.zip

6. Move the class_labels.csv file to GTSRB folder - mv  class_labels.csv GTSRB/.


******** SOFTWARE AND LIBRARIES USED ********

We followed the instructions given in Section 6 of Lesson 2 of the Deep Learning module. 

1. Change directory to final_capstone folder - cd final_capstone

2. We install the requirements by using the following command - sudo python3 -m pip install -r requirements/requirements-gpu.txt

This will install the below software and libraries - 

opencv-python==3.2.0.6
h5py==2.6.0
matplotlib==2.0.0
numpy==1.12.0
scipy==0.18.1
tqdm==4.11.2
keras==2.0.2
scikit-learn==0.18.1
pillow==4.0.0
ipykernel==4.6.1
tensorflow-gpu==1.0.0


******** EXPLORING THE PROJECT STRUCTURE **********


class_labels.csv - 
This file was not part of the dataset that we downloaded. This CSV file has the mapping between the class IDs and class names (meaningful German traffic sign names). I created this file for showing the predicted class label for the random images from web.

GTSRB - 
This folder has all the training and test images for the project. These images are downloaded from the http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

requirements - 
This folder has the requirements-gpu.txt file. As per Lesson 2 of the Deep learning module, we copy pasted this file here. This file has the versions of the different libraries that need to be installed. 

helpers - 
This folder has all the python files. All the python code written as a part of this project are inside these file. 
Following is the list of the python files. We will describe what each file does.

----------------------------------------------------------------------------------------------------------------------------------------------------

classDetails.py - This file has the method to show the details of the dataset. We print out the frequency of each class and we also print out a meaningful traffic sign name for each class. We plot the class IDs vs frequency plot. It shows the imbalance nature of our dataset. 

classLabels.py - This file has the method to read the CSV file which has the mapping between class IDs and class labels.

imageDetails.py - This file is used to print the image details of any random image from train or test dataset. It prints details like the image array itself, the image shape, the number of rows and columns in the image array, plotting the image itself.

lenetModel.py - This file has the method to return Lenet-5 model.

modelPerformance.py - This file has the method to print out different model performance metrics. It shows the precision, recall and F-beta score of the mode. It also prints out the classification report of the model. We calculate the confusion matrix for the model and plot it.

modifiedLenet.py - This file has the method to return the modified Lenet-5 model.

preprocessImage.py - This file has all the methods to read the images from the training and testing folders and preprocess the images.

simpleCNN.py - This file has the method to return a simple CNN model.

stratify.py - This file has the method to return train and validation sets such that train and validation dataset are equally balanced in terms of the classes.

testImages.py - This file has the methods to predict the class label for a random image of german traffic sign from the internet.

---------------------------------------------------------------------------------------------------------------------------------------------------

traffic_sign_recognition.ipynb - 
This is jupyter notebook which calls all our methods for the project.

test_traffic_sign_images - 
This  folder has the German traffic sign images from the web that we got for testing. 

