# Drowsy-Driver-Detection-System
  Built a model for drowsiness detection of a driver by real-time Eye-Tracking in videos.
 Designed end to end automated pipeline which  includes following major building blocks:
  1. State Informer- Captures current state of driver.
  2. State Detector-Detects the state using trained Deep Learning ML model.
  3. Alert Manager-Receives the response from state detector and broadcasts the necessary alerts as per level of seriousness.

# Tech Stack used
1. Keras (for building CNN model)
2. OpenCV
3. Flask
4. HTML,CSS
5. JavaScript
6. Bootstrap

# Project Prerequisites
The requirement for this Python project is a webcam through which we will capture images. You need to have Python (3.6 version recommended) installed on your system, then using pip, you can install the necessary packages.

1. OpenCV – pip install opencv-python (face and eye detection).
2. TensorFlow – pip install tensorflow (keras uses TensorFlow as backend).
3. Keras – pip install keras (to build our classification model).
4. Pygame – pip install pygame (to play alarm sound).
5. Flask - pip install  Flask (for deployment of Deep Learning Model)

# File structure 

1. haarcascade files-> folder contains of the xml files that are needed to detect the face and eyes of the person.
2. static folder-> contains html  css files for homepage of webapp and js file for homepage.
3. template folder  -> contains of html pages for home page for Droive webapp
4. app.py -> main file to run the project.
5. model.py ->  file contains the program through which we built our classification model by training on the dataset.The implementation of convolutional neural network  is in               this file.


# Instructions to run
$ python  app.py


# CNN Model 
  1. Dataset of closed and open eyes images is used.
  2. Model is trained on 150 epochs with 4846 images.
  3. There are 3 convolution layers added to CNN model.
  4. Activation Layers:Relu,Softmax
  5. Optimizer-Adam
  6. Accuracy of  Model on test data-94.32 percent.
 
