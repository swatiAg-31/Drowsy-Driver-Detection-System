from tensorflow import keras
import tensorflow as tf
from keras.layers import Dense,Dropout,Activation,Add,MaxPooling2D,Conv2D,Flatten
from keras.models import Sequential 
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
import seaborn as sns
train_dir = 'C:\\Users\\swati\\Desktop\\Drowsinessdetection\\testdata'
val_dir = 'C:\\Users\\swati\\Desktop\\Drowsinessdetection\\datanew\\train'
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=32,
        color_mode="grayscale",
        class_mode='categorical')
validation_generator = valid_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=32,
        color_mode="grayscale",
        class_mode='categorical')
 
num_train = len(train_generator.classes) #1234
batch_size = 32
num_val = len(validation_generator.classes) #218
  
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
MaxPooling2D(pool_size=(2,2)),
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit_generator(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=150,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size)
model.save('C:/Users/swati/Desktop/Drowsinessdetection/models/model2.h5', overwrite=True)
 
model.evaluate(validation_generator)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, 301)
 
#accuracy plot
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.figure()
#loss plot
plt.plot(epochs, loss, color='green', label='Training Loss')
plt.plot(epochs, val_loss, color='blue', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

