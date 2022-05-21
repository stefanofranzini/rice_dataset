#!/usr/bin/python3.8

import tensorflow as tf
import splitfolders as spf
import numpy as np
import matplotlib.pyplot as plt
import os

############################################

def plot_images(loader):

    plt.figure(figsize=(10, 10))
    for images, labels in loader.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

############################################

#### split folders

#spf.ratio('Rice_Image_Dataset', output="output", seed=1337, ratio=(.8, 0.1,0.1)) 

#### define some variables

BATCH = 64
IMG_WIDTH = 96
IMG_HEIGHT = 96

#### define loaders for train, test and validation

train_loader = tf.keras.preprocessing.image_dataset_from_directory("./output/train",seed=234,image_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=BATCH)
test_loader = tf.keras.preprocessing.image_dataset_from_directory("./output/test",seed=123,image_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=BATCH)
validation_loader = tf.keras.preprocessing.image_dataset_from_directory("./output/val",seed=123,image_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=BATCH)

#### get class names and plot some

class_names = train_loader.class_names
print(class_names)

plot_images(train_loader)
plt.show()

#### autotune and optimize

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_loader.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_dataset = test_loader.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = validation_loader.cache().prefetch(buffer_size=AUTOTUNE)

#### define model

data_augmentation = tf.keras.Sequential([ tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal",input_shape=(IMG_HEIGHT,IMG_WIDTH,3)),
                                          tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
                                          tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),])

model = tf.keras.models.Sequential([data_augmentation,
                                    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
                                    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
                                    tf.keras.layers.MaxPooling2D(),
                                  
                                    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
                                    tf.keras.layers.MaxPooling2D(),
                                    
                                    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
                                    tf.keras.layers.MaxPooling2D(),
                                    
                                    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
                                    tf.keras.layers.MaxPooling2D(),
                                    
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(len(class_names))
                                ])
                                
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
print(model.summary())

epochs = 10
history = model.fit(
  train_dataset,
  validation_data=val_dataset,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

'''
plt.figure(figsize=(10, 10))
for images, labels in test_loader.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predictions = model.predict(tf.expand_dims(images[i], 0))
        score = tf.nn.softmax(predictions[0])
        plt.ylabel("Predicted: "+class_names[np.argmax(score)])
        plt.title("Actual: "+class_names[labels[i]])
        plt.gca().axes.yaxis.set_ticklabels([])        
        plt.gca().axes.xaxis.set_ticklabels([])
'''                                     
                                            
                                            
                                            
                                            
                                            
                                            

