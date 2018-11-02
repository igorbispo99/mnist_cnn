import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np

import keras

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator

SIZE_H = 28
SIZE_W = 28
NUM_OF_CLASS = 10

GT_PATTERNS = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
])

def instantiate_cnn():
    print("Instantiating CNN...")
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(SIZE_W, SIZE_H, 1)))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(96, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.50))
    
    model.add(Dense(NUM_OF_CLASS, activation='softmax'))
    model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def generate_mnist():
    mnist = keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    
    x_train = np.array(1/255 * x_train).reshape(len(x_train), SIZE_H, SIZE_W, 1)
    x_test = np.array(1/255 * x_test).reshape(len(x_test), SIZE_H, SIZE_W, 1)

    y_t = []
    for y in y_test:
        y_t.append(GT_PATTERNS[y])
    
    y_test = np.array(y_t)

    y_t = []
    for y in y_train:
        y_t.append(GT_PATTERNS[y])
    
    y_train = np.array(y_t)

    return x_train, y_train, x_test, y_test

model = instantiate_cnn()
x_train, y_train, x_test, y_test = generate_mnist()

model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
model.save("mnist_cnn.h5")



