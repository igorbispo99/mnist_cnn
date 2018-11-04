import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import scipy.ndimage
import keras

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, UpSampling2D, Flatten, MaxPooling2D, Dropout, Reshape
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
    
    # Encoder Layers
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(SIZE_W, SIZE_H, 1)))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(96, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    # Squashing
    model.add(Reshape((2, 2, 1)))

    # Decdoder layers
    model.add(Conv2D(128, kernel_size=1, activation='relu'))
    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(96, kernel_size=3, padding='same', activation='relu'))
    model.add(UpSampling2D((7, 7)))

    print(model.summary())

    model.add(Conv2D(64, kernel_size=3, padding='same',activation='relu'))
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))

    model.add(Conv2D(1, kernel_size=3, activation='sigmoid', padding='same'))

    print(model.summary())

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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

datagen = ImageDataGenerator(
        shear_range=0.3,
        zoom_range=0.3,
        width_shift_range=0.3,
        height_shift_range=0.3
    )


model = instantiate_cnn()
x_train, y_train, x_test, y_test = generate_mnist()

model.fit_generator(datagen.flow(x_train, x_train, batch_size=32), steps_per_epoch=len(x_train)/32, epochs=20, validation_data=(x_test, x_test))

model.save("mnist_cnn_auto.h5")


