import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape()

x_train = x_train / 255
x_test = x_test / 255

model = keras.models.Sequential([keras.layers.Flatten(input_shape=(28, 28)), \
                                 keras.layers.Dense(256, activation='relu'),\
                                 keras.layers.Dense(128, activation='relu'), \
                                 keras.layers.Dense(64, activation='relu'), \
                                 keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)

plt.imshow(x_train[0], cmap='gray')
plt.show()

print(model.predict(x_train[0].reshape(1, 28, 28)))

print(np.argmax(model.predict(x_train[0].reshape(1, 28, 28))))