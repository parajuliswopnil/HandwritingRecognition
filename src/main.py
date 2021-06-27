import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import io as sio
import keras

# mat = sio.loadmat('matlab/emnist-letters.mat')
# data = mat['dataset']
#
# (x_train, y_train) = (data['train'][0, 0]['images'][0, 0], data['train'][0, 0]['labels'][0, 0])
# (x_test, y_test) = (data['test'][0, 0]['images'][0, 0], data['test'][0, 0]['labels'][0, 0])
#
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)
#
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=27, activation=tf.nn.softmax))
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=3)
#
# loss, accuracy = model.evaluate(x_test, y_test)
# print(accuracy)
# print(loss)


img = cv.imread('/home/swopnil/PycharmProjects/HandwritingRecognition/src/1.png')
img = np.array([img])
plt.imshow(img[0], cmap=plt.cm.binary)
plt.show()
