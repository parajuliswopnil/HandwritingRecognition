#!/usr/bin/env python
# coding: utf-8

# In[2]:

import gzip
import matplotlib.pyplot as plt
import numpy as np
import struct
import os
import random
import tensorflow as tf
import keras
import keras.utils
from IPython import get_ipython
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras import backend
import cv2 as cv



# In[9]:


## Load Data


# In[3]:


dataset_path = '/home/swopnil/PycharmProjects/HandwritingRecognition/gzip/'
log_path = '/home/swopnil/PycharmProjects/HandwritingRecognition/logs/'


# In[4]:


def read_idx(filename):
    print(f'Processing data from {filename}.')
    with gzip.open(filename, 'rb') as f:
        z, dtype, dim = struct.unpack('>HBB', f.read(4))
        print(f'Dimensions: {dim}')
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dim))
        print(f'Shape: {shape}')
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


# In[5]:


def load_emnist():
    train_images = dataset_path + 'emnist-letters-train-images-idx3-ubyte.gz'
    train_labels = dataset_path + 'emnist-letters-train-labels-idx1-ubyte.gz'
    test_images = dataset_path + 'emnist-letters-test-images-idx3-ubyte.gz'
    test_labels = dataset_path + 'emnist-letters-test-labels-idx1-ubyte.gz'
    train_x = read_idx(train_images)
    train_y = read_idx(train_labels)
    test_x = read_idx(test_images)
    test_y = read_idx(test_labels)
    return (train_x, train_y, test_x, test_y)


# In[6]:


raw_train_x, raw_train_y, raw_test_x, raw_test_y = load_emnist()


# In[7]:


labels = 'abcdefghijklmnopqrstuvwxyz'


# In[8]:


img_num = 30
plt.imshow(raw_train_x[img_num].T, cmap='gray')
print(labels[raw_train_y[img_num] - 1])
plt.colorbar()
plt.show()


# In[9]:


## Prepare data


# In[10]:


print(f'train_x shape: {raw_train_x.shape}')
print(f'train_y shape: {raw_train_y.shape}')
print(f'test_x shape: {raw_test_x.shape}')
print(f'test_y shape: {raw_test_y.shape}')


# In[11]:


img_height = len(raw_train_x[0])
img_width = len(raw_train_x[1])
input_shape = img_height * img_width
print(input_shape)


# In[12]:


train_x = raw_train_x.reshape(len(raw_train_x), input_shape)
print(train_x.shape)
test_x = raw_test_x.reshape(len(raw_test_x), input_shape)
print(test_x.shape)


# In[54]:


## Normalize


# In[13]:


train_x = train_x.astype('float32')
test_x = test_x.astype('float32')
print(train_x.dtype)


# In[14]:


train_x /= 255
test_x /= 255


# In[15]:


img_num = 30
plt.imshow(raw_train_x.reshape(len(train_x), 28, 28)[img_num].T, cmap='gray')
print(labels[raw_train_y[img_num] - 1])
plt.colorbar()
plt.show()


# In[16]:


labels = 'abcdefghijklmnopqrstuvwxyz'
n_cat = len(labels) + 1


# In[17]:


train_y = tf.keras.utils.to_categorical(raw_train_y)
test_y = tf.keras.utils.to_categorical(raw_test_y)


# In[18]:


print(raw_train_y[30])
print(train_y[30])


# In[19]:


## Define and compile model


# In[36]:


model = keras.models.Sequential()
model.add(Dense(16, input_dim=input_shape, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(n_cat, activation='softmax'))


# In[37]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[22]:


## Train the model


# In[38]:


tensorboard = TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True)
callbacks_list = [tensorboard]


# In[39]:


model.fit(train_x, train_y, epochs=15, batch_size=100, callbacks=callbacks_list)


# In[40]:


results = model.evaluate(test_x, test_y)


# In[90]:


print(results[0]*100, results[1]*100)


# In[41]:


model.save('digits.model')


# In[56]:


# img1 = cv.imread('/home/swopnil/PycharmProjects/HandwritingRecognition/src/2.png')
# img2= np.array([img1])
# img3 = img2.reshape(3, 784)
# prediction = model.predict(img3)
# print(prediction)
# print(f'the result is {np.argmax(prediction)}')
# # plt.imshow(img[0], cmap=plt.cm.binary)
# # plt.show()
#

# img1 = cv.imread('/home/swopnil/PycharmProjects/HandwritingRecognition/src/2.png')
# img2= np.array([img1])
# img3 = img2.reshape(3, 784)

img_num = 10000
img = raw_test_x.reshape(len(raw_test_x), input_shape)
print(labels[raw_test_y[img_num] - 1])
prediction = model.predict(img)
print(f'the result is {np.argmax(prediction)}')
# plt.imshow(img[0], cmap=plt.cm.binary)
# plt.show()