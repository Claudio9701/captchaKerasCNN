# coding: utf-8

import random
import string
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow
from skimage.transform import resize
from skimage.morphology import label
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, Activation, Dropout, Dense, Flatten, Conv2D, MaxPool2D, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adadelta
import tensorflow as tf

X_train = np.load('data/train_inputs.npy')
y_train = np.load('data/train_targets.npy')
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1]*y_train.shape[2])

height = 50
width = 150
num_classes = 36
word_len = 4
learning_rate = 0.1

input_tensor = Input(shape=(height, width, 3))

x = input_tensor

filters = [32, 64, 128, 256]

for filter in filters:

    x = Conv2D(filter, 3, padding='same', activation='relu')(x)
    x = Conv2D(filter, 3, padding='same', activation='relu')(x)
    x = Activation("relu")(x)
    x = MaxPool2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.25)(x)

x = [Dense(num_classes, activation='softmax', name='c%d'%(i+1))(x) for i in range(word_len)]
output = concatenate(x)

model = Model(inputs=input_tensor, outputs=output)

opt = Adadelta(lr=learning_rate)

model.compile(loss = 'categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

earlystopper = EarlyStopping(patience=5, verbose=1)

checkpointer = ModelCheckpoint(filepath="model.{epoch:02d}--{val_loss:.2f}-{val_acc:.4f}.hdf5",
                               verbose=1, save_best_only=True)

print(model.summary())

h = model.fit(X_train, y_train, epochs= 15, validation_split=0.1, callbacks=[earlystopper, checkpointer])

X_test = np.load('data/test_inputs.npy')
y_test = np.load('data/test_targets.npy')

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#
# def vec2char(predictions, charset=list(string.digits + string.ascii_letters)):
#
#     max_ix =  [np.argmax(prediction.reshape((4,36)), axis=1) for prediction in predictions]
#
#     for prediction in max_ix:
#         string_ = ''
#         for i in max_ix:
#             string_ += char_set[i]
#
#     print(string_)
