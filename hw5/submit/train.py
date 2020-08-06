# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:32:39 2020

@author: liu
"""
import h5py
import numpy as np
from tensorflow.keras.layers import Input, GRU, Dense, Dropout
import tensorflow.keras as tfk
from sklearn.utils import class_weight
import librosa
import tensorflow as tf
import pandas as pd


hf = h5py.File('data.h5','r')
testdata = np.array(hf.get('language'))
x1 = testdata[:,:,:-3]
y1 = testdata[:,:,-3:]
hf.close()

X = Input(shape = (600,64))
training = GRU(30,return_sequences=True,stateful=False)(X)
training = Dense(256, activation='relu')(training)
Y = Dense(3, activation='softmax')(training)
model = tfk.Model(inputs=X, outputs=Y)
model.compile(optimizer='adam', loss=tfk.losses.CategoricalCrossentropy(),metrics = ['accuracy'])
model.summary()
model.fit(x=x1,y=y1,batch_size=10, epochs=10, shuffle='True')
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png')
model.save_weights('weights.h5', overwrite=True)
model.save('model.h5')

# Stream inference model
StreamX = Input(batch_shape = (1,None,64))
training = GRU(40,return_sequences=True, stateful=True)(StreamX)
training = Dense(256, activation='relu')(training)
# training = Dense(32, activation='tanh')(training)
StreamY = Dense(3, activation='softmax')(training)
stream_model = tfk.Model(inputs=StreamX, outputs=StreamY)
stream_model.compile(optimizer='adam', loss=tfk.losses.CategoricalCrossentropy(), metrics=['accuracy'])
stream_model.summary()


stream_model.load_weights('weights.h5')
stream_model.save('stream_model2.h5')
stream_model = tf.keras.models.load_model('stream_model2.h5')
# n = len(y1)
# acc = []
# for s in range(23030):#23030
#     for j in range(600):
#         x = x1[s].reshape((1,1,64))
#         y = y1[s].reshape((1,1,3))
#         i = stream_model.evaluate(x,y)
#         print('Streaming-Model acc is {0}'.format(i))
#         acc.append(i)
#         stream_model.reset_states()
# accuracy = np.mean(acc)
