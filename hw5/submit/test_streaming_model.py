# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 23:29:02 2020

@author: liu
"""


import librosa
import numpy as np
import tensorflow as tf

data = np.zeros(67*600)
data = data.reshape((1,600,67))

path = 'train_mandarin/mandarin_0031.wav'#20，30，31
# path = 'train_english/english_0001.wav'#4，8，10
# path = 'train_hindi/hindi_0003.wav' #1

y, sr = librosa.load(path, sr=16000)
y, index = librosa.effects.trim(y)
mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length = int(sr*0.01))
mat = mat[:,:58800].T
mat = mat.reshape((98,600,64))

model = tf.keras.models.load_model('stream_model.h5')


def cus_predict(model, mat):
    # p = []
    label = np.zeros(3) 
    for i in range(98):
        for j in range(600):
            temp = mat[i][j]
            temp = temp.reshape((1,1,64))
            prediction = model.predict(temp)[0][0]
            # p.append(prediction)
            label[np.argmax(prediction)]+=1
            model.reset_states() 
    #         print(prediction)
    # print(np.argmax(label))   
    if np.argmax(label) ==0:
        return 'English'
    elif np.argmax(label) == 1:
        return 'Mandarin'
    else:
        return 'Hindi'

language = cus_predict(model, mat)
print('the input video is {0}'.format(language))