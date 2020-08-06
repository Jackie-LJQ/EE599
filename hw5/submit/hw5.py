# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:07:12 2020

@author: liu
"""

import librosa
import h5py
import numpy as np

data = np.zeros(67*600)
data = data.reshape((1,600,67))
label = np.array([1,0,0]*58800)
label = label.reshape((98,600,3))
for i in range(1,121):#121
    if i<10:
        path = '/home/ubuntu/train/train_english/english_000{0}.wav'.format(i)
        # path = 'F:/599dl/hw5/train/train_english/english_000{0}.wav'.format(i)
    elif i < 100:
        path = '/home/ubuntu/train/train_english/english_00{0}.wav'.format(i)
    else:
        path = '/home/ubuntu/train/train_english/english_0{0}.wav'.format(i)
    y, sr = librosa.load(path, sr=16000)
    y, index = librosa.effects.trim(y)
    mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.0025), hop_length = int(sr*0.01))
    mat = mat[:,:58800].T
    mat = mat.reshape((98,600,64))
    mat = np.concatenate((mat, label), axis=-1)
    data = np.concatenate((data, mat))
data = data[1:]

label = np.array([0,1,0]*58800)
label = label.reshape((98,600,3))
for i in range(1,83):#83
    if i<10:
        path = '/home/ubuntu/train/train_mandarin/mandarin_000{0}.wav'.format(i)
    else:
        path = '/home/ubuntu/train/train_mandarin/mandarin_00{0}.wav'.format(i)
    y, sr = librosa.load(path, sr=16000)
    y, index = librosa.effects.trim(y)
    mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.0025), hop_length = int(sr*0.01))
    mat = mat[:,:58800].T
    mat = mat.reshape((98,600,64))
    mat = np.concatenate((mat, label), axis=-1)
    data = np.concatenate((data, mat))

label = np.array([0,0,1]*58800)
label = label.reshape((98,600,3))
for i in range(1,31):#31
    if i<10:

        path = '/home/ubuntu/train/train_hindi/hindi_000{0}.wav'.format(i)
    else:
        path = '/home/ubuntu/train/train_hindi/hindi_00{0}.wav'.format(i)
    y, sr = librosa.load(path, sr=16000)
    y, index = librosa.effects.trim(y)
    mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.0025), hop_length = int(sr*0.01))
    mat = mat[:,:58800].T
    mat = mat.reshape((98,600,64))
    mat = np.concatenate((mat, label), axis=-1)
    data = np.concatenate((data, mat))

hf = h5py.File('data.h5', 'w')
np.random.shuffle(data)
hf.create_dataset('language', data=data)
hf.close()

# # sorted_indices = [12,32,61,901]
# # with h5py.File(features_file, 'r') as hf:
# #     nonseqential_of_my_data = hf['my_data'][sorted_indices]
