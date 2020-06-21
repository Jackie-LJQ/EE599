from tensorflow.keras.layers import AveragePooling2D, Dense, Conv2D, MaxPool2D, Flatten, Input,BatchNormalization, DepthwiseConv2D, Flatten
from tensorflow.keras.models import Model
from dataloader1 import polyvore_dataset
from utils import Config
import tensorflow.keras as tfk
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
if __name__=='__main__':
    dataset = polyvore_dataset(train='False')
    filepath = Config['test_file_path']
    f = open(filepath, "r")
    itemidlist=[]
    for x in f:
        itemidlist.append(x[:-1])
    # itemidlist = itemidlist[:2]
    image_dir = osp.join(Config['root_path'], 'images')
    testList = [osp.join(image_dir, x + '.jpg') for x in itemidlist]
    testdata = dataset.load_test(testList)
    from tensorflow.keras.models import load_model
    model = load_model('categorical_model_owndesign.h5')
    temp = model.predict(testdata)
    prediction_ownmodel = [np.argmax(i) for i in temp]
    results = np.vstack((itemidlist,prediction_ownmodel))
    results = results.T
    f1 = open('category_ownmodel_pred.txt', 'w')
    for row in results:
        temp = row[0]+' ' + row[1] + '\r\n'
        f1.write(temp)
    f1.close()

    model = load_model('categorical_model_finetune.h5')
    temp = model.predict(testdata)
    prediction_finetune = [np.argmax(i) for i in temp]
    results = np.vstack((itemidlist,prediction_finetune))
    results = results.T
    f2 = open('category_finetune_pred.txt', 'w')
    for row in results:
        temp = row[0]+' ' + row[1] + '\r\n'
        f2.write(temp)
    f2.close()
