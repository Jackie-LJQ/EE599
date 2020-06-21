from tensorflow.keras.layers import Dense, concatenate, Input, Conv2D, Flatten,DepthwiseConv2D,MaxPool2D,BatchNormalization
from tensorflow.keras.models import Model
from dataloader2 import polyvore_dataset
from utils import Config
import tensorflow.keras as tfk
import numpy as np
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

if __name__=='__main__':
    model = load_model('compatible_model.h5')
    
    dataset = polyvore_dataset(train=False)
    testList, pair_id = dataset.readCompat(Config['test_file_path'])
    # testList = testList[:80]
    # pair_id = pair_id[:80]
    image1, image2 = dataset.load_test(testList)
    temp = model.predict([image1, image2])
    prediction = [int(i>0.5) for i in temp]
    
     
    results = np.vstack((pair_id, prediction))
    results = results.T
    f = open('compatible_pred.txt', 'w')
    for row in results:
        temp = row[0] +' '+row[1]+'\n'
        f.write(temp)
    f.close()