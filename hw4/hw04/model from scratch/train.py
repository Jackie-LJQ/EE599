from tensorflow.keras.layers import AveragePooling2D, Dense, Conv2D, MaxPool2D, Input,BatchNormalization, DepthwiseConv2D, Flatten
from tensorflow.keras.models import Model
from dataloader import polyvore_dataset
from utils import Config
import tensorflow.keras as tfk
import matplotlib.pyplot as plt
import os.path as osp
from tensorflow.keras.models import load_model
import numpy as np
from tf.keras.utils import plot_model

if __name__=='__main__':

    # data generators
     dataset = polyvore_dataset()
     trainList, valList, nClass = dataset.readMeta()
     if Config['debug']:
         trainList = trainList[:500]
         valList = valList[:100]
     trainData = dataset.load(trainList, batchSize=Config['batch_size'])
     valData = dataset.load(valList, batchSize=Config['batch_size'])

# =============================================================================
# bulid model
# =============================================================================
     reg_val = 0.01
     drop_rate = 0.15
     x_input = Input(shape=(224, 224, 3), name='input_image')
     x = Conv2D(32, (3,3),activation='relu', name='cov1')(x_input)
     x = BatchNormalization()(x)
#     depthwise + pointwise 
     x = DepthwiseConv2D((3,3), name='dw1')(x)
     x = Conv2D(64, (1,1),activation='relu', name='pw1')(x)
     x = BatchNormalization()(x)
     x = MaxPool2D()(x)

     x = DepthwiseConv2D((3,3),name='dw2')(x)
     x = Conv2D(128, (1,1),activation='relu', name='pw2')(x)
     x = BatchNormalization()(x)
     x = MaxPool2D()(x)

     x = DepthwiseConv2D((3,3),name='dw3')(x)
     x = Conv2D(256, (1,1),activation='relu', name='pw3')(x)
     x = BatchNormalization()(x)
     x = MaxPool2D()(x)

     x = DepthwiseConv2D((3,3),name='dw4')(x)
     x = Conv2D(512, (1,1),activation='relu', name='pw4')(x)
     x = BatchNormalization()(x)
     x = MaxPool2D()(x)

     x = AveragePooling2D()(x)
     x = Flatten()(x)

     predictions = Dense(nClass, activation = 'softmax', name = 'predictions')(x)

     model = Model(x_input, predictions)
     # define optimizers
     optimizer = tfk.optimizers.RMSprop(learning_rate=Config['learning_rate'])
     model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
     model.summary()

     result = model.fit(trainData, validation_data=valData, epochs=Config['num_epochs'],shuffle=False)

# =============================================================================
# save and plot
# =============================================================================
     model.save('ownmodel.h5')
     acc = result.history['accuracy']
     val_acc = result.history['val_accuracy']
     epochs = np.arange(len(acc))
     
     plt.figure()
     plt.plot(epochs, acc, label='loss')
     plt.plot(epochs, val_acc, label='val_loss')
     plt.xlabel('epochs')
     plt.ylabel('Accuracy')
     plt.title('Learning Curve-own model')
     plt.legend()
     plt.savefig("own_model.png")
     plt.show()
     
     model = load_model('ownmodel.h5')
     plot_model(model, to_file='model.png')
# =============================================================================
# test model
# =============================================================================
     dataset = polyvore_dataset(train='False')
     filepath = osp.join(Config['root_path'], 'test_category_hw.txt')
     f = open(filepath, "r")
     itemidlist=[]
     for x in f:
         itemidlist.append(x[:-1])
     image_dir = osp.join(Config['root_path'], 'images')
     testList = [osp.join(image_dir, x + '.jpg') for x in itemidlist]
     testdata = dataset.load(testList)
     model = load_model('ownmodel.h5')
     temp = model.predict(testdata)
     prediction_ownmodel = [np.argmax(i) for i in temp]
     results = np.vstack((itemidlist,prediction_ownmodel))
     results = results.T
     f = open('ownmodel_pred.txt', 'w')
     for row in results:
         temp = row[0]+' ' + row[1] + '\r\n'
         f.write(temp)
# =============================================================================
# test finetune model        
# =============================================================================
    # model = load_model('fine_tunemodel.h5')
    # temp = model.predict(testdata)
    # prediction_finetune = [np.argmax(i) for i in temp]
    # results = np.vstack((itemidlist,prediction_ownmodel))
    # results = results.T
    # f = open('finetune_pred.txt', 'w')
    # for row in results:
    #     temp = row[0]+' ' + row[1] + '\r\n'
    #     f.write(temp)