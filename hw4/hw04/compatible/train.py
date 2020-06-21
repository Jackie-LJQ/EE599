from tensorflow.keras.layers import Dense, concatenate, Input, Conv2D, Flatten,DepthwiseConv2D,MaxPool2D,BatchNormalization
from tensorflow.keras.models import Model
from dataloader import polyvore_dataset
from utils import Config
import tensorflow.keras as tfk
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

# =============================================================================
# bulid model
# =============================================================================
if __name__=='__main__':

#     # data generators
    dataset = polyvore_dataset()
    trainList, valList = dataset.readCompat('compatibility_train.txt')
    nClass = 2
    if Config['debug']:
        trainList = trainList[:500]
        valList = valList[:100]
    trainData = dataset.load(trainList, batchSize=Config['batch_size'])
    valData = dataset.load(valList, batchSize=Config['batch_size'])
    reg_val = 0.01
    rop_rate = 0.1     
    x1_input = Input(shape=(224, 224, 3), name='input_image1')
    x1 = Conv2D(32, (3,3),activation='relu')(x1_input)
    x1 = BatchNormalization()(x1)
    x1 = DepthwiseConv2D((3,3))(x1)
    x1 = Conv2D(64, (1,1),activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool2D()(x1)

    x1 = DepthwiseConv2D((3,3))(x1)
    x1 = Conv2D(128, (1,1),activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool2D()(x1)

    x1 = DepthwiseConv2D((3,3))(x1)
    x1 = Conv2D(256, (1,1),activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool2D()(x1)

    x1 = DepthwiseConv2D((3,3))(x1)
    x1 = Conv2D(512, (1,1),activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool2D()(x1)

    first = Flatten()(x1)
    
    
    
    x2_input = Input(shape=(224, 224, 3), name='input_image2')
    x2 = Conv2D(32, (3,3),activation='relu')(x2_input)
    x2 = BatchNormalization()(x2)
    x2 = DepthwiseConv2D((3,3), name='dw1')(x2)
    x2 = Conv2D(64, (1,1),activation='relu', name='pw1')(x2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool2D()(x2)

    x2 = DepthwiseConv2D((3,3),name='dw2')(x2)
    x2 = Conv2D(128, (1,1),activation='relu', name='pw2')(x2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool2D()(x2)

    x2 = DepthwiseConv2D((3,3),name='dw3')(x2)
    x2 = Conv2D(256, (1,1),activation='relu', name='pw3')(x2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool2D()(x2)

    x2 = DepthwiseConv2D((3,3),name='dw4')(x2)
    x2 = Conv2D(512, (1,1),activation='relu', name='pw4')(x2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool2D()(x2)

    second = Flatten()(x2)
    merge_one = concatenate([first,second], name='merge')
    # merge = Dense(128, activation='relu')(merge_one)
    merge = Dense(64, activation='relu')(merge_one)
    predictions = Dense(1, activation = 'sigmoid')(merge)
    model = Model(inputs=[x1_input, x2_input], outputs=predictions)
    optimizer = tfk.optimizers.RMSprop(learning_rate=Config['learning_rate'])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    # training - num worker is obsolete now
    result = model.fit(trainData, validation_data=valData, epochs=Config['num_epochs'],shuffle=False)
# =============================================================================
# plot and save model
# =============================================================================
    model.save('model_compatible.h5')
    acc = result.history['accuracy']
    val_acc = result.history['val_accuracy']
    epochs = np.arange(len(acc))

    plt.figure()
    plt.plot(epochs, acc, label='acc')
    plt.plot(epochs, val_acc, label='val_acc')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve-compatible')
    plt.legend()
    plt.savefig("compatible.png")
    plt.show()
    
    from tensorflow.keras.models import load_model
    model = load_model('model_compatible.h5')
    plot_model(model, to_file='pairwise_model.png')
    
    dataset = polyvore_dataset(train=False)
    testList, testList2 = dataset.readCompat('compatibility_test_hw.txt')
    testList.append(testList2)
    testList = testList[:100]
    # testList = testList[:2]
    image1, image2 = dataset.load(testList)
    temp = model.predict([image1, image2])
    prediction = [int(i>0.5) for i in temp]
    
    List  = []
    f = open('compatibility_test_hw.txt', 'r')
    for row in f:
        List.append(row[:-1])
     
    results = np.vstack((List, prediction))
    result = result.T
    f = open('compatible_pred.txt', 'w')
    for row in results:
        temp = row[0]+' ' + row[1] + '\r\n'
        f.write(temp)