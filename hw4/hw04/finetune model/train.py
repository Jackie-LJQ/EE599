from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from dataloader import polyvore_dataset
from utils import Config
from tensorflow.keras.applications import MobileNet
import tensorflow.keras as tfk
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

if __name__=='__main__':

    # data generators
    dataset = polyvore_dataset()
    trainList, valList, nClass = dataset.readMeta() #read fig name and corresponding label
    if Config['debug']:
        trainList = trainList[:500]
        valList = valList[:100]
    trainData = dataset.load(trainList, batchSize=Config['batch_size'])
    valData = dataset.load(valList, batchSize=Config['batch_size'])
    
    reg_val = 0.01
    # drop_rate = 0.1
    # # build model
    base_model = MobileNet(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dropout(drop_rate)(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l=reg_val))(x)
    # x = Dense(256, activation='relu')(x)
    # x = Dropout(drop_rate)(x)
    predictions = Dense(nClass, activation = 'softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

    optimizer = tfk.optimizers.RMSprop(Config['learning_rate'])
    # define optimizers
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    result = model.fit(trainData, validation_data=valData, epochs=Config['num_epochs'],shuffle=False)
    
    
# =============================================================================
# use callback optimize lr    
# =============================================================================
    #def decay(epoch):
    #  if epoch < 3:
    #    return 1e-3
    #  elif epoch >= 3 and epoch < 7:
    #    return 1e-4
    #  else:
    #    return 1e-5
    #callbacks = [tf.keras.callbacks.LearningRateScheduler(decay),PrintLR()]
#result = model.fit(trainData, validation_data=valData, epochs=Config['num_epochs'],shuffle=False,callbacks=callbacks)
    
# =============================================================================
#plot and save
# =============================================================================
    model.save('fine_tunemodel.h5')
    plot_model(model, to_file='model.png')
     
    acc = result.history['accuracy']
    val_acc = result.history['val_accuracy']
    epochs = np.arange(len(acc))
     
    plt.figure()
    plt.plot(epochs, acc, label='acc')
    plt.plot(epochs, val_acc, label='val_acc')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve-finetune')
    plt.legend()
    plt.savefig("finetune_curve.png")
    plt.show()
#    from tensorflow.keras.models import load_model
#    model = load_model('fine_tunemodel.h5')
#    
    