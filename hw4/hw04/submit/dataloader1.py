import tensorflow as tf
import os, json, random
import os.path as osp
from utils import Config

class polyvore_dataset:
    def __init__(self, train=True):
        self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.Train = train
    '''
    decode one image
    '''
    def decodeImg(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image
    '''
    decode file name and return the raw image
    '''
    def process(self, pathAndLabel):
    
        x = tf.strings.split(pathAndLabel,';')
        image = self.decodeImg(x[0])
        
        image = tf.image.resize(image, (300, 300))
        image = tf.image.central_crop(image, 224 / 300)
        image = 2*image - 1  # value in [-1,1]
        
        return image, tf.strings.to_number(x[1])
        
    def process_test(self, pathAndLabel):
        image = self.decodeImg(pathAndLabel)
        image = tf.image.resize(image, (300, 300))
        image = tf.image.central_crop(image, 224 / 300)
        image = 2*image - 1  # value in [-1,1]
        
        return image
    '''
    load in data in a streaming fashion
    '''
    def load(self, fileList, batchSize=32):
        data = tf.data.Dataset.from_tensor_slices(fileList)
        data = data.map(self.process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # data = data.cache()  #
        data = data.batch(batchSize)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return data
        
    def load_test(self, fileList):
        data = tf.data.Dataset.from_tensor_slices(fileList)
        data = data.map(self.process_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.batch(1)
        return data

    '''
    read in meta info
    '''
    def readMeta(self):
        dictionary = {'max': 0}
        def translate(idOld):
            if idOld not in dictionary:
                dictionary[idOld] = dictionary['max']
                dictionary['max'] = dictionary['max'] + 1
            return str(dictionary[idOld])
        meta = open(os.path.join(self.root_dir, 'polyvore_item_metadata.json'), 'r')
        meta = json.load(meta)
        nameAndId = [os.path.join(self.image_dir, name + '.jpg' + ';' + translate(label['category_id'])) for name, label in
                     meta.items()]
        random.shuffle(nameAndId)
        idx = 1 - int(len(nameAndId) * 0.2) # 80/20 split
        return nameAndId[:idx], nameAndId[idx:], dictionary['max']+1
