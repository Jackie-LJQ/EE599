import tensorflow as tf
import os, json, random
import os.path as osp
from utils import Config

class polyvore_dataset:
    def __init__(self, train=True):
        self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.Train=train
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
        image1 = self.decodeImg(x[0])
        image1 = tf.image.resize(image1, (300, 300))
        image1 = tf.image.central_crop(image1, 224 / 300)
        print(tf.shape(image1))
        image2 = self.decodeImg(x[1])
        image2 = tf.image.resize(image2, (300, 300))
        image2 = tf.image.central_crop(image2, 224 / 300)
        image1 = 2*image1 - 1
        image2 = 2*image2 - 1  # value in [-1,1]
        if self.Train:
            return (image1,image2), tf.strings.to_number(x[2])
        else:
            return image1,image2

    def load_test(self, fileList):
        data = [self.process(path) for path in fileList]
        image1 = [row[0] for row in data]
        image2 = [row[1] for row in data]
        return image1,image2
    '''
    load in data in a streaming fashion
    '''
    def load(self, fileList, batchSize=32):
        data = tf.data.Dataset.from_tensor_slices(fileList)
        # data = data.cache()  #

        data = data.map(self.process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.batch(batchSize)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return data



    '''
    read in meta info
    '''
    def readCompat(self, filename):
        set_id = {}
        if self.Train:
            meta = open(os.path.join(self.root_dir, 'train.json'), 'r')
            meta = json.load(meta)
            for Set in meta:
                item_id = []
                for Items in Set['items']:
                    id = Items['item_id']
                    item_id.append(id)
                set_id[Set['set_id']] = item_id
            Compatfile = open(filename, 'r')
            pair_label = []
            for line in Compatfile:
                fields = line.split()
                setID, _ = fields[1].split('_')
                items = set_id[setID]
                label = fields[0]
                n = len(items)
                for i in range(n):
                    for j in range(i+1,n):
                        itemID = []
                        row=osp.join(self.image_dir,items[i])+'.jpg'+';'+osp.join(self.image_dir,items[j])+'.jpg'+';'+label
                        pair_label.append(row)
            random.shuffle(pair_label)
        # idx = 1 - int(len(pair_label) * 0.15) # 80/20 split
            idx = 100000
            ending = int(idx*1.15)
            return pair_label[:idx], pair_label[idx:ending]
        else:
            Compatfile = open(filename, 'r')
            pair = []
            pair_id = []
            for line in Compatfile:
                item1, item2 = line.split()
                row1=osp.join(self.image_dir,item1)+'.jpg'+';'+osp.join(self.image_dir,item2)+'.jpg'
                row2=item1+' '+item2
                pair.append(row1)
                pair_id.append(row2)
            return pair, pair_id
