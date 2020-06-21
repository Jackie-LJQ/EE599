from dataloader3 import polyvore_dataset
from utils import Config
import numpy as np
from tensorflow.keras.models import load_model
if __name__=='__main__':
    model = load_model('compatible_model.h5')
    
    dataset = polyvore_dataset(train=False)
    testList, pair_id, item_num = dataset.readCompat(Config['test_file_path']) #item_num is the num of items in a set
    # testList = testList[:16]
    # pair_id = pair_id[:2]
    # item_num = item_num[:2]
    image1, image2 = dataset.load_test(testList)
    temp = model.predict([image1, image2])
    prediction = []
    i, j = 0,0
    while i < len(temp):
        num = int(item_num[j]*(item_num[j]-1)/2)-1
        mean = np.average(temp[i:i+num])
        prediction.append(mean)
        i=i+num+1
        j+=1
     
    results = np.vstack((pair_id, prediction))
    results = results.T
    f = open('bonus.txt', 'w')
    for row in results:
        temp = row[0] +' '+row[1]+'\n'
        f.write(temp)
    f.close()