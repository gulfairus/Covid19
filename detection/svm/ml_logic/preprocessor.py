import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from google.cloud import storage
import requests
from io import BytesIO
import random
from detection.params import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory

def preprocess_data():
    # train_dir = "/home/user/code/gulfairus/Covid19/raw_data/cloud/train_test/train"
    # test_dir = "/home/user/code/gulfairus/Covid19/raw_data/cloud/train_test/test"

    # train_datagen = ImageDataGenerator(rescale=1. / 255, dtype='float32')
    # itr = train_datagen.flow_from_directory(
    # train_dir,
    # target_size=(64, 64),
    # batch_size=20227,
    # class_mode='categorical')

    # X_train, y_train = itr.next()
    # X_train = X_train.reshape(20227, 12288)
    # df_train = pd.DataFrame(X_train)
    # df_train['target'] = y_train

    # return df_train.shape

    main_dir = "/home/user/code/gulfairus/Covid19/raw_data/cloud/COVID-19_Radiography_Dataset"
    train_dir = "/home/user/code/gulfairus/Covid19/raw_data/cloud/train_test/train"
    test_dir = "/home/user/code/gulfairus/Covid19/raw_data/cloud/train_test/test"
    Categories=['COVID19','NORMAL','OPACITY','PNEUMONIA']
    flat_data_arr=[] #input array
    target_arr=[] #output array
    datadir='IMAGES/'
    #df_array = np.zeros((20227, 30000))
    #path which contains all the categories of images
    for i in Categories[:1]:

        print(f'loading... category : {i}')
        path=os.path.join(main_dir,i)
        k=0
        while k<5:
            for img in os.listdir(path):
                img_array=imread(os.path.join(path,img)).astype('float32')
                img_resized=resize(img_array,(64,64,3))
                #img = tf.keras.utils.array_to_img(img)
                #array = tf.keras.utils.image.img_to_array(img)
                flat_data_arr.append(img_resized.flatten())
                target_arr.append(Categories.index(i))
                k+=1
            print(f'loaded category:{i} successfully')
    flat_data=np.array(flat_data_arr)
    target=np.array(target_arr)

    return flat_data

a = preprocess_data()
print(a.shape)
