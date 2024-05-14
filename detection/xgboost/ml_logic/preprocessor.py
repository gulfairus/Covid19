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
import time
import pickle


def preprocess_data():

    # generate training,testing and validation batches
    train_dir = TRAIN_DATA_PATH
    test_dir = TEST_DATA_PATH

    Categories=['COVID19','NORMAL', 'OPACITY', 'PNEUMONIA']
    flat_data_arr=[] #input array
    target_arr=[] #output array
    datadir='IMAGES/'
    #path which contains all the categories of images
    for i in Categories:

        print(f'loading... category : {i}')
        path1=os.path.join(train_dir,i)
        for img in os.listdir(path1):
            img_array=imread(os.path.join(path1,img))
            img_resized=resize(img_array,(150,150,3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(Categories.index(i))
        print(f'loaded category:{i} successfully')
    train_data=pd.DataFrame(flat_data_arr)
    train_data['target']=target_arr

    for i in Categories:

        print(f'loading... category : {i}')
        path2=os.path.join(test_dir,i)
        for img in os.listdir(path2):
            img_array=imread(os.path.join(path2,img))
            img_resized=resize(img_array,(150,150,3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(Categories.index(i))
        print(f'loaded category:{i} successfully')
    test_data=pd.DataFrame(flat_data_arr)
    test_data['target']=target_arr

    return train_data, test_data




    # dgen_train = ImageDataGenerator(#samplewise_center=True,
    #                                 #samplewise_std_normalization=True,
    #                                 rescale = 1./255,
    #                                 validation_split=0.2,
    #                                 #rotation_range=20,
    #                                 shear_range=0.2,
    #                                 zoom_range = 0.2,
    #                                 horizontal_flip = False)
    # dgen_validation = ImageDataGenerator(rescale = 1./255)
    # dgen_test = ImageDataGenerator(rescale=1./255)

    # train_generator = dgen_train.flow_from_directory(train_dir,
    #                                                 target_size=(150,150),
    #                                                 subset = "training",
    #                                                 batch_size = 32,
    #                                                 class_mode = "categorical")

    # validation_generator = dgen_train.flow_from_directory(train_dir,
    #                                                 target_size=(150,150),
    #                                                 subset = "validation",
    #                                                 batch_size = 32,
    #                                                 class_mode = "categorical")

    # test_generator = dgen_test.flow_from_directory(test_dir,
    #                                                 target_size=(150,150),
    #                                                 batch_size = 32,
    #                                                 class_mode = "categorical")


    # return train_generator, validation_generator, test_generator
