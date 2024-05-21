import pandas as pd
import os
# from skimage.transform import resize
# from skimage.io import imread
import numpy as np
from google.cloud import storage
import requests
from io import BytesIO
import random
from detection.params import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import time
import pickle
from itertools import chain


def preprocess_data():

    # generate training,testing and validation batches

    # train_dir = TRAIN_DATA_PATH
    # test_dir = TEST_DATA_PATH
    # val_dir = VAL_DATA_PATH

    train_dir = TRAIN_DATA_PATH_CLOUD
    test_dir = TEST_DATA_PATH_CLOUD
    val_dir = VAL_DATA_PATH_CLOUD

    # def random_crop(image):
    #     height, width = image.shape[:2]
    #     random_array = np.random.random(size=4);
    #     w = int((width*0.5) * (1+random_array[0]*0.5))
    #     h = int((height*0.5) * (1+random_array[1]*0.5))
    #     x = int(random_array[2] * (width-w))
    #     y = int(random_array[3] * (height-h))

    #     image_crop = image[y:h+y, x:w+x, 0:3]
    #     image_crop = image.resize(image_crop, image.shape)
    #     return image_crop

    #dgen_train_norm = ImageDataGenerator(rescale = 1./255)
    dgen_train = ImageDataGenerator(rescale = 1./255,
                                    shear_range=0.2,
                                    zoom_range = 0.2,
                                    channel_shift_range = 20)
    dgen_validation = ImageDataGenerator(rescale = 1./255)
    dgen_test = ImageDataGenerator(rescale=1./255)

    train_generator = dgen_train.flow_from_directory(train_dir,
                                                    target_size=(150,150),
                                                    subset = "training",
                                                    batch_size = 32,
                                                    class_mode = "categorical")

    # train_generator_other = dgen_train_other.flow_from_directory(train_dir_other,
    #                                                 target_size=(150,150),
    #                                                 subset = "training",
    #                                                 batch_size = 32,
    #                                                 class_mode = "categorical")

    validation_generator = dgen_validation.flow_from_directory(val_dir,
                                                    target_size=(150,150),
                                                    batch_size = 32,
                                                    class_mode = "categorical")

    test_generator = dgen_test.flow_from_directory(test_dir,
                                                    target_size=(150,150),
                                                    batch_size = 32,
                                                    class_mode = "categorical")

    #train_merged_generator = chain(train_generator_norm, train_generator_other)

    return train_generator, validation_generator, test_generator

def preprocess_features(x):
    """
    Preprocess new images to predict.
    Input - binary by api query.
    Output - Generator
    """
    stream = BytesIO(x)
    img = Image.open(stream)
    image = img.resize((150,150))
    array = img_to_array(image)
    tensor = tf.expand_dims(array, axis=0)
    #print(f"tensor: {tensor}")
    return tensor
