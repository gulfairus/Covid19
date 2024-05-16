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
import time
import pickle


def preprocess_data():

    # generate training,testing and validation batches
    train_dir = TRAIN_DATA_PATH_CLOUD
    test_dir = TEST_DATA_PATH_CLOUD

    dgen_train = ImageDataGenerator(#samplewise_center=True,
                                    #samplewise_std_normalization=True,
                                    rescale = 1./255,
                                    validation_split=0.2,
                                    #rotation_range=20,
                                    shear_range=0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = False)
    dgen_validation = ImageDataGenerator(rescale = 1./255)
    dgen_test = ImageDataGenerator(rescale=1./255)

    train_generator = dgen_train.flow_from_directory(train_dir,
                                                    target_size=(150,150),
                                                    subset = "training",
                                                    batch_size = 32,
                                                    class_mode = "categorical")

    validation_generator = dgen_train.flow_from_directory(train_dir,
                                                    target_size=(150,150),
                                                    subset = "validation",
                                                    batch_size = 32,
                                                    class_mode = "categorical")

    test_generator = dgen_test.flow_from_directory(test_dir,
                                                    target_size=(150,150),
                                                    batch_size = 32,
                                                    class_mode = "categorical")


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
