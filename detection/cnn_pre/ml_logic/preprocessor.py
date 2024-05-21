"""
Preprocessing data using generators.
"""

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
from tensorflow.keras.applications.efficientnet import preprocess_input
import time
import pickle


def preprocess_data():
    """
    The function make generations for training, testing and validation by global variables.
    Images are rescaled and augmentaded (zoom and horizontal flip)
    """

    # Define directories by global variabels:
    # train_dir = TRAIN_DATA_PATH_CLOUD
    # test_dir = TEST_DATA_PATH_CLOUD

    train_dir_normal = TRAIN_DATA_PATH
    train_dir_no_normal = TRAIN_DATA_PATH
    test_dir = TEST_DATA_PATH

    # Create a generator with augmentation for training and validation:
    dgen_train_normal = ImageDataGenerator(
                                    rescale = 1./255,
                                    validation_split=0.2,
                                    #shear_range=0.2,
                                    #zoom_range = 0.2,
                                    #horizontal_flip = False,
                                    #preprocessing_function = preprocess_input
                                    )
    dgen_train_others = ImageDataGenerator(
                                    rescale = 1./255,
                                    validation_split=0.2,
                                    shear_range=0.2,
                                    zoom_range = 0.2,
                                    #horizontal_flip = False,
                                    )

    # datagen = ImageDataGenerator(
    #     featurewise_center = False,
    #     featurewise_std_normalization = False,
    #     rotation_range = 10,
    #     width_shift_range = 0.1,
    #     height_shift_range = 0.1,
    #     horizontal_flip = True,
    #     zoom_range = (0.8, 1.2),
    #     )

#     model_2.add(layers.RandomFlip("horizontal"))
# model_2.add(layers.RandomZoom(0.1))
# model_2.add(layers.RandomTranslation(0.2, 0.2))
# model_2.add(layers.RandomRotation(0.1))

    # Create a generator without augmentation for test:
    #dgen_test = ImageDataGenerator(rescale=1./255)
    dgen_test = ImageDataGenerator(rescale = 1./255)

    # Make generators by directories:
    # The classes wiil be the subdirectories
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
