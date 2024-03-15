"""
Preprocessing data using generators.
"""

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
    """
    The function make generations for training, testing and validation by global variables.
    Images are rescaled and augmentaded (zoom and horizontal flip)
    """

    # Define directories by global variabels:
    train_dir = TRAIN_DATA_PATH
    test_dir = TEST_DATA_PATH

    # Create a generator with augmentation for training and validation:
    dgen_train = ImageDataGenerator(rescale = 1./255,
                                    validation_split=0.2,
                                    shear_range=0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = False)

    # Create a generator without augmentation for test:
    dgen_test = ImageDataGenerator(rescale=1./255)

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
