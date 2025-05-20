import os
import PIL
from PIL import Image
import numpy as np
import pandas as pd
import pickle
import tarfile
from tqdm import tqdm_notebook as tqdm
tqdm().pandas()

from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input

main_dir = "/content/gdrive/My Drive/data/lung_cancer/"
tar = tarfile.open(os.path.join(main_dir, "Files", "images_001.tar.gz"))
model = DenseNet121(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')

def extract_save_features_001():
    model = DenseNet121(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
    tar = tarfile.open(os.path.join(main_dir, "Files", "images_001.tar.gz"))
    features = {}
    for member in tar.getmembers():
        if len(member.name.split('/'))==2:
            img = member.name.split('/')[1]
            image = Image.open(tar.extractfile(member))
            image = image.convert('RGB')
            image = image.resize((224,224))
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            image = image/255
        #    image = image - 1.0
            # image = image.flatten()
            # print(image.shape)
            feature = model.predict(image)
            features[img] = feature

            feature_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics/svm", timestamp + ".pickle")
            with open(metrics_path, "wb") as file:
                pickle.dump(metrics, file)
    return features
