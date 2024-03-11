import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from PIL import Image
from google.cloud import storage
import requests
from io import BytesIO
import random
from detection.params import *

from detection.svm.ml_logic.data import load_data_to_bq
#from detection.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from detection.svm.ml_logic.preprocessor import preprocess_data
#from detection.ml_logic.registry import load_model, save_model, save_results


def preprocess() -> None:
    #storage_client = storage.Client(GCP_PROJECT)
    #bucket = storage_client.get_bucket(BUCKET_NAME)


    df_processed = preprocess_data()

    load_data_to_bq(
        df_processed,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'df_processed',
        truncate=True
    )

    print("✅ preprocess() done \n")
