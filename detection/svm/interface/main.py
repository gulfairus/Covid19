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
from detection.svm.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from detection.svm.ml_logic.preprocessor import preprocess_data
from detection.svm.ml_logic.registry import load_model, save_model, save_results
from detection.svm.ml_logic.registry import mlflow_run, mlflow_transition_model

def preprocess() -> None:
    #storage_client = storage.Client(GCP_PROJECT)
    #bucket = storage_client.get_bucket(BUCKET_NAME)


    # load_data_to_bq(
    #     df_processed,
    #     gcp_project=GCP_PROJECT,
    #     bq_dataset=BQ_DATASET,
    #     table=f'df_processed',
    #     truncate=True
    # )

    print("✅ preprocess() done \n")
    return

#@mlflow_run
def train(
        learning_rate=0.0005,
        batch_size = 32,
        patience = 2,
        epochs=5
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\generating data..." + Style.RESET_ALL)


    train_generator, validation_generator, test_generator = preprocess_data()


    # Train model using `model.py`
    model = load_model()

    if model is None:
        model = initialize_model(input_shape=(150,150,3))

    model = compile_model(model, learning_rate=0.0001)
    model, history = train_model(
        model, train_data=train_generator, batch_size=batch_size,
        patience=patience,validation_data=validation_generator, epochs=epochs
    )

    val_accuracy = np.min(history.history['accuracy'])

    params = dict(
        context="train",
        #training_set_size=DATA_SIZE,
        #row_count=len(X_train_processed),
    )

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(accuracy=val_accuracy))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    # The latest model should be moved to staging
    # $CHA_BEGIN
    if MODEL_TARGET == 'mlflow':
        mlflow_transition_model(current_stage="None", new_stage="Staging")
    # $CHA_END

    print("✅ train() done \n")

    return val_accuracy


@mlflow_run
def evaluate(
        # min_date:str = '2014-01-01',
        # max_date:str = '2015-01-01',
        stage: str = "Production"
    ) -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return MAE as a float
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    model = load_model(stage=stage)
    assert model is not None

    train_generator, validation_generator, test_generator = preprocess_data()

    metrics_dict = evaluate_model(model=model, test_data=test_generator)
    accuracy = metrics_dict["accuracy"]

    params = dict(
        context="evaluate", # Package behavior
        #training_set_size=DATA_SIZE,
        #row_count=len(X_new)
    )

    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")

    return accuracy


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """
    from google.colab import files
    from keras.preprocessing import image
    uploaded = files.upload()
    print("\n⭐️ Use case: predict")


    model = load_model()
    assert model is not None

    for filename in uploaded.keys():
        img_path = os.getcwd+filename
        img = image.load_img(img_path, target_size=(150,150))
        images = image.img_to_array(img)
        images = np.expand_dims(images, axis=0)
        prediction = model.predict(images)




    print("\n✅ prediction done: ", prediction, prediction.shape, "\n")
    return prediction


if __name__ == '__main__':
    preprocess()
    train()
    evaluate()
    pred()
