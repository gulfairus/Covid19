import numpy as np
import time
from colorama import Fore, Style
from typing import Tuple
import efficientnet.keras as efn

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
from tensorflow.keras.regularizers import l2

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")



def initialize_model(input_shape: tuple = (150,150,3)) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    preproc_layer = preprocess_input()
    base_model = EfficientNetB7(input_shape = input_shape, include_top = False, weights = 'imagenet')
    base_model.trainable = False
    flatten_layer = layers.Flatten()
    dense_layer = layers.Dense(512, activation='relu')
    dropout_layer = layers.Dropout(0.2)
    prediction_layer = layers.Dense(4, activation='softmax')

    model = models.Sequential([
        preproc_layer,
        base_model,
        flatten_layer,
        dense_layer,
        dropout_layer,
        prediction_layer
    ])

    print("✅ Model initialized")
    print(model.summary)

    return model


def compile_model(model: Model, learning_rate) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="squared_hinge", optimizer=optimizer, metrics=["accuracy"])

    print("✅ Model compiled")

    return model

def train_model(
        model: Model,
        train_data,
        batch_size,
        patience,
        validation_data=None,
        epochs=None) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )


    print(f"✅ Model trained on rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history


def evaluate_model(
        model: Model,
        test_data,
        batch_size=32
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        test_data = test_data,
        batch_size=batch_size,
        verbose=0,
        callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"✅ Model evaluated, Accuracy: {round(accuracy, 2)}")

    return metrics
