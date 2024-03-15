import numpy as np
import time
from colorama import Fore, Style
from typing import Tuple

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.efficientnet import EfficientNetB7


end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")

def initialize_model(input_shape: tuple = (150,150,3)) -> Model:
    """
    Initialize the Neural Network with random weights
    """

    eff_model = EfficientNetB7(input_shape = input_shape, include_top = False, weights = 'imagenet')

    for layer in eff_model.layers:
      layer.trainable = False

    model = Sequential()
    model.add(eff_model)
    model.add(MaxPooling2D(2))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(4, activation='softmax'))

    print("✅ Model initialized")
    print(model.summary)

    return model


def compile_model(model: Model, learning_rate) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    print("✅ Model compiled")

    return model

def train_model(
        model: Model,
        train_data,
        batch_size=16,
        patience=5,
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
