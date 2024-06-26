import numpy as np
import time
from colorama import Fore, Style
from typing import Tuple
from detection.params import *

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model, load_model
from vit_keras import vit

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")



def initialize_model(input_shape: tuple) -> Model:
    """
    # Initialize the Neural Network with random weights
    """
    vit_model = vit.vit_b16(image_size=224,
                            #image_size=input_shape,
                         activation='relu',
                         pretrained=True,
                         include_top=True,
                         pretrained_top=False,
                         classes=4)

    data_augmentation = Sequential(
        [
            layers.experimental.preprocessing.Normalization(),
            layers.experimental.preprocessing.Resizing(224, 224)
        ],
        name="data_augmentation",
    )
    # for layer in vit_model.layers:
    #     layer.trainable = False
    inputs = Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    vit_model.trainable = False
    x = vit_model(augmented)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(4, activation='softmax'))
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(4, activation='softmax')(x)
    model = Model(inputs=inputs,outputs=outputs)

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


    print(f"✅ Model trained ")

    return model, history


def evaluate_model(
        model: Model,
        test_data,
        batch_size,
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model..." + Style.RESET_ALL)

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

    print(f"✅ Model evaluated")

    return metrics
