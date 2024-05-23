import numpy as np
import time
from colorama import Fore, Style
from typing import Tuple

# Timing the TF import
# print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
# start = time.perf_counter()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
from tensorflow.keras.models import Model, load_model


# end = time.perf_counter()
# print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")



def initialize_model(input_shape) -> Model:
    """
    Initialize the Neural Network with random weights
    """

    # eff_model = EfficientNetB7(input_shape = input_shape, include_top = False, weights = 'imagenet')


    # for layer in eff_model.layers:
    #   layer.trainable = False

    # model = Sequential()
    # model.add(eff_model)
    # model.add(MaxPooling2D(2))
    # model.add(Flatten())
    # model.add(BatchNormalization())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4, activation='softmax'))

    inputs = Input(shape=input_shape)
    base_model = EfficientNetB7(include_top=False, pooling='avg', input_shape=(150,150,3), weights=None)
    for layer in base_model.layers:
        layer.trainable =  False
    #x = preprocess_input(inputs)
    x = base_model(inputs)
    #x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(2560, activation="relu")(x)
    x = Dropout(0.2)(x)
    # x = BatchNormalization()(x)
    # x = Dense(128, activation="relu")(x)
    # x = Dropout(0.2)(x)
    # x = BatchNormalization()(x)
    # x = Dense(64, activation="relu")(x)
    # x = Dropout(0.2)(x)
    outputs = Dense(4, activation="softmax")(x)
    model = Model(inputs=inputs,outputs=outputs)

    print("✅ Model initialized")
    print(model.summary)

    return model


def compile_model(model: Model, learning_rate) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    #model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=[tf.keras.metrics.Recall()])
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
        monitor="val_accuracy",
        patience=patience,
        min_delta=.01,
        mode='auto',
        restore_best_weights=True,
        verbose=1,
        start_from_epoch = 0
    )

    rlr = ReduceLROnPlateau( monitor="val_accuracy",
                            factor=0.01,
                            patience=patience,
                            verbose=0,
                            mode="max",
                            min_delta=0.01)

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, rlr],
        verbose=1
    )




    print(f"✅ Model trained")

    return model, history


def evaluate_model(
        model: Model,
        test_data,
        batch_size=32
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model ..." + Style.RESET_ALL)

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

    print(f"✅ Model evaluated, accuracy: {round(accuracy, 2)}")

    return metrics
