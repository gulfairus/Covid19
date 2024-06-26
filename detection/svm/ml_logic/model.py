import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")



def initialize_model(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    reg = regularizers.l2(0.001)

    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3), padding="SAME", activation="relu", input_shape=(150,150,3)))
    model.add(Conv2D(32,kernel_size=(3,3), padding="SAME", activation="relu"))
    model.add(Conv2D(32,kernel_size=(3,3), padding="SAME", activation="relu"))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    #model.add(Rescaling(1./255, input_shape=input_shape))

    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation="relu"))
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation="relu"))
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation="relu"))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation="relu"))
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation="relu"))
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation="relu"))
    model.add(MaxPooling2D(2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation="relu"))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation="relu"))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation="relu"))
    model.add(MaxPooling2D(2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(1024, kernel_size=(3,3), padding='same', activation="relu"))
    model.add(Conv2D(1024, kernel_size=(3,3), padding='same', activation="relu"))
    model.add(Conv2D(1024, kernel_size=(3,3), padding='same', activation="relu"))
    model.add(MaxPooling2D(2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # model.add(Conv2D(2048, kernel_size=(3,3), padding='same', activation="relu"))
    # model.add(Conv2D(2048, kernel_size=(3,3), padding='same', activation="relu"))
    # model.add(Conv2D(2048, kernel_size=(3,3), padding='same', activation="relu"))
    # model.add(MaxPooling2D(2))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(2048, activation='relu'))
    #model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))


    model.add(Dense(4, activation="softmax"))


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
        min_delta=.01,
        mode='auto',
        restore_best_weights=True,
        verbose=1,
        #start_from_epoch = 10
    )

    rlr = ReduceLROnPlateau( monitor="val_loss",
                            factor=0.2,
                            patience=patience,
                            verbose=0,
                            mode="auto",
                            min_delta=0.001)

    #steps = len(train_names)//batch_size

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
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"✅ Model evaluated")

    return metrics
