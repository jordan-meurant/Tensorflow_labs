import math

import keras.layers
import numpy as np
import keras.callbacks as c
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.callbacks import EarlyStoppingAtMinLoss


def load_csv_data(file, names, target=None):
    x_data = pd.read_csv(file, names=names)
    if target is None:
        return x_data
    else:
        return x_data, x_data.pop(target)


def exercice_4():
    x_data = load_csv_data("../Data/table_2_11.csv", names=['x1', 'x2'])

    x = np.arange(0, 40)
    x = np.random.choice(x, size=30)

    normalizer = keras.layers.Normalization(input_shape=[1, ], axis=None)
    normalizer.adapt(x_data)

    model = tf.keras.Sequential([
        normalizer,
        keras.layers.Dense(units=1)
    ])

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.09),
        loss='mae', metrics='accuracy')

    history = model.fit(x_data[['x1']], x_data[['x2']], epochs=150,
                        callbacks=[EarlyStoppingAtMinLoss(0.9)],
                        verbose=1)

    predictions = model.predict(x)

    plt.scatter(x, predictions, label='Predictions [−10, 100]', c='g')

    plot_data(x, predictions, x_data)
    plot_loss(history)


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_data(x, y, x_data):
    plt.scatter(x_data[['x1']], x_data[['x2']], label='Data', c='r')
    plt.plot(x, y, color='k', label='Predictions')
    plt.title('Résultat obtenu via la droite de régression')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    exercice_4()
