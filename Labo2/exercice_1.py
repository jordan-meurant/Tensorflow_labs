import math

import keras.layers
import keras.callbacks as c
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import cycle

from utils.callbacks import EarlyStoppingAtMinLoss

cycol = cycle('bgrcmk')


def load_csv_data_with_classes(file, names, classes):
    data = pd.read_csv(file, names=names)
    x_data = data[['x1', 'x2']]
    target_data = data[classes]
    return x_data, target_data


def exercice_1():
    x_data, target = load_csv_data_with_classes("../Data/table_3_1.csv", names=['x1', 'x2', 'y1', 'y2', 'y3'],
                                                classes=['y1', 'y2', 'y3'])

    neurones = 3
    model = tf.keras.Sequential(
        [keras.layers.Dense(units=neurones, activation='linear', input_dim=2),
         # keras.layers.Dense(units=3, activation='linear', )
         ])
    print(target.shape)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(loss="mse", optimizer=optimizer)
    history = model.fit(x_data, target, epochs=2000,
                        callbacks=[EarlyStoppingAtMinLoss(0.02)],
                        verbose=1)

    w = model.get_weights()

    x = np.asarray([-2, 6])

    for i in range((len(target.y1))):
        if target.y1[i] == 1:
            plt.scatter(x_data.x1[i], x_data.x2[i], c='b')

    for i in range((len(target.y2))):
        if target.y2[i] == 1:
            plt.scatter(x_data.x1[i], x_data.x2[i], c='g')

    for i in range((len(target.y3))):
        if target.y3[i] == 1:
            plt.scatter(x_data.x1[i], x_data.x2[i], c='r')

    for i in range(0, neurones):
        y = -(w[0][0][i] * x + w[1][i]) / w[0][1][i]
        plot_data(x, y)

    plt.xlim(-2, 5)
    plt.ylim(-5, 9)
    plt.show()


def plot_data(x, y):
    plt.plot(x, y, c=next(cycol))
    plt.title('Model')


if __name__ == '__main__':
    exercice_1()
