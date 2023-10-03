import math

import keras.layers
import numpy as np
import keras.callbacks as c
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.callbacks import EarlyStoppingAtMinLoss


def load_csv_data(file, names, target):
    x_data = pd.read_csv(file, names=names)
    return x_data, x_data.pop(target)


def exercice_2():
    x_data, target = load_csv_data("../Data/table_2_9.csv", names=['x1', 'x2', 'y'], target='y')

    # create model
    model = tf.keras.Sequential()
    model.add(keras.layers.Dense(units=1, activation='linear', input_shape=[2]))
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    model.compile(loss="mse", optimizer=optimizer)

    # display model
    model.summary()
    # callback = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=2)
    # learn

    history = model.fit(x_data, target, epochs=1000,
                        callbacks=[EarlyStoppingAtMinLoss(0.24)],
                        verbose=1)

    # Predict (compute) the output
    # y_predicted = model.predict(x_data)

    x_grid = [-2, 10]
    # [0] = weights, [1] = biases
    w = model.layers[0].get_weights()

    # display the result
    plt.scatter(x_data[['x1']], x_data[['x2']])
    plt.xlabel('x1')
    plt.ylabel('x2')

    y = -(w[0][0] * x_grid + w[1][0]) / w[0][1]

    plt.plot(x_grid, y, '-r')
    plt.grid()
    plt.title('Résultat obtenu via la droite de décision')
    plt.show()

    plt.plot(history.epoch, history.history["loss"], 'g', label='Training loss')
    plt.title('Fonction coût')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    exercice_2()
