import math

import keras.layers
import numpy as np
import keras.callbacks as c
import keras.initializers
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.callbacks import EarlyStoppingAtMinLoss


def plot_decision_boundary(x, y, model, ndim=1):
    """
    This function plots the decision boundary of an existing model for a 2 dimensional input.
    Example : plot_decision_boundary(np.linspace(0,1,10), np.linspace(0,2,10), model, 3)
    will plot the prediction of the model for a 100 points mesh between [0, 1] and [0, 2] for
    the x and y axis respectively. The model should have 3 output neurons.
    :param x: Data at which the model is evaluated - first dimension.
    :param y: Data at which the model is evaluated - second dimension.
    :param model: Model to evaluate
    :param ndim: Total number of outputs neurons in the model.
    :return: the figure and axes obtained.
    """
    xx, yy = np.meshgrid(x, y)
    plt_in_data = np.c_[xx.ravel(), yy.ravel()]
    labels = model.predict(plt_in_data)
    fig, ax = plt.subplots()
    labels = labels.reshape(xx.shape[0], xx.shape[1], ndim)
    z = np.zeros((ndim, xx.shape[0], xx.shape[1]))
    for i in range(ndim):
        z[i] = np.rint(labels[:, :, i])
        ax.contour(xx, yy, z[i], levels=[0, 1])
        ax.contourf(xx, yy, z[i], levels=[0, 0.5, 1], alpha=0.5)
    return fig, ax


def load_csv_data(file, names, target):
    x_data = pd.read_csv(file, names=names)
    return x_data, x_data.pop(target)


def exercice_1():
    x_data, target = load_csv_data("../Data/table_4_3.csv", names=['x1', 'x2', 'y'], target='y')
    initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=1)
    model = tf.keras.Sequential(
        [keras.layers.Dense(units=2, activation='sigmoid', kernel_initializer=initializer, input_dim=2),
         keras.layers.Dense(units=1, activation='sigmoid', kernel_initializer=initializer)])

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.8)
    model.compile(loss="mean_squared_error", optimizer=optimizer)

    model.fit(x_data, target, epochs=2000, verbose=1, batch_size=4,
              callbacks=[EarlyStoppingAtMinLoss(0.001)]
              )

    plot_decision_boundary(np.linspace(-1, 2, 100), np.linspace(-1, 2, 100), model, 1)
    plt.scatter(x_data[['x1']], x_data[['x2']])
    plt.grid()
    plt.show()


if __name__ == '__main__':
    exercice_1()
