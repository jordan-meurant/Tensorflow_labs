import math

import keras.layers
import numpy as np
import keras.callbacks as c
import keras.initializers
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.confusion_matrix import confusion_matrix

from utils.callbacks import EarlyStoppingAtGoodAccuracy, EarlyStoppingAtMinLoss
from utils.myPlots import plot_matrix


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


def load_csv_data_with_classes(file, names, classes):
    data = pd.read_csv(file, names=names)
    x_data = data[['x1', 'x2', 'x3', 'x4']]
    target_data = data[classes]
    return x_data, target_data


def exercice_3():
    data = pd.read_csv("../Data/iris.csv",
                       names=['length_sepal', 'width_sepal', 'length_petal', 'width_petal', 'species'])
    X = data.drop(['species'], axis=1)
    y = pd.get_dummies(data.species, prefix='output')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation="sigmoid",
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1)),
        tf.keras.layers.Dense(10, activation="sigmoid",
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1)),
        tf.keras.layers.Dense(10, activation="sigmoid",
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1)),

        tf.keras.layers.Dense(3, activation="sigmoid",
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1)),
    ])

    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.SGD(learning_rate=1.2),
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=2000,
              callbacks=[EarlyStoppingAtMinLoss(0.001)]
              )

    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test.values.argmax(axis=1), predictions.argmax(axis=1))
    plot_matrix(cm, ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])


if __name__ == '__main__':
    exercice_3()
