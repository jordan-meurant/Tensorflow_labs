import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.confusion_matrix import confusion_matrix

from utils.callbacks import EarlyStoppingAtGoodAccuracy
from utils.myPlots import plot_matrix


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.grid(True)
    plt.show()


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
        ax.contourf(xx, yy, z[i], levels=[0, 0.5, 1], alpha=0.3)
    return fig, ax


def exercice_3():
    data = pd.read_csv("../Data/iris.csv",
                       names=['length_sepal', 'width_sepal', 'length_petal', 'width_petal', 'species'])

    # sns.set(style="ticks")
    # sns.set_palette("husl")
    # sns.pairplot(data.iloc[:, 0:6], hue="species")
    # plt.show()

    # plt.figure()
    # sns.heatmap(data.corr(), annot=True)
    # plt.show()

    # fig = px.scatter_3d(data, x="length_petal", y="width_petal", z="length_sepal", size="width_sepal",
    #                     color="species")
    # fig.show()

    data['species'] = data.species.map(
        {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

    X = data.drop(['species'], axis=1)
    y = data.species

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(units=1, activation='linear', kernel_initializer='random_normal')])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.3)
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=1000)

    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test.values, predictions)
    plot_matrix(cm, labels)

    tf.keras.utils.plot_model(model, to_file="C:\\modelTensorflow\\model1N.png",
                              show_shapes=True)
    model3_neurones = tf.keras.Sequential(
        [tf.keras.layers.Dense(units=3, activation='linear', kernel_initializer='random_normal'), ])
    model3_neurones.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.3), metrics=['accuracy'])
    model3_neurones.fit(X_train, y_train, epochs=1000)

    predictions = model3_neurones.predict(X_test)

    # nb predictions correcte pour chaque classe
    cm = confusion_matrix(y_test.values, predictions.argmax(axis=1))
    plot_matrix(cm, labels)

    tf.keras.utils.plot_model(model3_neurones, to_file="C:\\modelTensorflow\\model3N.png",
                              show_shapes=True)
    # --------------------------------------------------------
    # 4. Transformer les données pour obtenir un réseau monocouche de 3 neurones utile.
    # convert species into dummy variables
    data = pd.read_csv("../Data/iris.csv",
                       names=['length_sepal', 'width_sepal', 'length_petal', 'width_petal', 'species'])
    X = data.drop(['species'], axis=1)
    y = pd.get_dummies(data.species, prefix='output')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(3, activation="softmax"),
    ])

    model2.compile(loss='mse',
                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
                   metrics=['accuracy'])

    history2 = model2.fit(X_train, y_train, epochs=1000, callbacks=[EarlyStoppingAtGoodAccuracy(0.98)])

    predictions = model2.predict(X_test)
    cm = confusion_matrix(y_test.values.argmax(axis=1), predictions.argmax(axis=1))
    plot_matrix(cm, labels)

    tf.keras.utils.plot_model(model2, to_file="C:\\modelTensorflow\\model3NUtile.png",
                              show_shapes=True)


if __name__ == '__main__':
    exercice_3()
