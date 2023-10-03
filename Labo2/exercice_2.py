import math
import seaborn as sns
import keras.layers
import keras.callbacks as c
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import cycle

from joblib.numpy_pickle_utils import xrange
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.confusion_matrix import confusion_matrix

from utils.myPlots import plot_matrix

cycol = cycle('bgrcmk')


def load_csv_data_with_classes(file, headers, x_data, classes):
    data = pd.read_csv(file, names=headers)
    x_data = data[x_data]
    target_data = data[classes]
    return x_data, target_data


def real_accuracy(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`


def to_matrix_5_x_5(values):
    matrix = np.array(values)
    matrix = matrix.reshape(-1, 5)

    return matrix


def exercice_2():
    draw_symboles = False
    header_list = []
    classes = []
    data = []
    for i in range(1, 26):
        header_list.append('x' + str(i))
        data.append('x' + str(i))
    for i in range(1, 5):
        header_list.append('y' + str(i))
        classes.append('y' + str(i))

    x_data, target = load_csv_data_with_classes("../Data/table_3_5.csv", headers=header_list, x_data=data,
                                                classes=classes)

    if draw_symboles:
        for i in xrange(x_data.shape[0]):
            matrix = to_matrix_5_x_5(x_data.iloc[i, :].values)
            plot_matrix(matrix)

    X_train, X_test, y_train, y_test = train_test_split(x_data, target, test_size=0.4)

    # create model
    model = tf.keras.Sequential()
    model.add(keras.layers.Dense(units=4, activation="linear", kernel_initializer='random_normal'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    model.compile(loss="mse", optimizer=optimizer, metrics=[real_accuracy])
    history = model.fit(x_data, target, epochs=100,
                        verbose=1)

    tf.keras.utils.plot_model(model, to_file="C:\\modelTensorflow\\model.png",
                              show_shapes=True)

    predictions = model.predict(X_test.values)

    w = model.get_weights()[0]
    for i in xrange(x_data.shape[0]):
        print(i)
        plot_matrix(to_matrix_5_x_5(w[:, i]), vmin=0, vmax=1)

    plot_accuracy(history)

    x_test, y_test = load_csv_data_with_classes("../Data/table_3_5_noisy.csv", headers=header_list, x_data=data,
                                                classes=classes)

    predictions = model.predict(x_test)

    # nb predictions correcte pour chaque classe (attention n'accepte pas les valeurs négatives)
    cm = confusion_matrix(y_test.values.argmax(axis=1), predictions.argmax(axis=1))

    plot_matrix(cm, labels=["S1", "S2", "S3", "S4"])


def plot_accuracy(history):
    plt.plot(history.history['real_accuracy'], label='real_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_data(x, y):
    plt.plot(x, y, c=next(cycol))
    plt.title('Résultat obtenu via la droite de décision')
    plt.legend()


if __name__ == '__main__':
    exercice_2()
