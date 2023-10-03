import keras.callbacks as c
import keras.layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.callbacks import EarlyStoppingAtMinLoss


def load_csv_data(file, names, target):
    x_data = pd.read_csv(file, names=names)
    return x_data, x_data.pop(target)


def exercice_1():
    # x_data, target = load_csv_data("../Data/table_2_1.csv", names=['x1', 'x2', 'y'], target='y')

    ds_columns = ['x1', 'x2', 'y']
    all_data = pd.read_csv("../Data/table_2_1.csv", names=ds_columns)
    x_data, target_data = all_data, all_data.pop('y')
    print(x_data[['x1']])
    print(x_data[['x2']])

    # create model
    model = tf.keras.Sequential()
    model.add(keras.layers.Dense(units=1, activation='linear', input_shape=[2]))
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(loss="mse", metrics="mae", optimizer=optimizer)

    # display model
    model.summary()
    # callback = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=2)
    # learn
    # By default, any change in the performance measure, no matter how fractional, will be considered an improvement
    # Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_
    # delta, will count as no improvement.

    #history = model.fit(x_data, target_data, epochs=2000,
    #                   callbacks=c.EarlyStopping(monitor="loss", mode="auto", min_delta=0.002, patience=3, verbose=1),
    #                   verbose=1)
    history = model.fit(x_data, target_data, epochs=1000,
                        callbacks=[EarlyStoppingAtMinLoss(0.3)],
                        verbose=1)

    x_grid = [-2, 2]
    # [0] = weights, [1] = biases
    w = model.layers[0].get_weights()

    # display the result
    plt.scatter(x_data[['x1']], x_data[['x2']])
    plt.xlabel('x')
    plt.ylabel('y')

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


def accuracy(y_predicted, target):
    acc = (y_predicted - target.to_numpy()) ** 2
    acc = np.sqrt(acc)
    print("voici l'accuracy : ", acc)


if __name__ == '__main__':
    print("hello")
    exercice_1()
