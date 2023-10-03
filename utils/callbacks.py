import keras.callbacks as c
import keras.layers
import numpy as np


class EarlyStoppingAtMinLoss(keras.callbacks.Callback):

    def __init__(self, seuil=0.3):
        super(EarlyStoppingAtMinLoss, self).__init__()
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.seuil = seuil

    def on_train_begin(self, logs=None):
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.seuil):
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print("Restoring model weights from the end of the best epoch.")

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


class EarlyStoppingAtGoodAccuracy(keras.callbacks.Callback):

    def __init__(self, acc=0.8):
        super(EarlyStoppingAtGoodAccuracy, self).__init__()
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.acc = acc

    def on_train_begin(self, logs=None):
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("accuracy")
        if np.greater(current, self.acc):
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print("Restoring model weights from the end of the best epoch.")

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
