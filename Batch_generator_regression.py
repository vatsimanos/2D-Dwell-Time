import numpy as np
import tensorflow as tf
from tensorflow import keras



class Batch_generator_regression(keras.utils.Sequence):

    def __init__(self, X, y, batch_size,number_of_rates):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.number_of_rates = number_of_rates

    def __len__(self):
        return int(np.floor(len(self.y[0])/self.batch_size))

    def __getitem__(self, idx):

        batch_X = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = np.zeros((self.number_of_rates, self.batch_size))

        for i in range(self.number_of_rates):
            batch_y[i] = self.y[i][idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_X, [batch_y[i] for i in range(self.number_of_rates)]