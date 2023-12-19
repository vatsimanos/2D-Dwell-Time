import numpy as np
import tensorflow as tf
from tensorflow import keras



class Batch_generator_topology(keras.utils.Sequence):

    def __init__(self, X, y, batch):
        self.X = X
        self.y = y
        self.batch = batch

    def __len__(self):
        return int(np.floor(len(self.y)/self.batch))

    def __getitem__(self, idx):
        batch_X = self.X[idx * self.batch:(idx + 1) * self.batch]
        batch_y= self.y[idx * self.batch:(idx + 1) * self.batch]
        
        return batch_X, batch_y
