import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import Data_importer_regression as di_r


def root_abs_error(y_true, y_pred):
    abs_diff = tf.abs(y_true - y_pred)
    return tf.reduce_mean(tf.sqrt(abs_diff), axis=-1)

folder_number =  2
folder = f"trained_NNs/{folder_number}/" # path to the trained NN
model = keras.models.load_model(folder, custom_objects = {"root_abs_error":root_abs_error})

histogram_dir = r"2D_histograms/"  # path to the 2D-histogram
dataset = "HISTOGRAM_DATASET.npy" # name of the 2D-histogram


import_ts = di_r.Data_importer_regression(histogram_dir,dataset)
histo, _ = import_ts.load_ts(100,60,60,8,False)


y1,y2,y3,y4,y5,y6,y7,y8 = model.predict(histo)

r1 = 10**y1
r2 = 10**y2
r3 = 10**y3
r4 = 10**y4
r5 = 10**y5
r6 = 10**y6
r7 = 10**y7
r8 = 10**y8

print(r1,r2,r3,r4,r5,r6,r7,r8)

print("COMPLETED")
