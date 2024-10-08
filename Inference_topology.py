import numpy as np
import tensorflow as tf
from tensorflow import keras



def root_abs_error(y_true, y_pred):
    abs_diff = tf.abs(y_true - y_pred)
    return tf.reduce_mean(tf.sqrt(abs_diff), axis=-1)

NN = "10M_A"
folder = f"trained_NNs/1/{NN}/" # path to the trained NN
model = keras.models.load_model(folder, custom_objects = {"root_abs_error":root_abs_error})

histogram_dir = r"2D_histograms/" # path to the 2D-histogram
dataset = "HISTOGRAM_DATASET.npy" # name of the 2D-histogram

histo = np.load(histogram_dir+dataset)
histo = np.expand_dims(histo,axis=0)
histo = np.expand_dims(histo,axis=-1)
histo = np.where(histo == 0, 0, np.log10(histo ** 2)) # 2D-histogram scaling

a = model.predict(histo)


print(a)

print("COMPLETED")
