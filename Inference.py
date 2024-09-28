import numpy as np
import tensorflow as tf
from tensorflow import keras



def root_abs_error(y_true, y_pred):
    abs_diff = tf.abs(y_true - y_pred)
    return tf.reduce_mean(tf.sqrt(abs_diff), axis=-1)

folder = f"PATH_TO_SAVED_NN/"
model = keras.models.load_model(folder, custom_objects = {"root_abs_error":root_abs_error})

histogram_dir = r"PATH_TO_HISTOGRAM/"
dataset = "2D_HISTOGRAM_NAME.npy"

histo = np.load(histogram_dir+dataset)
histo = np.expand_dims(histo,axis=0)
histo = np.expand_dims(histo,axis=-1)
histo = np.where(histo == 0, 0, np.log10(histo ** 2))

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
