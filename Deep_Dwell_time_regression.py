import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import Batch_generator_regression as bg_r
import Data_importer_regression as di_r
import Custom_layers as cl



#----------------SETTINGS----------------#
save_folder = "RESULT_FOLDER"
save_location = f"/PATH_TO_RESULT_FOLDER/{save_folder}/" # path to folder were all results of the run are saved

data = "NAME_OF_DATASET.npy"          
data_path= f'/PATH_TO_DATASET/' # directory containing training files

hist_dim = (60,60) # dimensions of 2D Dwell Time Histograms
train_number = 10000
val_number = 10000
test_number = 10000
max_epochs = 1000
global_batch_size = 1024 
number_of_rates = 8
rearrange_labels = True # set to True if topology is symmetric
starting_lr = 0.001
example_histogram_number = 20 # number of histograms from test dataset to save as images in result folder
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, min_lr=0.0000000001, verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
save_model = True
save_test_data = False # set to True to save entire test dataset in result folder
#----------------SETTINGS----------------#

def root_abs_error(y_true, y_pred):
    abs_diff = tf.abs(y_true - y_pred)
    return tf.reduce_mean(tf.sqrt(abs_diff), axis=-1)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')

for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

X_train = np.zeros((train_number, hist_dim[0], hist_dim[1], 1), dtype=np.float32)
X_val = np.zeros((val_number, hist_dim[0], hist_dim[1], 1), dtype=np.float32)
X_test = np.zeros((test_number, hist_dim[0], hist_dim[1], 1), dtype=np.float32)

y_train = []
for i in range(number_of_rates):
    y_train.append(np.zeros((train_number), dtype=np.float32))

y_val = []
for i in range(number_of_rates):
    y_val.append(np.zeros((val_number), dtype=np.float32))

y_test = []
for i in range(number_of_rates):
    y_test.append(np.zeros((test_number), dtype=np.float32))

import_ts = di_r.Data_importer_regression(data_path,data)

print("import train data")
for i in range(0, int(train_number / 1000)):
    X_helper, y_helper = import_ts.load_ts(1000,hist_dim[0],hist_dim[1],number_of_rates,rearrange_labels)
    X_train[i * 1000:i * 1000 + 1000] = X_helper

    for j in range(number_of_rates):
        y_train[j][i * 1000:i * 1000 + 1000] = y_helper[j]

    print("\rimported", (i + 1) * 1000, "/", train_number, end="")
print()

print("import val data")

for i in range(0, int(val_number / 1000)):
    X_helper, y_helper = import_ts.load_ts(1000,hist_dim[0],hist_dim[1],number_of_rates,rearrange_labels)
    X_val[i * 1000:i * 1000 + 1000] = X_helper

    for j in range(number_of_rates):
        y_val[j][i * 1000:i * 1000 + 1000] = y_helper[j]

    print("\rimported", (i + 1) * 1000, "/", val_number, end="")

print()
print("import test data")

for i in range(0, int(test_number / 1000)):
    X_helper, y_helper = import_ts.load_ts(1000,hist_dim[0],hist_dim[1],number_of_rates,rearrange_labels)
    X_test[i * 1000:i * 1000 + 1000] = X_helper

    for j in range(number_of_rates):
        y_test[j][i * 1000:i * 1000 + 1000] = y_helper[j]

    print("\rimported", (i + 1) * 1000, "/", test_number, end="")

if save_test_data == True: np.save(f'{save_location}/test_data.npy', X_test) # save test dataset

my_training_batch_generator = bg_r.Batch_generator_regression(X_train, y_train, global_batch_size,number_of_rates)
my_eval_batch_generator = bg_r.Batch_generator_regression(X_val, y_val, global_batch_size,number_of_rates)

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():

    input = tf.keras.layers.Input((hist_dim[0], hist_dim[1], 1))

    x = cl.inception_A_res(input)
    x = cl.inception_A_res(x)
    x = cl.inception_A_res(x)
    x = cl.inception_A_res(x)

    x = cl.reduction_A_res(x)

    x = cl.inception_B_res(x)
    x = cl.inception_B_res(x)
    x = cl.inception_B_res(x)
    x = cl.inception_B_res(x)
    x = cl.inception_B_res(x)
    x = cl.inception_B_res(x)
    x = cl.inception_B_res(x)
    x = cl.inception_B_res(x)
    x = cl.inception_B_res(x)
    x = cl.inception_B_res(x)

    x = tf.keras.layers.Flatten()(x)

    out = [tf.keras.layers.Dense(1)(x) for i in range(number_of_rates)]    

    model = tf.keras.Model(inputs=input, outputs=out)

    losses = ['logcosh' for i in range(number_of_rates)]

    model.compile(
        loss=losses,
        optimizer=tf.keras.optimizers.Adam(learning_rate=starting_lr),
        metrics= [root_abs_error])


print(model.summary())
print('start training')
print()
history = model.fit(x = my_training_batch_generator, validation_data = my_eval_batch_generator, epochs = max_epochs,callbacks = [reduce_lr,early_stop], verbose = 2)

print("model evaluate: validation data ", val_number)
model.evaluate(X_val,y_val,  verbose = 2)
print("model evaluate: test data ", test_number)
model.evaluate(X_test,y_test,  verbose = 2)

if save_model == True: model.save(save_location)
np.save(save_location + '/history', history.history)

# list all data in history
print("history logged: ",history.history.keys())

# summarize history for loss
mng = plt.get_current_fig_manager()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.grid(b=True)
plt.xscale('log')
plt.title('Training Curves',fontsize=35)
plt.ylabel('Loss',fontsize=30)
plt.xlabel('Epochs',fontsize=30)
plt.legend(['training', 'validation'], loc='upper right',fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig(save_location+f'{save_folder}.png')
plt.close()


predictions = model.predict(X_test)

for i in range(number_of_rates):
    np.save(save_location + f'/pred{i+1}.npy', predictions[i])
    np.save(save_location + f'/label{i+1}.npy', y_test[i])

    mng = plt.get_current_fig_manager()
    plt.scatter(10**y_test[i],10**predictions[i],marker='.')
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(1e2,1e5)
    plt.xlim(1e2,1e5)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.title(f'confusion distribution k{i}',fontsize=35)
    plt.ylabel('prediction [Hz]',fontsize=30)
    plt.xlabel('ground truth [Hz]',fontsize=30)
    plt.savefig(save_location+f'{save_folder}_scatter{i}.png')
    plt.close()



X_test = np.squeeze(X_test)

for i in range(0, example_histogram_number):

    plt.pcolormesh(X_test[i], cmap="gray")
    plt.xlabel("lower")
    plt.ylabel("upper")
    plt.title(str(y_test))
    plt.savefig(save_location + f'{save_folder}_histo_{i}.png')
    plt.close()

print(">----------------COMPLETED----------------<")