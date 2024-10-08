import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import Batch_generator_topology as bg_t
import Data_importer_topology as di_t
import Custom_layers as cl



#----------------SETTINGS----------------#

save_folder = "RESULT_FOLDER" # name of the result folder
save_location = f"/PATH_TO_RESULT_FOLDER/{save_folder}/" # path to the result folder

data_names = "topology_datasets.txt"
data_path= f'/PATH_TO_DATASETS/' #  path to the training datasets (containing all the different Markov topologies)

hist_dim = (60,60) # dimensions of 2D Dwell Time Histograms
train_number = 10000000
val_number = 180000
test_number = 180000
max_epochs = 1000
global_batch_size = 4096
starting_lr = 0.001
example_histogram_number = 20 # number of histograms from test dataset to save as images in result folder
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=0.0000000001, verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
save_model = True
save_test_data = False # set to True to save entire test dataset in result folder
#----------------SETTINGS----------------#

gpu_devices = tf.config.experimental.list_physical_devices('GPU')

for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


paths_array = np.loadtxt(data_names, dtype='str')

number_of_classes = paths_array.shape[0]

print(f"number of classes: {number_of_classes}")

X_train = np.zeros((train_number, hist_dim[0], hist_dim[1], 1), dtype=np.float32)
X_val = np.zeros((val_number, hist_dim[0], hist_dim[1], 1), dtype=np.float32)
X_test = np.zeros((test_number, hist_dim[0], hist_dim[1], 1), dtype=np.float32)

y_train = np.zeros((train_number,number_of_classes), dtype=np.float32)
y_val = np.zeros((val_number,number_of_classes), dtype=np.float32)
y_test = np.zeros((test_number,number_of_classes), dtype=np.float32)

import_ts = di_t.Data_importer_topology(data_path,paths_array)

print("import train data")
for i in range(0, int(train_number / 1000)):
    X_helper, y_helper = import_ts.load_ts(1000, hist_dim[0],hist_dim[1])
    X_train[i * 1000:i * 1000 + 1000] = X_helper
    y_train[i * 1000:i * 1000 + 1000] = y_helper

    print("\rimported", (i + 1) * 1000, "/", train_number, end="")
print()

print("import val data")
for i in range(0, int(val_number / 1000)):
    X_helper, y_helper = import_ts.load_ts(1000, hist_dim[0],hist_dim[1])
    X_val[i * 1000:i * 1000 + 1000] = X_helper
    y_val[i * 1000:i * 1000 + 1000] = y_helper
    print("\rimported", (i + 1) * 1000, "/", val_number, end="")
print()

print("import test data")
for i in range(0, int(test_number / 1000)):
    X_helper, y_helper = import_ts.load_ts(1000, hist_dim[0],hist_dim[1])
    X_test[i * 1000:i * 1000 + 1000] = X_helper
    y_test[i * 1000:i * 1000 + 1000] = y_helper
    print("\rimported", (i + 1) * 1000, "/", test_number, end="")

if save_test_data == True: np.save(f'{save_location}/test_data.npy', X_test) # save test dataset

my_training_batch_generator = bg_t.Batch_generator_topology(X_train, y_train, global_batch_size)
my_eval_batch_generator = bg_t.Batch_generator_topology(X_val, y_val, global_batch_size)



mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():

    input = tf.keras.layers.Input((hist_dim[0], hist_dim[1], 1))

    x = cl.inception_A_res(input)
    x = cl.inception_A_res(x)
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

    x = cl.channel_increase(x)

    x = cl.inception_C_res(x)
    x = cl.inception_C_res(x)
    x = cl.inception_C_res(x)
    x = cl.inception_C_res(x)
    x = cl.inception_C_res(x)

    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    output = tf.keras.layers.Dense(number_of_classes, activation='softmax', name='topology')(x)

    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics= ['accuracy'])
        

print(model.summary())
print('start training')
print()
history = model.fit(x = my_training_batch_generator, validation_data=my_eval_batch_generator,epochs=max_epochs,callbacks=[reduce_lr,early_stop], verbose = 2)


print("evaluation")
print("model evaluate: evaulation data ", val_number)
model.evaluate(X_val,y_val, verbose = 2)
print("model evaluate: test data ", test_number)
model.evaluate(X_test,y_test, verbose = 2)

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
plt.title('Model Performance',fontsize=35)
plt.ylabel('Loss',fontsize=30)
plt.xlabel('Epochs',fontsize=30)
plt.legend(['training', 'validation'], loc='upper right',fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig(save_location+f'{save_folder}.png')
plt.close()

predictions = model.predict(X_test)
np.save(save_location + '/pred.npy', predictions)
np.save(save_location + '/label.npy', y_test)

#Save loss during trtaining
np.save(save_location + '/train_loss.npy', history.history['loss'])
np.save(save_location + '/valid_loss.npy', history.history['val_loss'])

fig, ax = plt.subplots()
ax.set_aspect("equal")
hist, xbins, ybins, im = ax.hist2d(np.argmax(y_test, axis=1),np.argmax(predictions, axis=1), bins=range(19))
plt.title('Confusion Matrix')
plt.ylabel('prediction')
plt.xlabel('ground truth')
ax.set_yticklabels([])
ax.set_xticklabels([])
sum = np.sum(hist.T, axis=0)

for i in range(len(ybins)-1):
    for j in range(len(xbins)-1):
        ax.text(xbins[j]+0.5,ybins[i]+0.5, int(100 * hist.T[i,j]/(sum[i] + np.finfo(float).eps)), color="w", ha="center", va="center", fontweight="bold", fontsize=5)
			

plt.savefig(save_location + f'{save_folder}_confusion_matrix.png', dpi=300)
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