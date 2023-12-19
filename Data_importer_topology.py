import os
import numpy as np
import re
import tensorflow as tf



class Data_importer_topology():

    def __init__(self,dir,paths_array):

        self.dir = dir
        self.paths_array = paths_array
        self.number_of_classes = paths_array.shape[0]
        os.chdir(dir)
        self.load_helper_array = np.zeros(self.number_of_classes , dtype=object)
        
        for i in range(self.number_of_classes):
            self.load_helper_array[i] = open(self.paths_array[i], 'rb')



    def close(self):

        for i in range(self.number_of_classes):
            self.load_helper_array[i].close()



    def load_ts(self, number,dim1,dim2):
        X = np.zeros((number, dim1,dim2))
        y = np.zeros((number))

        for i in range(number):
            idx = (i + self.number_of_classes ) % self.number_of_classes
            ts_helper = np.load(self.load_helper_array[idx], allow_pickle=True)

            if(ts_helper.ndim == 1):
                helper_2 = ts_helper[1]
            else:
                helper_2 =  np.squeeze(ts_helper)[1]

            temp = helper_2[:dim1,:dim2]
            X[i] = temp
            y[i] = idx

            if(False): # for testing
         
                plt.pcolormesh(X[i], cmap="gray")
                plt.xlabel("closed")
                plt.ylabel("open")
                #textlabel = "COC model - k12 = " + str(y1_test[i]) + " - k21 = " + str(y2_test[i]) + " k23 = " + str(y3_test[i]) + " - k32 = " + str(y4_test[i])
                #plt.title(textlabel)
                plt.savefig(save_location + f'{run_type}_histo_{i}.png')
                plt.close()
            
            X[i] = np.where(X[i]==0,0,np.log10(X[i]**2)) #log scaling of bin occupancy



        y = y.reshape((len(y), 1))
        y = tf.keras.utils.to_categorical(y, self.number_of_classes)
        X = np.expand_dims(X, axis=3)

        return (X, y)