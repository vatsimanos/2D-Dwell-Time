import os
import numpy as np
import re
import matplotlib.pyplot as plt



class Data_importer_regression():

    def __init__(self,dir,data):
        self.dir = dir
        self.data = data
        os.chdir(dir)
        self.f1 = open(data, 'rb')

    def close(self):
        self.f1.close()

    def load_ts(self, number,dim1,dim2,number_of_rates,rearrange):
        X = np.zeros((number, dim1,dim2))
        y = []
        for i in range(number_of_rates):
            y.append(np.zeros((number), dtype=np.float32))

        for i in range(number):
            ts_helper = np.load(self.f1, allow_pickle=True)        
            string_helper = re.split('_', ts_helper[0])
            position = string_helper.index('kij')


            for j in range(number_of_rates):
                y[j][i] = (int(string_helper[position + j+1]))


            helper_2 = ts_helper[1]
            temp = helper_2[:dim1,:dim2]
            X[i] = temp


            X[i] = np.where(X[i]==0,0,np.log10(X[i]**2)) #log scaling of bin occupancy

            if(rearrange == True):

                if (y[0][i] < y[-1][i]):
                    
                    y_helper = np.zeros(number_of_rates)

                    for j in range(number_of_rates):
                        y_helper[j] = y[j][i]

                    for j in range(number_of_rates):
                        y[j][i] = y_helper[-(1+j)]

        y = np.log10(y)
        X = np.expand_dims(X, axis=3)

        return (X, y)