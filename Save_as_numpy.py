import numpy as np
from datetime import datetime as dt



now = dt.now()
_ID = now.strftime(f"%d%m%Y%H%M%S")

#----------------SETTINGS----------------#
binary_path = f'/PATH_TO_BINARY/binary.dat'
np_array_path = f"/PATH_TO_SAVE_NUMPY/numpy_data_{_ID}.npy" 
histogram_import_number = 1000000 # number of histograms to save from binary
hist_dim = (60,60) # dimensions of 2D Dwell Time Histograms as they have been simulated with the 2D-Fit
#----------------SETTINGS----------------#



print("converting . . . ")

with open(binary_path,'r',encoding="ISO-8859-1") as f, open(np_array_path, 'ab') as f2:
    for i in range(histogram_import_number):
        name = f.read(1000)
        name = name.split('.')[0]
        name = name.split('/')[-1]  
        data = np.fromfile(f, dtype=np.float64, count=int(hist_dim[0] * hist_dim[1]))
        hist = np.transpose(data.reshape(hist_dim))
        arr = np.array([name, hist], dtype=object)
        np.save(f2, arr)
f2.close
f.close

print(">----------------COMPLETED----------------<")


