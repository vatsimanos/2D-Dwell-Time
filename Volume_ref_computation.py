import numpy as np
import matplotlib.pyplot as plt
import os

path = r"PATH_TO_HISTOGRAMS/"

predicted_dir = f"NAME_OF_HISTOGRAM_FILE.npy"


os.chdir(path)
f1 = open(predicted_dir, 'rb')
predicted_histos = np.zeros((100,60,60))

for i in range(100):
    predicted_histos[i] = np.load(f1, allow_pickle=True)[1]

predicted_histos = np.where(predicted_histos==0,0,np.log10(predicted_histos**2))

match_sub = []
for j in range(100):
    for i in range(j):
        compared = (predicted_histos[j] - predicted_histos[i]) * (predicted_histos[j] + predicted_histos[i])
        compared = np.sqrt(np.abs(compared))
        match_sub = np.append(match_sub,((np.sum(compared) / (np.sum(predicted_histos[j]) + np.sum(predicted_histos[i])))))

mean = np.mean(match_sub)
std = np.sqrt(np.var(match_sub))
print(mean,std)

