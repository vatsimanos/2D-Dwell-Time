import numpy as np
import matplotlib.pyplot as plt
import os

path = r"2D_histograms/" # path to the 2D-histograms

predicted_dir = f"PREDICTED_2D_HISTOGRAM.npy" # name of the predicted 2D-histograms file
ground_truth_dir = f"RECORDED_2D_HISTOGRAM.npy" # name of the ground truth 2D-histogram file

os.chdir(path)
f1 = open(predicted_dir, 'rb')
predicted_histos = np.zeros((100,60,60))

for i in range(100):
    predicted_histos[i] = np.load(f1, allow_pickle=True)[1]

predicted_histos = np.where(predicted_histos==0,0,np.log10(predicted_histos**2))  # log scaling of bin occupancy for predicted histograms
ground_truth = np.load(ground_truth_dir)
ground_truth = np.where(ground_truth==0,0,np.log10(ground_truth**2)) # log scaling of bin occupancy for recorded histograms

match_sub = np.zeros(100)
for i in range(100):
    compared = (ground_truth-predicted_histos[i])*(ground_truth+predicted_histos[i])
    compared = np.where(compared < 0, np.sqrt(-compared), np.sqrt(compared))
    match_sub[i] = ((np.sum(compared) / (np.sum(predicted_histos[i]) + np.sum(ground_truth))))

mean = np.mean(match_sub)
std = np.sqrt(np.var(match_sub))
print(mean,std)



