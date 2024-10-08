#Efthymios Oikonomou 03.11.2022
import numpy as np
import matplotlib.pyplot as plt
import time

'''SETTINGS START'''
path = r"ts/"  # path to time series and save location for 2D-histogram
name = "TIME_SERIES.txt"  # name of the file of the time series
histogram_dataset_name = "2D_HISTOGRAM_NAME.npy"  # name of the resulting Numpy file of the 2D-histogram
ts = np.loadtxt(path + name)  # change data type accordingly

ts = np.expand_dims(ts,axis=0)  # disable for multiple histograms

lower_lvl = 20000 # lower amplitude level
upper_lvl = 22000 # higher amplitude level
sigma = 1000 # the standard deviation of the noise
sampling_frequency = 100000
data_size = 1 # number of ts to transform
'''SETTINGS END'''


order = 4
half_jump_magnitude = 0.5
amplitude = upper_lvl - lower_lvl
SNR = amplitude/sigma
SNR = SNR * SNR
t_res = int(1 + 32/SNR)



def compute_threshold(t_res, order, half_jump_magnitude):

    g = np.zeros(9)

    g[0] = half_jump_magnitude

    for t in range(t_res):
        g = np.cumsum(g)


    threshold = g[order]

    g[0] = - half_jump_magnitude

    for t in range(t_res):
        g = np.cumsum(g)
        if g[order] > threshold: threshold = g[order]


    return threshold


def HOHD(time_series,lower,upper,threshold):
    #calculat test values
    print(time_series.shape)
    amplitude = upper - lower
    ts_length = time_series.shape[0]
    order = 4
    g_all_orders_up = np.zeros(9)
    g_all_orders_down = np.zeros(9)
    jump_list = np.zeros(ts_length)
    half_jump = 0.5

    mystery_variable = 0

    i = 0
    lastmin = i
    lastmax = i
    while i < (time_series.shape[0]):
        if i > time_series.shape[0]:
            break
        z = time_series[i]

        # up

        if (mystery_variable == 1):
            lastmin = i
        else:
            g_all_orders_up[0] = (z - lower) / amplitude - half_jump - mystery_variable
            g_all_orders_up = np.cumsum(g_all_orders_up)
            if g_all_orders_up[1] <= 0:
                g_all_orders_up[:] = 0
                lastmin = i

        # down

        if (mystery_variable == 0):
            lastmax = i
        else:
            g_all_orders_down[0] = (z - lower) / amplitude + half_jump - mystery_variable
            g_all_orders_down = np.cumsum(g_all_orders_down)
            if g_all_orders_down[1] >= 0:
                g_all_orders_down[:] = 0
                lastmax = i


        if (g_all_orders_up[order] > threshold):
            i = lastmin
            lastmax = i
            jump_list[i] = 1
            mystery_variable = 1
            g_all_orders_up[:] = 0
            g_all_orders_down[:] = 0


        if (g_all_orders_down[order] < -threshold):
            i = lastmax
            lastmin = i
            jump_list[i] = -1
            mystery_variable = 0
            g_all_orders_up[:] = 0
            g_all_orders_down[:] = 0

        i = i + 1

    return jump_list

def two_dim_hist(jump_list, detailed_balance):

    idx_list = np.nonzero(jump_list)

    idx_list_extended = np.append(idx_list,0)
    idx_list_shifted = np.insert(idx_list,0,0)

    dwell_times = idx_list_extended - idx_list_shifted

    dwell_times = np.delete(dwell_times,0,axis=0)
    dwell_times = np.delete(dwell_times, 0, axis=0)

    dwell_times = np.delete(dwell_times, -1, axis=0)

    upper_dwell_times = dwell_times[np.arange(1,dwell_times.shape[0],2)]
    lower_dwell_times = dwell_times[np.arange(0,dwell_times.shape[0],2)]

    if(upper_dwell_times.shape > lower_dwell_times.shape):
        upper_dwell_times = np.delete(upper_dwell_times, -1)

    elif(upper_dwell_times.shape < lower_dwell_times.shape):
        lower_dwell_times = np.delete(lower_dwell_times, -1)

    logged_upper_dwell_times = np.log10(upper_dwell_times*(1/sampling_frequency))
    logged_lower_dwell_times = np.log10(lower_dwell_times*(1/sampling_frequency))

    two_dim_dwell_time_hist = np.flip(
        np.histogram2d(logged_lower_dwell_times, logged_upper_dwell_times, bins=60, range=[[-5, 1], [-5, 1]])[0],
        axis=0)

    if(detailed_balance == True):
        upper_dwell_times = dwell_times[np.arange(0, dwell_times.shape[0], 2)]
        lower_dwell_times = dwell_times[np.arange(1, dwell_times.shape[0], 2)]
        upper_dwell_times = np.delete(upper_dwell_times, 0)

        if (upper_dwell_times.shape > lower_dwell_times.shape):
            upper_dwell_times = np.delete(upper_dwell_times, -1)

        elif (upper_dwell_times.shape < lower_dwell_times.shape):
            lower_dwell_times = np.delete(lower_dwell_times, -1)

        logged_upper_dwell_times_DB = np.log10(upper_dwell_times * (1 / sampling_frequency))
        logged_lower_dwell_times_DB = np.log10(lower_dwell_times * (1 / sampling_frequency))

        two_dim_dwell_time_hist_DB = np.flip(
            np.histogram2d(logged_lower_dwell_times_DB, logged_upper_dwell_times_DB, bins=60, range=[[-5, 1], [-5, 1]])[0],
            axis=0)

        two_dim_dwell_time_hist = two_dim_dwell_time_hist + np.flip(two_dim_dwell_time_hist_DB.T,axis=(0,1))

    return two_dim_dwell_time_hist


def exe(time_series,t_res,order,half_jump_magnitude,O_lvl,C_lvl):
    print("computing")
    threshold = compute_threshold(t_res, order, half_jump_magnitude)
    print('start HOHD')
    start_time = time.time()
    jump_list = HOHD(time_series, O_lvl, C_lvl, threshold)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time: ", elapsed_time)
    print('end HOHD')
    temp = two_dim_hist(jump_list, True)
    return temp


if __name__ == "__main__":

    histograms = np.zeros((data_size,60,60))
    for i in range(data_size):
        histograms[i] = np.flip(exe(ts[i], t_res, order, half_jump_magnitude,lower_lvl, upper_lvl),axis = 0).T

    histograms = np.squeeze(histograms, axis=0)
    np.save(path + f"{histogram_dataset_name}", histograms)
