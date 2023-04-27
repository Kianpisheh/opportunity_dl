import os

import pandas as pd
import numpy as np
import scipy
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


def prepare_opportunity_objects_acc(path, milli_g=False):

    columns_idxs = [0, 134, 135, 136, 139, 140, 141, 144, 145, 146, 149, 150, 151,
                    154, 155, 156, 159, 160, 161, 164, 165, 166, 169, 170, 171,
                    174, 175, 176, 179, 180, 181, 184, 185, 186, 189, 190, 191,
                    207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218,
                    219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 244]


    sensors = ["cup", "sallami", "water", "cheese", "bread", "knife1",
               "milk", "spoon", "sugar", "knife2", "plate", "glass",
               "door1", "lazrchair", "door2", "dishwasher", "upperdrawer",
               "lowerdrawer", "middledrawer", "fridge"]  
    
    # read sensor data
    dirs = os.listdir(path)
    samples = []
    # objects_name_np = np.unique(list(objects_name.values()))
    for d in dirs:
        if ".dat" not in d or "ADL" not in d:
            continue
        sensor_data = pd.read_csv(path + "/" + d, delimiter=" ", header=None)
        desired_sensor_data = sensor_data.iloc[:, columns_idxs]

        if not milli_g:
            desired_sensor_data.iloc[:,1:-1] = desired_sensor_data.iloc[:,1:-1]/1000

        desired_sensor_data = fill_nan(desired_sensor_data)
        desired_sensor_data = desired_sensor_data[~np.isnan(desired_sensor_data).any(axis=1)]

        # fig, axes = plt.subplots(6,10)
        # for i in range(6):
        #     for j in range(10):
        #         idx0 = i*10 + j
        #         axes[i,j].plot(desired_sensor_data.iloc[:,1+ idx0])
        #         axes[i,j].set_title(columns_idxs[1+ idx0])

        
        # filtering
        nyquist_freq = 0.5 * 30
        normal_cutoff = 0.1 / nyquist_freq
        b, a = butter(5, normal_cutoff, btype='highpass', analog=False)
        if desired_sensor_data.shape[0] < 10:
            print("data sample skipped")
            continue
        desired_sensor_data.iloc[:, 1:-1] = filtfilt(b, a, desired_sensor_data.iloc[:, 1:-1], axis=0)


        # low pass filter to remove noise
        nyquist_freq = 0.5 * 30
        normal_cutoff = 0.7 / nyquist_freq
        b, a = butter(5, normal_cutoff, btype='low', analog=False)
        desired_sensor_data.iloc[:, 1:-1] = filtfilt(b, a, desired_sensor_data.iloc[:, 1:-1], axis=0)
    

        mags = []
        for s in range(20):
            sensor_acc = desired_sensor_data.iloc[:,1+s*3:1+(s+1)*3]
            sensor_mag = np.linalg.norm(sensor_acc, axis=1)
            mags.append((sensor_mag > 0.25)*1)
        
        # fuse sensor magnitudes
        sensor_magnitudes = np.array(mags).T


        # ------visualization-------#
        # desired_sensor_data.iloc[:,1:-1] = desired_sensor_data.iloc[:,1:-1] > 0.1
        # for i in range(6):
        #     for j in range(10):
        #         idx0 = i*10 + j
        #         axes[i,j].plot(desired_sensor_data.iloc[:,1+ idx0])
        #         axes[i,j].set_title(columns_idxs[1+ idx0])
        # fig2, axes2 = plt.subplots(4,5, sharex=True, sharey=True)
        # for i in range(4):
        #     for j in range(5):
        #         idx0 = i*5 + j
        #         axes2[i,j].plot(mags[idx0])
        #         axes2[i,j].set_title(sensors[idx0])
        # fig3, ax3 = plt.subplots()
        # all_mags = np.zeros(mags[0].shape)
        # for mag in mags:
        #     all_mags += mag
        # ax3.plot(all_mags)
        # plt.show()


        # find activity intervals
        timestamps = desired_sensor_data.values[:,0].reshape((-1,1))
        activity_column_data = desired_sensor_data.values[:,-1]
        activity_intervals = np.where(np.diff(activity_column_data) != 0)[0]
        activity_intervals = np.insert(activity_intervals, 0, 0)
        activity_intervals = np.insert(activity_intervals, len(activity_intervals), activity_column_data.shape[0])

        for i in range(0, len(activity_intervals)-1):
            sample_i = sensor_magnitudes[activity_intervals[i]+1:activity_intervals[i+1]-1,:]
            sample_i = np.hstack((timestamps[activity_intervals[i]+1:activity_intervals[i+1]-1], sample_i, np.ones((sample_i.shape[0], 1))* activity_column_data[activity_intervals[i]+1]))
            if sample_i.shape[0] < 5:
                continue
            if sample_i[0,-1] != sample_i[-1,-1]:
                raise Exception
            # ignore relaxing activity
            activity_num = sample_i[0, -1]
            if activity_num == 0 or activity_num == 101:
                continue
            samples.append(sample_i)
           

    return samples

    
def get_object_usage_feature_vector(data_samples):

    all_segments_featurs = []
    for i, sample in enumerate(data_samples):
        # segmentation
        segments = get_segments(sample, 15000, 0.2)

        # feature extraction (usage percentage)
        segments_featurs = []
        for segment in segments:
            features = get_object_usage_feature(segment)
            if sum(features) == 0: # ignore all zeros features
                continue
            segments_featurs.append(features)

        all_segments_featurs.append(np.array(segments_featurs))


    return all_segments_featurs


def get_object_usage_feature(segment):

    # find object usage/movement (percentage)
    features_vector = np.zeros((1, segment.shape[1]))
    features_vector[0,0] = segment[0,0]
    features_vector[0,-1] = segment[0,-1]
    features_vector[0,1:-1] = np.mean(segment[:,1:-1], axis=0).tolist()

    return [element for sublist in features_vector.tolist() for element in sublist]
              

def get_segments(data_sample, win_size, overlap=0.1): 
    segments = []
    start_idx = 0
    next_start_idx = 0

    while data_sample[-1,0] - data_sample[start_idx,0] > win_size:
        prev_diff = 0
        next_idx_found = False
        for i in range(start_idx+1, data_sample.shape[0]):
            current_diff = data_sample[i,0] - data_sample[start_idx,0]

            # if current_diff > win_size:
            #     start_idx += 1
            #     break

            # segments overlap
            if current_diff >= (1-overlap) * win_size and not next_idx_found:

                # check for sensor stroke
                if current_diff > win_size:
                    start_idx = i + 1 # skip the whole drama
                    break

                if abs(current_diff - (1-overlap) * win_size) <= abs(prev_diff - (1-overlap) * win_size):
                    next_start_idx = i
                else:
                    next_start_idx = i - 1

                next_idx_found = True


            if current_diff >= win_size:
                if abs(current_diff - win_size) <= abs(prev_diff - win_size):
                    if i + 1 - start_idx > 5:
                        segments.append(data_sample[start_idx:i+1])
                else:
                    if i - start_idx > 5:
                        segments.append(data_sample[start_idx:i])

                start_idx = next_start_idx
                break
            prev_diff = current_diff

    return segments



def get_features(segment):
    
    features = []
    sampling_rate = 1000 / np.mean(np.diff(segment[:,0])) 
    
    # segment: T x (1 + num_sensors)
    sensor_data = segment[:,1:-1]

    # features
    means = np.mean(sensor_data, axis=0)
    stds = np.std(sensor_data, axis=0)
    variances = stds**2
    magnitudes = np.linalg.norm(sensor_data, axis=0)
    diff_magnitudes = np.linalg.norm(np.diff(sensor_data, axis=0), axis=0)
    entropies = scipy.stats.entropy(sensor_data, axis=0)
    entropies[np.where(entropies == -np.inf)[0]] = 0.00001
    corr_coeffs_matrix = np.corrcoef(sensor_data, rowvar=False)
    corr_coeffs = corr_coeffs_matrix[~np.eye(corr_coeffs_matrix.shape[0],dtype=bool)]
    zero_crossings = np.sum(np.diff(np.sign(sensor_data), axis=0) != 0, axis=0) / sensor_data.shape[0]
    fft_coeff = np.abs(np.fft.fft(sensor_data, axis=0))
    fft_freqs = np.fft.fftfreq(fft_coeff.shape[0], 1 / sampling_rate)
    fft_coeff = fft_coeff[:int(sensor_data.shape[0]/2)]
    fft_freqs = fft_freqs[:int(sensor_data.shape[0]/2)]
    mean_freqs = np.dot(fft_coeff.T, fft_freqs) / fft_coeff.sum(axis=0)
    fft_magnitudes = np.linalg.norm(fft_coeff, axis=0)
    fft_energies = np.sum(fft_coeff * fft_coeff, axis=0)
    

    # feature feasion
    features.extend(means.tolist())
    features.extend(stds.tolist())
    features.extend(variances.tolist())
    features.extend(magnitudes.tolist())
    features.extend(diff_magnitudes.tolist())
    features.extend(entropies.tolist())
    features.extend(zero_crossings.tolist())
    features.extend(mean_freqs.tolist())
    features.extend(fft_magnitudes.tolist())
    features.extend(fft_energies.tolist())

    features.extend(corr_coeffs.tolist())
    

    return features


def prepare_opportunity_bindary(path):

    objects_name = {
        501:"Bottle",
        502:"Salami",
        503:"Bread",
        504:"Sugar",
        505:"Dishwasher",
        506:"Switch",
        507:"Milk",
        508:"Drawer3 (lower)",
        509:"Spoon",
        510:"Knife cheese",
        511:"Drawer2 (middle)",
        512:"Table",
        513:"Glass",
        514:"Cheese",
        515:"Chair",
        516:"Door1",
        517:"Door2",
        518:"Plate",
        519:"Drawer1 (top)",
        520:"Fridge",
        521:"Cup",
        522:"Knife salami",
        523:"Lazychair",
        301:"Bottle",
        302:"Salami",
        303:"Bread",
        304:"Sugar",
        305:"Dishwasher",
        306:"Switch",
        307:"Milk",
        308:"Drawer3 (lower)",
        309:"Spoon",
        310:"Knife cheese",
        311:"Drawer2 (middle)",
        312:"Table",
        313:"Glass",
        314:"Cheese",
        315:"Chair",
        316:"Door1",
        317:"Door2",
        318:"Plate",
        319:"Drawer1 (top)",
        320:"Fridge",
        321:"Cup",
        322:"Knife salami",
        323:"Lazychair"
    }

    activity_name = {
        101:"Relaxing",
        102:"Coffee time",
        103:"Early morning",
        104:"Cleanup",
        105:"Sandwich time",
    }

    left_object_column = 247
    right_object_column = 248
    activity_column = 243

  
    # read sensor data
    dirs = os.listdir(path)
    samples = []
    objects_name_np = np.unique(list(objects_name.values()))
    for d in dirs:
        if ".dat" not in d:
            continue
        sensor_data = pd.read_csv(path + "/" + d, delimiter=" ", header=None)
        desired_sensor_data = sensor_data.iloc[:, [0, left_object_column, right_object_column, activity_column]]
        
        activity_column_data = desired_sensor_data.values[:,3]
        activity_intervals = np.where(np.diff(activity_column_data) != 0)[0]
        start_idx = 0
        for end_idx in activity_intervals:
            sample_i = desired_sensor_data.values[start_idx:end_idx+1,:]
            start_idx = end_idx + 1
            samples.append(sample_i)
        samples.append(desired_sensor_data.values[start_idx:-1,:])

    
    activities_df_list = []
    for s in samples:
        s_df = pd.DataFrame(np.zeros((s.shape[0], len(objects_name_np))), columns=objects_name_np, dtype=int)
        for obj in objects_name:
            idxs_left = np.where(s[:,1] == obj)[0]
            idxs_right = np.where(s[:,2] == obj)[0]
            col = objects_name[obj]
            oo = obj
            if obj < 400:
                oo += 200
            if idxs_left.shape[0] > 0:
                s_df[col].iloc[idxs_left] = oo
            if idxs_right.shape[0] > 0:
                s_df[col].iloc[idxs_right] = oo

        activities_df_list.append(s_df)


    return activities_df_list



def prepare_opportunity_onbody(path, milli_g=False):
    columns_idxs = [
                    0, 7, 8, 9, 10, 11, 12, 22, 23, 24, 
                    25, 26, 27, 28, 29, 30, 31, 32, 33, 244
                    ]  

    # read sensor data
    dirs = os.listdir(path)
    samples = []
    # objects_name_np = np.unique(list(objects_name.values()))
    for d in dirs:
        if ".dat" not in d or "ADL" not in d:
            continue
        sensor_data = pd.read_csv(path + "/" + d, delimiter=" ", header=None)
        desired_sensor_data = sensor_data.iloc[:, columns_idxs]
        if not milli_g:
            desired_sensor_data.iloc[:,1:-1] = desired_sensor_data.iloc[:,1:-1]/1000

        # # handle nan values
        desired_sensor_data = fill_nan(desired_sensor_data)
        desired_sensor_data = desired_sensor_data[~np.isnan(desired_sensor_data).any(axis=1)]

        fig, axes = plt.subplots(3,6)
        for i in range(3):
            for j in range(6):
                idx0 = i*6 + j
                axes[i,j].plot(desired_sensor_data.iloc[:,1+ idx0])
                axes[i,j].set_title(columns_idxs[1+ idx0])
                axes.set_ylim(-1, 1)


        nyquist_freq = 0.5 * 30
        normal_cutoff = 0.1 / nyquist_freq
        b, a = butter(5, normal_cutoff, btype='highpass', analog=False)
        w, h = scipy.signal.freqs(b, a)
        fig2, ax = plt.subplots()
        ax.semilogx(w, 20 * np.log10(abs(h)))
        plt.title('Butterworth filter frequency response')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude [dB]')
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')

        desired_sensor_data.iloc[:, 1:-1] = filtfilt(b, a, desired_sensor_data.iloc[:, 1:-1], axis=0)


        for i in range(3):
            for j in range(6):
                idx0 = i*6 + j
                axes[i,j].plot(desired_sensor_data.iloc[:,1+ idx0])
                axes[i,j].set_title(columns_idxs[1+ idx0])
                axes.set_ylim(-1, 1)

        plt.show()

        activity_column_data = desired_sensor_data.values[:,-1]
        activity_intervals = np.where(np.diff(activity_column_data) != 0)[0]
        activity_intervals = np.insert(activity_intervals, 0, 0)
        activity_intervals = np.insert(activity_intervals, len(activity_intervals), activity_column_data.shape[0])
        for i in range(0, len(activity_intervals)-1):
            sample_i = desired_sensor_data.values[activity_intervals[i]+1:activity_intervals[i+1]-1,:]
            sample_i2 = sample_i[~np.isnan(sample_i).any(axis=1)]
            if sample_i[0,-1] != sample_i[-1,-1]:
                raise Exception
            # ignore relaxing activity
            activity_num = sample_i[0, -1]
            if activity_num == 0 or activity_num == 101:
                continue
            samples.append(sample_i)
           

    return samples


def fill_nan(sensor_data):
    # th: sec

    timestamps = sensor_data.iloc[:,0]
    for c in range(1, sensor_data.shape[1]-1):
        
        col = sensor_data.iloc[:,c]
        nan_indeces0 = np.where(np.isnan(col))[0]
        nan_indeces = np.where(np.diff(np.where(np.isnan(col)))[0] > 1)[0]
        nan_indeces += 1
        nan_indeces = np.insert(nan_indeces, 0, 0)
        
        for i in range(nan_indeces.shape[0]-1):
            start_idx = nan_indeces0[nan_indeces[i]]
            end_idx = nan_indeces0[nan_indeces[i+1]-1]
            if end_idx - start_idx > 40:
                continue

            # interpolation
            if start_idx == 0:
                a = 0
            else:
                a = col[start_idx-1]
            if end_idx == timestamps.shape[0] -1:
                b = 0
            else:
                b = col[end_idx+1]
            
            d = timestamps[end_idx+1]
            if start_idx != 0:
                d = d - timestamps[start_idx-1]
            for idx in range(start_idx, end_idx+1):
                d_i = timestamps[idx]
                if start_idx != 0:
                    d_i = d_i - timestamps[start_idx-1]
                sensor_data.iloc[idx, c] = ((1 - d_i/d) * a) + ((d_i/d) * b)


    return sensor_data 