# Preprocess EEG data for single subject using preprocessing_module.py
# EEG Study 1
# Lab: William Hardy Building, Bekinstein Lab, University of Cambridge
# Equipment: EGI 128 channel HydroCel Geodesic Sensor Net, Net Station
# Author: Ekim Luo
# Last Updated: 2023/9/13
import sys
sys.path.append("..")
from grape.preprocessing_module import Participant
import mne
import pandas as pd
import os

pid = '002'
data_dir = '../data'

rejection_df = pd.DataFrame(columns=['pid', 'total_epochs', 'dropped_epochs', 'perc_dropped'])
amplitude_latency_df = pd.DataFrame(columns=['pid', 'component', 'amplitude_peak', 'latency_peak'])

cleaned_data_dir = 'cleaned_data'
cleaned_data = 'amplitude_latency_metrics.csv'
rejection_data = 'rejection_metrics.csv'

# Create the full path to the csv file
cleaned_file_path = os.path.join(cleaned_data_dir, cleaned_data)
rejection_file_path = os.path.join(cleaned_data_dir, rejection_data)

# Check if csv files exist or create them
if not os.path.exists(cleaned_file_path):
    amplitude_latency_df.to_csv(cleaned_file_path, index=False)

if not os.path.exists(rejection_file_path):
    rejection_df.to_csv(rejection_file_path, index=False)

# Define channels for each region
N1_channels = ['E22', 'E18', 'E16', 'E10', 'E3', 'E19', 'E11', 'E4', 'E20', 'E12','E5',
               'E26', 'E15', 'E9', 'E2', 'E23', 'E124', 'E118']

N2_channels = ['E26', 'E22', 'E15', 'E9', 'E2', 'E23', 'E18', 'E16', 'E10', 'E3',
               'E24', 'E19', 'E11', 'E4', 'E124', 'E20', 'E12', 'E5', 'E118', 'E13', 
               'E6', 'E112', 'E7', 'E106']


P3_channels = ['E58', 'E65', 'E70', 'E75', 'E83', 'E90', 'E96', 'E51', 'E59', 'E66', 
               'E71', 'E76', 'E84', 'E91', 'E97', 'E47', 'E52', 'E60', 'E67', 'E72', 
               'E77', 'E85', 'E92', 'E98', 'E42', 'E53', 'E61', 'E62', 'E78', 'E86', 
               'E93', 'E37', 'E54', 'COM', 'E79', 'E87', 'E31', 'E55', 'E80', 'VREF']

LPP_channels = P3_channels  # LPP uses the same channels as P3

time_window_N1 = [0.07, 0.15]
time_window_N2 = [0.2, 0.3]
time_window_P3 = [0.35, 0.45]
time_window_LPP = [0.5, 0.8]

# Load in data for this participant. 
def load_data(pid, data_dir,
              montage = 'GSN-HydroCel-128', 
              channel_types = {'E127': 'eog', 'E126': 'eog', 'E17': 'eog', 'E15': 'eog', 'E21': 'eog', 'E14': 'eog', 'VREF':'misc'}):
    '''
    Load raw EEG data from the filepath
    '''
    mff_file = os.path.join(data_dir, f'{pid}.mff')
    raw_data = mne.io.read_raw_egi(mff_file, preload=True, verbose=0)
    # Set channel types
    raw_data.set_channel_types(channel_types)    
    # Set montage
    raw_data.set_montage(montage)
    return raw_data

my_data = load_data(pid,data_dir)

# Set verbose="WARNING" to see less output from each method. 
p = Participant(pid, data_dir, verbose="INFO")

out = p.add_raw_data(my_data)

p.filter_data()
p.find_bad_channels()
p.remove_bad_channels()
p.fit_ica(method = 'infomax')
p.find_bad_ica()
p.remove_bad_ica()
p.apply_ica()
p.create_epochs()

# Create evoked objects for each component
evoked_n1 = p.create_evoked(comp_channels = N1_channels)
evoked_n2 = p.create_evoked(comp_channels = N2_channels)
evoked_p3 = p.create_evoked(comp_channels = P3_channels)
evoked_lpp = p.create_evoked(comp_channels = LPP_channels)

# Export evoked objects to fif files
mne.Evoked.save(evoked_n1, 'Data/01_EEG_evoked/' + pid + '_N1-ave.fif')
mne.Evoked.save(evoked_n2, 'Data/01_EEG_evoked/' + pid + '_N2-ave.fif')
mne.Evoked.save(evoked_p3, 'Data/01_EEG_evoked/' + pid + '_P3-ave.fif')
mne.Evoked.save(evoked_lpp, 'Data/01_EEG_evoked/' + pid + '_LPP-ave.fif')

# Find peak amplitude and latency for each component
amp_n1, lat_n1 = p.find_peaks(evoked_n1, time_window_N1)
amp_n2, lat_n2 = p.find_peaks(evoked_n2, time_window_N2)
amp_p3, lat_p3 = p.find_peaks(evoked_p3, time_window_P3)
amp_lpp, lat_lpp = p.find_peaks(evoked_lpp, time_window_LPP)

# Append rejection metrics to DataFrame
new_metrics = pd.DataFrame(
    {'pid': pid,
     'total_epochs': p.total_epochs,
     'dropped_epochs': p.dropped_epochs,
     'perc_dropped': p.perc_dropped},
    index=[0])

rejection_df = pd.concat([rejection_df.astype(new_metrics.dtypes),
                          new_metrics], ignore_index=True)

# Append amplitude and latency metrics to DataFrame for all components
new_metrics = pd.DataFrame(
    {'pid': [pid,pid,pid,pid],
     'component': ['N1','N2','P3','LPP'],
     'amplitude_peak': [amp_n1,amp_n2,amp_p3,amp_lpp],
     'latency_peak': [lat_n1,lat_n2,lat_p3,lat_lpp]},)

amplitude_latency_df = pd.concat([amplitude_latency_df.astype(new_metrics.dtypes),
                                  new_metrics],ignore_index=True)

# Export DataFrames to csv
rejection_df.to_csv(rejection_file_path, mode = 'a', header = True, index=False)
amplitude_latency_df.to_csv(cleaned_file_path, mode = 'a', header = False, index=False)

print('Participant ' + pid + ' processed.')