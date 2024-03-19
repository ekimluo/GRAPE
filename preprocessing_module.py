# Create module for preprocessing EEG data
# EEG Study 1
# Lab: William Hardy Building, Bekinstein Lab, University of Cambridge
# Equipment: EGI 128 channel HydroCel Geodesic Sensor Net, Net Station
# Author: Ekim Luo
# Last Updated: 2023/8/15
import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
import nolds
import pandas as pd
import os

# Create a Participant class to store all attributes and methods for single participant
class Participant:
    def __init__(self, pid, filepath):
        # Basic attributes
        self.pid = pid
        self.filepath = filepath
        self.event_id = None
        self.raw_data = None
        self.events = None
        
        # Preprocessing attributes
        self.filtered_data = None
        
        self.bad_channels = None
        self.exclude_from_bad = None
        self.bad_channels_names = None
        self.bad_channels_manual = None
        self.no_bads_data = None
        
        self.ica = None
        self.ica_sources = None
        self.ica_scores = None
        self.eog_scores = None
        self.ica_maps = None

        self.ica_bads = None
        self.ica_bads_idx = None
        self.postica_data = None
        
        self.epochs = None
        self.event_id = None
        self.events = None
        self.epoch_rejection_rate = None

        # Component attributes
        self.evoked = None
        self.amplitude_peak = None
        self.latency_peak = None
        self.comp = None

        # Plot attributes
        self.fig_raw_psd = None
        self.fig_raw = None
        self.fig_filtered_psd = None
        self.fig_filtered = None
        self.fig_no_bads_psd = None
        self.fig_postica_psd = None
        self.fig_postica = None
        
    def load_data(self, montage = 'GSN-HydroCel-128', channel_types = {'E127': 'eog', 'E126': 'eog', 'E17': 'eog', 'E15': 'eog', 'E21': 'eog', 'E14': 'eog', 'VREF':'misc'}):
        '''
        Load raw EEG data from the filepath
        '''
        mff_file = os.path.join(self.filepath, f'{self.pid}.mff')
        self.raw_data = mne.io.read_raw_egi(mff_file, preload=True)
        # Set channel types
        self.raw_data.set_channel_types(channel_types)    
        # Set montage
        self.raw_data.set_montage(montage)
        return self.raw_data
    
    def filter_data(self, highpass = 0.1, lowpass = 40):
        '''
        Filter raw data
        '''
        self.filtered_data = self.raw_data.copy().filter(highpass, lowpass)
        return self.filtered_data
    
    def find_bad_channels(self, exclude_from_bad = ['STRT', 'fixc', 'resp', 'STI 014', 'stim', 'VREF']):
        '''
        Identify bad channels using ISD.
        '''
        eeg_data = self.filtered_data.get_data()
        # Compute the standard deviation for each channel
        std = eeg_data.std(axis=1)
        std_median = np.median(std)
        # Calculate the 75th percentile of standard deviations across all channels
        perc_75 = np.percentile(std, 75)
        std_thresh_lower = 1e-4
        std_thresh_upper = 100
        std_p = std.std()
        std_thresh_fixed = 5

        self.bad_channels = []

        for j in range(eeg_data.shape[0]):
            if np.abs(std[j] - std_median) > perc_75:
                self.bad_channels.append(j)
            elif std[j] < std_thresh_lower or std[j] > std_thresh_upper:
                self.bad_channels.append(j)
            elif std[j] > std_thresh_fixed * std_p:
                self.bad_channels.append(j)
            elif j == 0:
                self.bad_channels.append(j)
        
        self.bad_channels_names = [self.filtered_data.ch_names[i] for i in self.bad_channels if self.filtered_data.ch_names[i] not in exclude_from_bad]
        return 'There are {} bad channels found using automated bad channel detection for subject {}'.format(len(self.bad_channels_names), self.pid)

    def remove_bad_channels(self, bad_channels_names = None, exclusion_thresh = 0.75):
        '''
        Remove bad channels
        ''' 
        # Channels to exclude from bad channel detection
        if bad_channels_names is None:
            bad_channels_names = self.bad_channels_names
        self.no_bads_data = self.filtered_data.copy().drop_channels(self.bad_channels_names)
        self.no_bads_data = self.filtered_data.copy().drop_channels(self.bad_channels_names)
    
        # Check if total bad channels exceeds threshold for participant exclusion in %
        total_channels = len(self.filtered_data.ch_names)
        total_bads = len(self.bad_channels_names)
        if total_bads > exclusion_thresh * total_channels:
            raise ValueError(f'More than {exclusion_thresh * 100}% bad channels found for subject {self.pid}. Please check the data.')
        
        return '{} bad channels were removed for subject {}'.format(len(self.bad_channels_names), self.pid)
    
    def fit_ica(self, data = None, resamp_sfreq = 250.0, ncomps=15, method=['infomax', 'fastica', 'picard']):
        '''
        Fit ICA to filtered data excluding bad channels
        '''
        if data is None:
            data = self.no_bads_data.copy()
        
        # Downsample data to 250 Hz for memory efficiency
        data.resample(sfreq=resamp_sfreq)

        self.ica = ICA(n_components=ncomps, random_state=42, method=method)
        self.ica.fit(data)

        return self.ica

    def find_bad_ica(self, data = None, sfreq=500.0, highpass=0.1, lowpass=40, rej_thresh=2, eog_channels=['E127','E126','E17','E15','E21','E14']):
        '''
        Find bad ICA components using FASTER
        '''
        if data is None:
            data = self.no_bads_data.copy()

        # Get ICA scores
        self.ica_sources = self.ica.get_sources(data)
        self.ica_scores = self.ica_sources.get_data()

        # Check if any of the EOG channels exist
        eog_avail = True
        avail_eogs = [ch for ch in eog_channels if ch in data.ch_names]
        if not avail_eogs:
            print('No EOG channels found for subject {}'.format(self.pid))
            eog_avail = False
        else:
            # Correlate ICA scores with EOG channels
            eog_data = data.copy().pick_channels(eog_channels)
            self.eog_scores = eog_data.get_data()[0]

            cor_eog = np.corrcoef(self.ica_scores, self.eog_scores)
            zscore_eog = zscore(cor_eog)

        # Calculate spatial kurtosis of ICA components
        self.ica_maps = self.ica.get_components()
        kurtosis_vals = np.sum(self.ica_maps**4, axis=1) / np.sum(self.ica_maps**2, axis=1)**2
        zscore_kurtosis = zscore(kurtosis_vals)

        # Calculate filter band slope
        ica_psd, freqs = mne.time_frequency.psd_array_welch(self.ica_scores, fmin=highpass, fmax=lowpass, sfreq=sfreq)
        slope = [np.mean(np.gradient(psd)) for psd in ica_psd]
        zscore_slope = zscore(slope)

        # Calculate the Hurst exponent for each ICA component
        hurst_values = [nolds.hurst_rs(score) for score in self.ica_scores]
        zscore_hurst = zscore(hurst_values)

        # Calculate median gradient
        median_gradients = [np.median(np.gradient(score)) for score in self.ica_scores]
        zscore_gradient = zscore(median_gradients)

        # Find bad ICA components using FASTER
        self.ica_bads = []

        for i, name in enumerate(self.ica_sources.info['ch_names']):
            conditions = [
                abs(zscore_kurtosis[i]) > rej_thresh,
                abs(zscore_slope[i]) > rej_thresh,
                abs(zscore_hurst[i]) > rej_thresh,
                abs(zscore_gradient[i]) > rej_thresh
            ]
            if eog_avail:
                conditions.append(abs(zscore_eog[i, 0]) > rej_thresh)
            if any(conditions):
                self.ica_bads.append(name)
        return 'There are {} bad ICA components for subject {}'.format(len(self.ica_bads), self.pid)
        
    def remove_bad_ica(self, ica_bads = None):
        '''
        Remove bad ICA components
        '''
        if self.ica_bads is None:
            self.ica_bads = self.ica_bads

        self.ica_bads_idx = [self.ica_sources.info['ch_names'].index(i) for i in self.ica_bads]
        self.ica.exclude = self.ica_bads_idx

        return '{} bad ICA components removed for subject {}'.format(len(self.ica_bads_idx), self.pid)
    
    def apply_ica(self, data = None, exclude = None):
        '''
        Apply ICA to filtered data excluding bad channels
        '''
        if data is None:
            data = self.no_bads_data.copy()
        if exclude is None:
            self.ica.exclude.extend(self.ica_bads_idx)
            exclude = self.ica.exclude
        self.postica_data = self.ica.apply(data, exclude = exclude)
        return self.postica_data
    
    def create_epochs(self, data = None, stim_channel = 'stim', tmin = -0.2, tmax = 1.2, reject = dict(eeg = 75e-4), baseline = (-0.2, 0)):
        '''
        Create epochs from cleaned data.
        '''
        if data is None:
            data = self.postica_data.copy()
        
        self.events = mne.find_events(data, verbose = False, stim_channel = stim_channel)
        self.event_id = np.unique(self.events[:, 2])[0]
        self.events = self.events[self.events[:, 2] == self.event_id]
        self.epochs = mne.Epochs(data, events=self.events, event_id=self.event_id, tmin=tmin, tmax=tmax, baseline=baseline, reject=reject, preload=True)
        
        # Calculate epoch dropped percentage
        self.total_epochs = len(self.epochs.drop_log)
        self.dropped_epochs = sum(1 for log in self.epochs.drop_log if len(log) > 0)
        self.perc_dropped = self.dropped_epochs / self.total_epochs * 100

        # Reference epochs to average reference
        self.epochs.set_eeg_reference('average')
        return self.epochs
    
    def create_evoked(self, comp_channels):
        '''
        Create evoked data from epochs.
        '''
        epochs_comp = self.epochs.copy().pick_channels(comp_channels)
        evoked = epochs_comp.average()
        return evoked
    
    def find_peaks(self, evoked, tw = [0.2, 0.3]):
        '''
        Find the peak amplitude and latency in the time window for the component.
        '''
        if evoked is None:
            evoked = self.evoked.copy()

        # Find the time window
        mask = np.logical_and(evoked.times >= tw[0], evoked.times <= tw[1])
        data = evoked.data[:, mask]
        
        # Find the absolute peak in the time window
        channel_idx, time_idx = np.unravel_index(np.argmax(np.abs(data)), data.shape)
        self.amplitude_peak = data[channel_idx, time_idx]
        self.latency_peak = evoked.times[mask][time_idx]

        return self.amplitude_peak, self.latency_peak
    
    def generate_plots(self, data=None, data_title='Filtered', scalings=dict(eeg=300e-6), show=True):
        '''
        Generate plots for data at each preprocessing stage.
        '''
        if data is None:
            data = self.filtered_data.copy()

        # Compute and plot PSD
        self.fig_raw_psd = data.plot_psd(fmax=30, show=show)
        # Setting title for the PSD plot
        if self.fig_raw_psd.axes:
            self.fig_raw_psd.axes[0].set_title('{} Data Power Spectral Density Plot for {}'.format(data_title, self.pid))

        # Plot raw data
        self.fig_raw = data.plot(start=5, duration=10, n_channels=20, scalings=scalings, show=show)
        if self.fig_raw:
            self.fig_raw.suptitle('{} Data for {}'.format(data_title, self.pid), y=0.98)

        plt.tight_layout()