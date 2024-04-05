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
    def __init__(self, pid, filepath, **kwargs):
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
        
        self.kwargs = kwargs
        
    def __check_verbosity(self,kwargs):
        v = kwargs.get("v",
                       kwargs.get("verbose",
                                  self.kwargs.get("v",
                                                  self.kwargs.get("verbose",20))))
        # Match MNE verbosity settings
        if isinstance(v,str):
            v = {"debug":10,"info":20,"warning":30,"error":40,"critical":50}.get(v.lower(),20)
        elif isinstance(v,bool):
            v = 20 if v else 30
        return v
        
    def load_data(self, montage = 'GSN-HydroCel-128', channel_types = {'E127': 'eog', 'E126': 'eog', 'E17': 'eog', 'E15': 'eog', 'E21': 'eog', 'E14': 'eog', 'VREF':'misc'}):
        '''
        Load raw EEG data from the filepath.
        
        NOTE: Needs to be removed. The code is too use-case specific, user-written functions should prepare data to be added via add_raw_data method. 
        '''
        mff_file = os.path.join(self.filepath, f'{self.pid}.mff')
        self.raw_data = mne.io.read_raw_egi(mff_file, preload=True)
        # Set channel types
        self.raw_data.set_channel_types(channel_types)    
        # Set montage
        self.raw_data.set_montage(montage)
        return self.raw_data
    
    def add_raw_data(self,raw_data):
        """Assign raw_data to this participant. 
        The raw_data should be an instance of RawEGI. 
        See the mne documentation (mne.io.Raw) for attributes and methods.

        Args:
            raw_data (raw): instance of RawEGI.

        Returns:
            raw: RawEGI instance that has been added to the object as raw_data. 
        """
        self.raw_data = raw_data
        return self.raw_data
    
    def filter_data(self, highpass = 0.1, lowpass = 40, **kwargs):
        """Filter the participant's raw data. 

        Args:
            highpass (float, optional): High-pass cutoff frequency. Defaults to 0.1.
            lowpass (int, optional): Low-pass cutoff frequency. Defaults to 40.

        Returns:
            raw: RawEGI instance corresponding to filtered data. 
        """
        v = self.__check_verbosity(kwargs) # Print-output verbosity.
        
        self.filtered_data = self.raw_data.copy().filter(highpass, lowpass, verbose=v)
        return self.filtered_data
    
    def find_bad_channels(self, exclude_from_bad = ['STRT', 'fixc', 'resp', 'STI 014', 'stim', 'VREF'],**kwargs):
        """Identify bad channels using ISD.

        Args:
            exclude_from_bad (list, optional): Channel names to exclude from the search for bad channels. Defaults to ['STRT', 'fixc', 'resp', 'STI 014', 'stim', 'VREF'].
        """
        v = self.__check_verbosity(kwargs) # Print-output verbosity.
        
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
        if v<30:
            print('There are {} bad channels found using automated bad channel detection for subject {}'.format(len(self.bad_channels_names), self.pid))

    def remove_bad_channels(self, bad_channels_names = None, exclusion_thresh = 0.75, **kwargs):
        """Remove bad channels.

        Args:
            bad_channels_names (list, string, optional): Names of bad channels. Defaults to None.
            exclusion_thresh (float, optional): Fraction of channels that can be excluded before an error is thrown. Defaults to 0.75.

        Raises:
            ValueError: An error is raised if the fraction of bad channels removed exceeds the exclusion_threshold. 
        """
        v = self.__check_verbosity(kwargs) # Print-output verbosity.
        # Channels to exclude from bad channel detection
        if bad_channels_names is None:
            bad_channels_names = self.bad_channels_names
        self.no_bads_data = self.filtered_data.copy().drop_channels(self.bad_channels_names)
    
        # Check if total bad channels exceeds threshold for participant exclusion in %
        total_channels = len(self.filtered_data.ch_names)
        total_bads = len(self.bad_channels_names)
        if total_bads > exclusion_thresh * total_channels:
            raise ValueError(f'More than {exclusion_thresh * 100}% bad channels found for subject {self.pid}. Please check the data.')
        
        if v<30:
            print('{} bad channels were removed for subject {}'.format(len(self.bad_channels_names), self.pid))
    
    def fit_ica(self, data = None, resamp_sfreq = 250.0, ncomps=15, method=['infomax', 'fastica', 'picard'],**kwargs):
        """Fit ICA to filtered data excluding bad channels. 
        See mne.preprocessing.ICA for documentation of the algorithm. 

        Args:
            data (raw, optional): RawEGI instance representing data to fit ICA to. If None, self.no_bads_data is used. Defaults to None.
            resamp_sfreq (float, optional): Resampling frequency for data. Defaults to 250.0.
            ncomps (int, optional): Number of components (passed to ICA algorithm). Defaults to 15.
            method (list, optional): Methods to use in ICA fitting (passed to ICA algorithm).. Defaults to ['infomax', 'fastica', 'picard'].

        Returns:
            ica: ICA instance. 
        """
        v = self.__check_verbosity(kwargs) # Print-output verbosity.
        
        if data is None:
            data = self.no_bads_data.copy()
        
        # Downsample data to 250 Hz for memory efficiency
        data.resample(sfreq=resamp_sfreq,verbose=v,**kwargs)

        self.ica = ICA(n_components=ncomps, random_state=42, method=method, verbose=v, **kwargs)
        self.ica.fit(data,verbose=v,**kwargs)

        return self.ica

    def find_bad_ica(self, data = None, sfreq=500.0, highpass=0.1, lowpass=40, rej_thresh=2.0, 
                     eog_channels=['E127','E126','E17','E15','E21','E14'],
                     **kwargs):
        """Find bad ICA components using FASTER. 
        Can supply kwargs to pass to the Hurst exponent algorithm: see the documentation for nolds.hurst_rs(). 

        Args:
            data (raw, optional): RawEGI instance representing data to fit ICA to. If None, self.no_bads_data is used. Defaults to None.
            sfreq (float, optional): _description_. Defaults to 500.0.
            highpass (float, optional): High-pass cutoff frequency for filter band slope calculation. Defaults to 0.1.
            lowpass (int, optional): Low-pass cutoff frequency for filter band slope calculation. Defaults to 40.
            rej_thresh (float, optional): Rejection threshold for FASTER. Defaults to 2.0.
            eog_channels (list, string, optional): EOG channel names. Defaults to ['E127','E126','E17','E15','E21','E14'].
        """
        v = self.__check_verbosity(kwargs) # Print-output verbosity.
        
        if data is None:
            data = self.no_bads_data.copy()

        # Get ICA scores
        self.ica_sources = self.ica.get_sources(data,**kwargs)
        self.ica_scores = self.ica_sources.get_data(verbose=v,**kwargs)

        # Check if any of the EOG channels exist
        eog_avail = True
        avail_eogs = [ch for ch in eog_channels if ch in data.ch_names]
        if not avail_eogs:
            print('No EOG channels found for subject {}'.format(self.pid))
            eog_avail = False
        else:
            # Correlate ICA scores with EOG channels
            eog_data = data.copy().pick(avail_eogs,verbose=v,**kwargs)
            self.eog_scores = eog_data.get_data(verbose=v,**kwargs)[0]

            cor_eog = np.corrcoef(self.ica_scores, self.eog_scores)
            zscore_eog = zscore(cor_eog)

        # Calculate spatial kurtosis of ICA components
        self.ica_maps = self.ica.get_components()
        kurtosis_vals = np.sum(self.ica_maps**4, axis=1) / np.sum(self.ica_maps**2, axis=1)**2
        zscore_kurtosis = zscore(kurtosis_vals)

        # Calculate filter band slope
        ica_psd, freqs = mne.time_frequency.psd_array_welch(self.ica_scores, fmin=highpass, fmax=lowpass, sfreq=sfreq, verbose=v, **kwargs)
        slope = [np.mean(np.gradient(psd)) for psd in ica_psd]
        zscore_slope = zscore(slope)

        # Calculate the Hurst exponent for each ICA component
        hurst_values = [nolds.hurst_rs(score,**kwargs) for score in self.ica_scores]
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
        if v<30:
            print('There are {} bad ICA components for subject {}'.format(len(self.ica_bads), self.pid))
        
    def remove_bad_ica(self, ica_bads = None, **kwargs):
        """Remove bad ICA components.

        Args:
            ica_bads (list, optional): List of bad ICA components. If None, uses self.ica_bads. Defaults to None.

        """
        v = self.__check_verbosity(kwargs) # Print-output verbosity.
        
        if ica_bads is None:
            ica_bads = self.ica_bads

        self.ica_bads_idx = [self.ica_sources.info['ch_names'].index(i) for i in ica_bads]
        self.ica.exclude = self.ica_bads_idx

        if v<30:
            print('{} bad ICA components removed for subject {}'.format(len(self.ica_bads_idx), self.pid))
    
    def apply_ica(self, data = None, exclude = None, **kwargs):
        """Apply ICA to filtered data excluding bad channels

        Args:
            data (raw, optional): RawEGI instance. If None, uses self.no_bads_data. Defaults to None.
            exclude (list, optional): Bad channels to exclude. If None, self.ica.exclude is extended using self.ica_bads_idx, and the resulting list is used. Defaults to None.

        Returns:
            Raw, Epochs or Evoked: Processed data
        """
        v = self.__check_verbosity(kwargs) # Print-output verbosity.
        
        if data is None:
            data = self.no_bads_data.copy()
        if exclude is None:
            self.ica.exclude.extend(self.ica_bads_idx)
            exclude = self.ica.exclude
        self.postica_data = self.ica.apply(data, exclude = exclude, verbose=v,**kwargs)
        return self.postica_data
    
    def create_epochs(self, data = None, stim_channel = 'stim', tmin = -0.2, tmax = 1.2, reject = dict(eeg = 75e-4), baseline = (-0.2, 0), **kwargs):
        """Create epochs from cleaned data. 
        
        The stim_channel arg is supplied to mne.find_events, and the others args are supplied to mne.Epochs. 
        See the relevant mne documentaton for details.

        Args:
            data (_type_, optional): cleaned data. If None, self.postica_data is used. Defaults to None.
            stim_channel (str, optional): String specifying the stim channel. See mne.find_events documentation for details. Defaults to 'stim'.
            tmin (float, optional): Minimum time cutoff. Defaults to -0.2.
            tmax (float, optional): Maximum time cutoff. Defaults to 1.2.
            reject (dict, optional): Dictionary of rejection thresholds. Defaults to dict(eeg = 75e-4).
            baseline (tuple, optional): Baseline values. Defaults to (-0.2, 0).

        Returns:
            Epochs: mne.Epochs instance of epochs created by this function. 
        """
        v = self.__check_verbosity(kwargs) # Print-output verbosity
        
        if data is None:
            data = self.postica_data.copy()
        
        self.events = mne.find_events(data, stim_channel = stim_channel, verbose=v, **kwargs)
        self.event_id = np.unique(self.events[:, 2])[0]
        self.events = self.events[self.events[:, 2] == self.event_id]
        self.epochs = mne.Epochs(data, events=self.events, event_id=self.event_id, tmin=tmin, tmax=tmax, baseline=baseline, reject=reject, preload=True, verbose=v, **kwargs)
        
        # Calculate epoch dropped percentage
        self.total_epochs = len(self.epochs.drop_log)
        self.dropped_epochs = sum(1 for log in self.epochs.drop_log if len(log) > 0)
        self.perc_dropped = self.dropped_epochs / self.total_epochs * 100

        # Reference epochs to average reference
        self.epochs.set_eeg_reference('average',verbose=v,**kwargs)
        return self.epochs
    
    def create_evoked(self, comp_channels, **kwargs):
        """Create evoked data from epochs.

        Args:
            comp_channels (list, string): A list of channel names to pick out. 

        Returns:
            EvokedArray: evoked data. 
        """
        v = self.__check_verbosity(kwargs) # Print-output verbosity
        
        avail_channels = [ch for ch in comp_channels if ch in self.epochs.ch_names]
        
        epochs_comp = self.epochs.copy().pick(avail_channels,verbose=v,**kwargs)
        evoked = epochs_comp.average()
        return evoked
    
    def find_peaks(self, evoked, tw = [0.2, 0.3]):
        """Find the peak amplitude and latency in the time window for the component.

        Args:
            evoked (EvokedArray): evoked data to finds peaks in. 
            tw (list, float, optional): Time window to use. Defaults to [0.2, 0.3].

        Returns:
            tuple, float: Values corresponding to the amplitude peak and latency peak. 
        """
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
        """Generate plots for data at each preprocessing stage.

        Args:
            data (raw, optional): Data to plot. If None, self.filtered_data is used. Defaults to None.
            data_title (str, optional): Named of data - used as plot title. Defaults to 'Filtered'.
            scalings (dict, optional): Dictionary of scalings to use. Defaults to dict(eeg=300e-6).
            show (bool, optional): Whether to show the plots that are produced. Defaults to True.
        """
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