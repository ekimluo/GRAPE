## S3     EEG Data Proprocessing
>[!Info]
>This section outlines the detailed steps and rationale for pre-processing EEG data using Python. This document is intended to supplement the EEG study writeup, as well as serving as a free and open resource for my students and others who want to learn how to preprocess EEG data starting with a single subject. The code snippets provided here are not meant to be fully reproducible and the reader needs to supplement the code, such as including raw EEG data, for a fully deployable script.

### Using `mne` 

EEG data were preprocessed in Python using the [`mne`](https://mne.tools/stable/index.html) package. Recordings are imported using the submodule `mne.io`. Given that our recording devices were manufactured by [Electrical Geodesics](https://www.egi.com/), Inc (EGI), we use the function [`mne.io.read_raw_egi()`](https://mne.tools/stable/generated/mne.io.read_raw_egi.html) to import raw data. 

Participant data were contained in `mff` files, which are typical file formats using EGI recording devices. In each `mff` directory, a set of `xml` files contain metadata about the spatial coordinates of EEG electrodes, epoch segmentation, events, workspace settings, and so on. Metadata can be accessed at any stage of data preprocessing for raw, epochs and evoked data using the `.info` attribute of the `mne` module. The actual EEG recording data is contained in `signal1.bin`.

Typically, continuous raw data are read to create `Raw` objects. Data can subsequently be divided into `Epochs`, which are discontinuous data. Finally, data are averaged across all epochs into `Evoked` objects. Data can be easily visualised using the `.plot()` method. Data can also be accessed in the form of `numpy.ndarrays` by using the `.data`  attribute or [`get_data()`](https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.get_data) method. We can also typically access the channel names by accessing the property the [`ch_names`](https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.ch_names), which returns a list of channel names. 

>[!Info] Accessing data as a numpy array
>`Evoked` objects are created by averaging over epochs, so they are typically smaller in size compare to the raw recording or epoched data. Therefore, we can access `Evoked` data using the `data` attribute. On the other hand, the `get_data()` method gets all epochs as a 3D array. 

#### [`Epochs` objects](https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs)

`Epochs` objects typically store segmented EEG data. Data can be accessed using the `get_data()` method, which returns a 3D `numpy` array that reflect `(number of epochs, number of channels, number of samples)`.

#### `Evoked` objects

`Evoked` objects typically store EEG signals that have been averaged over epochs in order to estimate stimulus-*evoked* neural activity. The data in an `Evoked` object can be accessed using the `data` attribute, and are stored in a 2D `numpy` array in the shape to reflect `(number of channels, number of samples)`. 

#### Key documentation

- [Creating MNE-Python data structures from scratch](https://mne.tools/stable/auto_tutorials/simulation/10_array_objs.html#tut-creating-data-structures)
- [The Evoked data structure: evoked/averaged data](https://mne.tools/stable/auto_tutorials/evoked/10_evoked_overview.html)
- [Overview of MEG/EEG analysis with MNE-Python](https://mne.tools/stable/auto_tutorials/intro/10_overview.html#sphx-glr-auto-tutorials-intro-10-overview-py)
- [Querying the Raw object](https://mne.tools/stable/auto_tutorials/raw/10_raw_overview.html)

### Selecting ERP components
*The components **N1, N2, P3, LPP** are selected (Galang et al., 2021; Coll, 2018).*
#### Defining time windows
- For N1, 70 - 150 ms
- For N2, 200 - 300 ms
- For P3, 350 ms - 450 ms
- For LPP, 500 ms - 800 ms
#### Defining ROIs
>[!Info]
>ROIs are regions of interest. In EEG/MEG analysis, ROIs can refer to groups of electrodes or sensor locations on the scalp. These groups of electrodes are selected because they are believed to be particularly relevant to the cognitive processes under investigation.

*The electrode scheme is used by Galang et al., 2021; Galang et al., 2020; Fan & Han, 2008.*
- N2 is analysed using average waveforms from the **Frontal-Central** electrodes.
- P3 is analysed using the average waveforms from the **Central-Parietal** electrodes.
- LPP is analysed using the average waveforms from the **Central-Parietal** electrodes.
### Getting started on preprocessing
#### Loading libraries
```python
import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
import nolds

mne.set_log_level('error')
random_state = 42
```
#### Reading EGI data
```python
p_id = str(input('Enter subject ID: '))

raw_file = 'data/eeg_raw/' + p_id + '.mff'
raw = mne.io.read_raw_egi(raw_file, preload = True)
```
#### Setting channel types
```python
raw.set_channel_types({'VREF':'misc'})
```
#### Setting montage
*We are using a 128-channel net*
```python
montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
raw.set_montage(montage)
```
####  Finding events
*Note that the event code for stimulus is 1.*
```python
events = mne.find_events(raw)
```
The `events` array contains 3 values per row. The first value is the time of the event. The second value represents the state change. The third value is the event or trigger code.
#### Plotting events
*Check that events are found and plotted correctly. `sfreq` stands for sampling frequency. In this case, our sampling frequency is 500, so the output to `raw.info['sfreq']` would be 500.0.*
```python
mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp, event_id = {'Stimulus':stim_code}, show = True)
```
#### Setting annotations
*We may want to annotate the event code to clarify what the event code stands for. In this case, event code 1 is stimulus.*
```python
raw.set_annotations(mne.annotations_from_events(events, event_desc={stim_code:'Stimulus'}, sfreq=raw.info['sfreq']))

print(raw.annotations)
```
### Filtering the data
*This step addresses frequency content by removing unwanted noise and artifacts, specifically low-frequency drifts, high-frequency noise, and line noise.*
- Apply band pass filter at high-pass = 0.1 Hz, low-pass = 40 Hz (Coll, 2018; Galang et al., 2021).
- Given that the low pass filter is 40 Hz, we don't need to apply a notch filter at 50 Hz. However, we put the code here anyways for illustration, but the code for applying notch does not change the dataframe. Apply notch filter to filter out electrical noise = 50 Hz. Note that the `notch_filter` method modifies the dataframe `raw_filt`.
```python
highpass = 0.1
lowpass = 40
notch = 50

raw_filt = raw.copy().filter(highpass, lowpass)
raw_filt.notch_filter(notch)
```

It is a good idea to compare the before and after of EEG data, so here we plot the pre-filtered raw data and the filtered data to see if noise level has reduced. 
```python
fig_raw = raw.plot(start=5,duration=20,n_channels=20,show=True,scalings=rej_thresh)

fig_raw.suptitle('Raw data for {}'.format(p_id), y = 0.98)

plt.tight_layout()

  

fig_filt = raw_filt.plot(start=5,duration=20,n_channels=20,show=True,scalings=rej_thresh)

fig_filt.suptitle('Filtered data for {}'.format(p_id), y = 0.98)

plt.tight_layout()
```

This should produce plots that look like these: 
*Raw data*
![[002_raw.png]]

*Filtered data*
![[002_filt.png]]
### Excluding bad channels
#### Detecting bad channels manually
*To detect bad channels manually, we use `raw_filt.plot()` to inspect the data. We perform this step iteratively until the data look clean enough for the next steps.*
```python
raw_filt.plot(start = 5, duration = 20, n_channels = 20, show = True, scalings = dict(eeg = 75e-6))
plt.show();

print(f'Subject {}'.format(p_id))
```
- Use `Home` and `End` to adjust timeframe. 
- Use `PgUp` and `PgDn` to adjust the number of channels plotted.
- Use `+` and `-` to adjust power.
- Click on the channel name to remove it. 
#### Detecting bad channels automatically
*Following Komosar et al., we apply an Iterative Standard Deviation (ISD) method for automated bad channel detection (2022). Their comparison of ISD and visual inspection methods in bad channel detection found no significant differences and has an accuracy rate of 99.69% for both gel-based and dry EEG for resting state, and 99.38% for EEG with head movements for both setups. This means the ISD works great for both resting and task EEG recordings.*

First, calculate the standard deviation (SD) of the signal for each channel
$j$ (see eq 1).
*Equation 1:*
$$
SD_{j} = \sqrt{\frac{1}{N-1}\sum\limits_{i=1}^{n}|V_{(i, j)}-\bar{V}_{j}|^2}
$$

Implement the above function in Python below.
```python
eeg_data = raw_filt.get_data()
std = eeg_data.std(axis = 1)
```
The `std` variable represents the standard deviation of the EEG signal amplitudes for each channel. This gives us an idea of how much the amplitude values vary around the mean for each channel. A higher standard deviation indicates greater variability in the EEG signal amplitudes for that particular channel. 

We also create an empty list to store bad channels later.
```python
bad_channels = []
```

Equations 2-5 represent 4 criteria used to eliminate outliers from the population of SDs. In eq 2, we use the median ($M$) of the population of standard deviations in the $k$-th iteration and calculate the absolute difference between the individual SD and M. In eq 3, $SD_{j,k}$ is the standard deviation of the $j$-th channel at the $k$-th iteration. In eq 5, $SD_{p}$ is the standard deviation of all individual channel standard deviations in the $k$-th iteration.
*Equation 2:* 
$$
|SD_{(j, k)}-M_{k}| > 75th\:percentile
$$
We implement eq 2 in Python below.
```python
std_median = np.median(std)
perc_75 = np.percentile(abs_diff, 75)

for j in range(eeg_data.shape[0]):
	if np.abs(std[j]-std_median) > perc_75:
		bad_channels.append(j)
```
- We use the `np.abs()` function to compute the absolute value element-wise for each value in the array. The resulting `abs_diff` variable contains the absolute differences for each channel.
- We use the `np.percentile()` function to get the value below the specified percentage of the data, and in this case, 75%.
- We calculate the absolute difference between the standard deviation of each channel and the population median of standard deviations, then compare that to the $75th$ percentile. If a channel has a standard deviation that exceeds the population median of standard deviations, we add it to the list of `bad_channels`.

*Equation 3*:
$$
SD_{(j,k)} < 10^{-4}\mu V
$$

*Equation 4:*
$$
SD_{(j,k)}>100\mu V
$$

We implement eq 3 and 4 together in Python, since we are looking at the lower and upper limits of standard deviation thresholds.
```python
std_thresh_lower = 1e-4 
std_thresh_upper = 100

for j in range(eeg_data.shape[0]):
    if std[j] < std_thresh_lower or std[j] > std_thresh_upper:
        bad_channels.append(j)
```

*Equation 5:*
$$
SD_{p(k)}>5
$$
We implement eq 5 in Python.
```python
std_p = std.std()
std_thresh_fixed = 5

for j in range(eeg_data.shape[0]):
    if std[j] > std_thresh_fixed * std_p:
        bad_channels.append(j)
```

Since we went through eq 2-5 in separate for loops, there might be repetitive elements generated in the final list of `bad_channels`. Therefore, we reassign the variable to unique elements only.
```python
len(list(set(bad_channels)))
```

We can also combine the above steps in one for loop and define all variables at the top.
```python
eeg_data = raw_filt.get_data()
std = eeg_data.std(axis=1)
std_median = np.median(std)
perc_75 = np.percentile(abs_diff, 75)
std_thresh_lower = 1e-4 
std_thresh_upper = 100
std_p = std.std()
std_thresh_fixed = 5

bad_channels = []
for j in range(eeg_data.shape[0]):
    if np.abs(std[j] - std_median) > perc_75:
        bad_channels.append(j)
    elif std[j] < std_thresh_lower or std[j] > std_thresh_upper:
        bad_channels.append(j)
    elif std[j] > std_thresh_fixed * std_p:
        bad_channels.append(j)
```

>[!Info]
>A basic way of using standard deviation to reject bad channels is using a fixed rejection threshold. This method lacks the flexibility to identify bad channels on a subject-by-subject basis. Some subjects may have especially noisy data, and using a rigid "one-size-fits-all" rejection threshold for bad channel detection would mean excluding too much data for some subjects while preserving too much noise for others. Here is an example where we set 3 standard deviations as the fixed rejection threshold. This means all channels that are outside of 3 standard deviations from the average amplitude are appended to a new list and can be dropped later. This method is handy if combined with visual inspection, although you need to iteratively perform visual inspection and manual annotation of bad channels, as this method usually only excludes really bad channels.
>```python
>avg_amp = np.mean(eeg_data, axis = 1)
>amp_thresh = 3 * avg_amp_grand
>
>bad_channels = []
>for i in range(len(avg_amp)):
>	if np.abs(avg_amp[i]) > amp_thresh:
>		bad_channels.append(raw_filt.ch_names[i])
>
>print('Bad channels:{}'.format(bad_channels))
>```
###### Dropping channels
*Finally, to drop channels, we use the `drop_channels()` method.* 

First, retrieve the channel labels based on the list of indices stored in `bad_channels` using the `ch_names` attribute. We accomplish this using list comprehension. Then, drop channels using the labels we retrieved.
```python
bad_channels_names = [raw_filt.ch_names[i] for i in bad_channels]
raw_no_bads = raw_filt.copy().drop_channels(bad_channels_names)
```

After dropping bad channels, we inspect the data in a plot. 
```python
raw_no_bads.plot(start=5,duration=20,n_channels=20,show=True,scalings=rej_thresh_epochs)
```
The data look much cleaner here.
![[037_no_bads.png]]

We also inspect the power spectral density (PSD) plot, which shows the distribution of power contained within a signal across different frequencies. We use the `plot_psd()` method and specify the maximum frequency to display in the PSD plot using the low pass filter, so that we can zoom in on the frequency range of interest. 
```python
raw_no_bads.plot_psd(fmax=lowpass, show=True)
```

The resulting PSD plot is relatively smooth and continuous in shape and does not contain rogue, distorted lines, which is what it should look like. There is a small spike around 10 Hz, which could be an artifact such as small electrical interference, neural activity in the alpha oscillation band (8 - 13 Hz) that is commonly observed when a participant's eyes are closed, or something else. We will clean this up in the following step.
![[037_psd.png]]

>[!Info]
>In a PSD plot, the x-axis represents frequency, often in logarithmic scale, while the y-axis represents the power or magnitude of the signal at each frequency. The power at each frequency is calculated using methods like Fourier Transform or Wavelet Transform. The PSD plot gives insight into the frequency components present in a signal and how much power is associated with each frequency.

### Removing artifacts
#### Extracting ICA components
*Using Independent Component Analysis (ICA), we fit the ICA model, then select ICA components to exclude. Afterwards, we use the `.apply` method to exclude these components.*
```python
n_components = 15
method = 'infomax'

ica = ICA(n_components = n_components, random_state = random_state, method = method)
ica.fit(raw_no_bads.copy())
```
- `raw.info['bads']` sets the `bads` attribute of the `info` dictionary. This attribute is used to specify which channels are considered bad and should be excluded from analysis. 
- `mne.preprocessing.ICA()` creates an instance of the Independent Component Analysis (ICA) object and is used to separate the mixed signals in EEG data into statistically independent components. 
	- For `method`, we can also use `fastica` or `picard` if `infomax` does not work well.
- We use the `.fit()` method to fit the ICA to the EEG data and specify the rejection threshold. 

After fitting ICA, plot the components. We can inspect the topomap for bi-hemispheric frontal activity or eye blinks. Brain-related ICs typically have focused, localised scalp distributions. Artifactual ICs often have broader, diffuse distributions. 
```python
ica.plot_components(inst=raw_filt, show=True, title = 'ICA Components for {}'.format(p_id))
```

>[!Tip]
>Click on a component name to exclude it.
>

We also examine the time course of each IC by plotting their time courses. Brain-related ICs are smooth and continuous, while artifactual ICs exhibit sudden, sharp changes.
```python
ica.plot_sources(inst = raw_no_bads)
```

>[!Info]
>Independent Component Analysis (ICA) is a signal processing technique that separates a multivariate signal into additive, statistically independent components. It is commonly used to preprocess EEG and other biophysical signals to uncover sources of underlying neural activity in the data. 
>
>ICA assumes that the recorded signal is a linear combination of independent source signals. Unlike Principal Component Analysis (PCA), it does not assume orthogonality, and instead assumes that the sources are statistically independent. However,  ICA assumes the number of sources does not exceed the number of channels, which may not always hold true. 
>
>In EEG, ICA is used to separate the recorded scalp data into underlying sources, often referred to as "components." These components can correspond to different neural or non-neural sources, such as brain activity, eye blinks, muscle artifacts, and more. 
>
>Once the ICA decomposition is performed, we can identify components associated with specific artifacts or cognitive processes based on their spatial patterns and temporal behaviors. Therefore, ICA is particularly valuable for removing artifacts, such as eye blinks and muscle activity. Moreover, ICA can help identify and reject bad channels that are contaminated with artifacts. Channels with high-amplitude noise, poor signal quality, or other issues can be identified based on the behavior of the corresponding ICA components.
>

#### Selecting ICA components automatically
*We use the Fully Automated Statistical Thresholding for EEG Artifact Rejection (FASTER) method (Nolan et al., 2010). Nolan et al. also used the infomax method for ICA. Five criteria are implemented to evaluate components to exclude where z-score > 3:
1. Correlation with EOG channels
2. Spatial kurtosis
3. Slope in filter band
4. Hurst exponent
5. Median gradient

The **slope** of the spectrum over the low-pass filter band is calculated as follows to represent mean slope of the power spectrum of the component $x$ time-course, where $f_{LP1}$ is the high-pass band and $f_{LP2}$ is the low pass band.
$$
\frac{dF(x_{ct})}{df}|f_{LP1}<f<f_{LP2}
$$

The **Hurst exponent** is a measure of long-range dependence within a signal. Human EEG has $H$ of around $0.7$. The Hurst exponent of component $c$ time-course is $H_{x_{c_{x}}}$. 

The **median gradient** is the slope of the component $c$ time-course:
$$
median\frac{d(x_{c_{t}})}{dt}
$$

First, calculate ICA component scores from IC sources.
```python
ica_sources = ica.get_sources(raw_no_bads)
ica_scores = ica_sources.get_data().std(axis=1)
```
- We use `get_sources()` to extract the IC sources from the ICA decomposition for `raw_no_bads`. These sources are the underlying signals that ICA aims to separate. 
- We then use `get_data()` to retrieve the data from these sources, which generates a `NumPy` array where each row represents a different IC, and each column represents a time point in the signal. We use `std` to calculate the standard deviation along each row or across time points to compute a measure of how much each IC signal varies over time. 
- The resulting `ica_scores` array contains the standard deviations of the IC sources, which is a measure of the variability of each component's signal over time. We use this information in artifact detection.

For criteria 1, we first select two vertical electrooculography (VEOG) and four horizontal electrooculography (VEOG) channels to monitor eye movements, since not every EOG channel contains signal.
```python
eog_channels = ['E127','E126','E17','E15','E21','E14']
```

We then calculate the correlation coefficient and z-score using `scipy.stats` of each IC time series with the 4 EOG channels.
```python
eog_scores = raw_no_bads[eog_channels][0]
cor_eog = np.corrcoef(ica_scores, eog_scores)
zscore_eog = zscore(cor_eog, axis=0)
```

For criteria 2, calculate the spatial kurtosis of ICA components and their z-scores.
```python
ica_maps = ica.get_components()
spatial_kurtosis = np.sum(ica_maps ** 4, axis=1) / np.sum(ica_maps ** 2, axis=1) ** 2
zscore_kurtosis = zscore(spatial_kurtosis)
```

For criteria 3, calculate the slope of the spectrum over the low-pass filter band. We use the `mne.time_frequency.psd_array_welch` method to compute PSD using Welch's method.
```python
ica_psd, freqs = mne.time_frequency.psd_array_welch(ica_scores, fmin=highpass, fmax=lowpass, sfreq=sfreq)

slope = np.zeros(ica_psd.shape[0])
for i in range(ica_psd.shape[0]):
    slope[i] = np.mean(np.gradient(np.log(ica_psd[i])))
zscore_slope = zscore(slope)
```

For criteria 4, calculate the Hurst exponent for each ICA component's time-course. We use the` hurst_rs` function from the `nolds` library.
```python
hurst_values = np.zeros(n_components)
for i in range(n_components):
    hurst_values[i] = nolds.hurst_rs(ica_scores[i])
zscore_hurst = zscore(hurst_values)
```

For the final criteria 5, calculate the median gradient by computing the derivative of the time-course, then calculating the median of those derivative values. 
```python
median_gradients = np.zeros(n_components)
for i in range(n_components):
    derivative = np.gradient(ica_scores[i])
    median_gradients[i] = np.median(derivative)
zscore_gradient = zscore(median_gradients)
```

Finally, we combine all the criteria and create a list of bad ICA components. 
```python
ica_bads = []

for i in range(len(ica_sources.info['ch_names'])):
    if abs(zscore_eog[i, 0]) > rej_thresh_ica:
        ica_name = ica_sources.info['ch_names'][i]
        ica_bads.append(ica_name)
    elif abs(zscore_kurtosis[i]) > rej_thresh_ica:
        ica_name = ica_sources.info['ch_names'][i]
        ica_bads.append(ica_name)
    elif abs(zscore_slope[i]) > rej_thresh_ica:
        ica_name = ica_sources.info['ch_names'][i]
        ica_bads.append(ica_name)
    elif abs(zscore_hurst[i]) > rej_thresh_ica:
        ica_name = ica_sources.info['ch_names'][i]
        ica_bads.append(ica_name)
    elif abs(zscore_gradient[i]) > rej_thresh_ica:
        ica_name = ica_sources.info['ch_names'][i]
        ica_bads.append(ica_name)

print('There are {} bad ICA components'.format(len(ica_bads)))
```

>[!Info] **What is a bad ICA Component?**
>A bad independent component analysis (ICA) component refers to a component that does not represent meaningful or relevant information in the data. In ICA, the goal is to separate a multivariate signal into statistically independent components. However, sometimes certain components may arise that are noise, artifacts, or do not contribute to the underlying structure of the data.
>
> Bad ICA components can be identified by their characteristics, such as having a high amplitude in specific frequency ranges (indicating noise), showing repetitive patterns (suggesting artifacts), or lacking any discernible pattern altogether. These components are typically considered undesirable as they can distort the interpretation and analysis of the data.

We then apply ICA. 
```python
ica_bads_idx = [ica_sources.info['ch_names'].index(i) for i in ica_bads]
ica.exclude.extend(ica_bads_idx)
raw_postica = ica.apply(raw_no_bads.copy(), exclude = ica.exclude)
```
explain the code snippet above line by line here:
- First, create a list `ica_bads_idx` which contains the indices of the bad channels in `ica_sources.info['ch_names']`. 
- Then, extend the `exclude` attribute of the ICA object (`ica.exclude`) with the indices of the bad channels.
- Finally, apply ICA to a copy of the raw data (`raw_no_bads.copy()`) using the updated exclude list, and assigns the result to `raw_postica`.

>[!Info] **What does it mean to exclude an ICA component?**
>To exclude an Independent Component Analysis (ICA) component means to remove or disregard that specific component from the analysis or calculation. ICA is a statistical technique used to separate a multivariate signal into independent non-Gaussian components. Each component represents a source signal contributing to the observed data. 
>
>Excluding an ICA component can be done for various reasons, such as if the component is considered noise, artifact, or irrelevant to the analysis. By excluding a specific component, it is not taken into account when interpreting or analyzing the data. This can help focus on the relevant components and improve the accuracy of subsequent analyses or interpretations.

After performing ICA, generate topographic maps to examine the data. Check that there is no more bi-hemispheric frontal activity, and manually exclude bad channels if the improvements are not good enough.
```python
ica.plot_components(inst=raw_postica, show=True, title = 'ICA Components for {}'.format(p_id))
ica.plot_sources(raw_postica) # raw data after performing ICA
```
### Segmenting data into epochs
Epochs are time-locked to stimulus onset (event code = 1) and the end of a response period marked by a participant key press.

We calculate the response period here.
```python
duration_fix = 0.5
duration_blank = 0.15

stimulus_times = events[events[:,2] == stim_code][:, 0]
response_times = (np.diff(stimulus_times) - (duration_fix + duration_blank) * sfreq) / sfreq # in seconds
avg_response_time = np.mean(response_times)
print('Average response time is {} seconds'.format(avg_response_time))
```
- We use the `np.diff()` function to calculate the difference between consecutive stimulus trial times. This gives you the time intervals between consecutive stimuli. 
- The sum of `duration_fix` and `duration_blank`, representing fixation cross duration and blank screen duration in seconds respectively, is mulitplied by the sampling frequency `sfreq` to convert them into the same unit as `stimulus_times`. 
- We divide the output of `np.diff(stimulus_times) - (duration_fix + duration_blank)` by the sampling frequency `sfreq` to convert the result into seconds. 
- We calculate the average response time to inspect that the output looks reasonable. 

>[!Info]
Epoch segmentation refers to the process of dividing a continuous data stream into smaller segments or epochs. Each epoch typically represents a fixed duration or number of data points. The choice of epoch duration should be long enough to capture meaningful information but short enough to avoid losing important temporal details. For example, in EEG (electroencephalography) analysis, epochs are often defined as short time windows (e.g., 1-2 seconds) to capture specific brain activity patterns.

First, we define the time windows. The pre-stimulus period includes fixation cross and blank screen, which is 0.65 seconds. The stimulus duration depends on participant's key press response, so we implement a variable epoch length to reflect how long it took each participant to respond to each trial, rather than a fixed epoch duration for all trials. This is done to capture the most meaningful time window down to the miliseconds. 
```python
tmin = -0.65

```
#### Baseline correction window

>[!Info]
>Baseline correction is a normalisation method. This removes any differences in the overall amplitudes of the conditions (Luck, 2005). The reason for normalisation after ICA is to ensure that the reference calculation is based on the cleanest possible data. Bad channels and artifacts can introduce significant noise, and referencing data with noise can potentially amplify that noise in the entire dataset.
>
>Baseline correction is particularly relevant when studying ERP components like N2, P3, LPP. It's common to use a pre-stimulus baseline period to remove baseline shifts and isolate the evoked activity.
>
>It is important to consider that baseline correction assumes that the baseline period is free of relevant activity. It's important to choose a suitable baseline window and ensure that it doesn't include any ongoing neural activity.

First, define the time window and create epochs. The start of each epochs in seconds is adjusted for a 500 ms fixation cross and a 150 ms blank interval. The end of each epoch in seconds is adjusted for 64 stimulus trials at 1500 ms each, and response time.
```python
tmin = -0.65
tmax = 2.0

epochs = mne.Epochs(raw, events, event_id = 1, tmin = tmin, tmax = tmax, baseline = None)
```

Inspect the epochs to ensure that they are correctly created. First, plot individual epochs to visually inspect the data for any artifacts. 
```python
epochs.plot(n_channels = 10)
```

Then plot average ERPs across all epochs. 
```python
epochs.average().plot(picks = raw.info['ch_names'][:-n_ref_chans], spatial_colors = True)
plt.tight_layout();
```
### Rejecting epochs
*First, set a rejection threshold. According to Coll , this was the most used method across 36 studies (2018). This step addresses extreme amplitude values. After filtering, there are data segments that might still contain unusually high amplitudes due to artifacts or other non-neural sources. These segments can skew the analysis and interpretation of EEG data. Setting a rejection threshold allows you to identify and exclude these segments from further analysis, ensuring that your results are not influenced by extreme amplitude values.*

>[!Info]
> A low-frequency drift refers to a slow and gradual change in the baseline signal amplitude over time, especially in the low frequency range. This drift is unrelated to neural activity and often caused by non-neural sources, such as equipment and electrode impedance changes. A high-pass filter is typically applied to filter out drifts.
 
**Example** of setting a threshold:
```python
rej_thresh = dict(eeg = 100e-6)
```

The value `100e-6` is in the unit of microvolts ($\mu$V) and means **times 10 to the power of -6**, which corresponds to microunits. In this case, any EEG channel data that exceeds 100 $\mu$V will be marked as a bad segment and rejected. Setting a higher rejection threshold allows more variability in the data, while lower values will be more stringent in rejecting data segments. In Coll's meta-analysis (2018), the median rejection threshold across 36 studies was around 75 $\mu$V. 

Therefore, we set the rejection threshold here accordingly. Setting the threshold **for current study**:
```python
rej_thresh_epochs = dict(eeg = 75e-6)
```

```python
epochs = Epochs(raw, events, event_id = 1, tmin = tmin, tmax = tmax, baseline = baseline, rejection = rej_thresh)
```
### Applying baseline correction
*We perform baseline correction on the remaining epochs. Subtract the mean amplitude of a pre-stimulus baseline period from each epoch to ensure a consistent baseline across epochs. Baseline correction involves subtracting a reference baseline period's mean amplitude from the data in each epoch. In this case, the baseline correction uses a baseline window from -200 ms to 0 ms, which is before the event onset. The mean amplitude of this baseline window is subtracted from the entire epoch's data, effectively removing any pre-existing voltage shifts or drifts that might have occurred before the event. This is also called the pre-stimulus baseline.*
```python
baseline = (200, 0)
epochs.apply_baseline(baseline)
```
### Re-referencing the data
*We set the reference electrode as the average, then subtract the average of all electrodes from each electrode.This is considered a normalisation step.* 
```python
epochs = epochs.set_eeg_reference(ref_channels = ['VREF'])
epochs.average().plot_topomap()
```
### Selecting channels 
*We select 64 scalp electrodes (Coll, 2018). Having 128 electrodes means we have good spatial resolution, although 60-64 electrodes sufficiently cover the key regions of the brain. Having a higher number of electrodes can also lead to more noise due to increased inter-electrode impedance variation, and there is a higher likelihood of capturing various artifacts such as muscle activity, eye blinks, and movement-related artifacts. Selecting electrodes after filtering the data allows us to work with cleaner data, and ensures that the same set of electrodes is used across conditions and participants.*

- **N1** (Frontal): E22, E18, E16, E10, E3, E19, E11, E4, E20, E12, E5, E26, E15, E9, E2, E23, E124, E118
- **N2** (Frontal-Central): E26, E22, E15, E9, E2, E23, E18, E16, E10, E3, E24, E19, E11, E4, E124, E20, E12, E5, E118, E13, E6, E112, E7, E106
- **P3** (Central-Parietal): E58, E65, E70, E75, E83, E90, E96, E51, E59, E66, E71, E76, E84, E91, E97, E47, E52, E60, E67, E72, E77, E85, E92, E98, E42, E53, E61, E62, E78, E86, E93, E37, E54, COM, E79, E87, E31, E55, E80, VREF
- **LPP** (Central-Parietal): same electrodes as **P3**. 
### Interpolating epochs 
*Interpolating epochs involves filling in missing data points in epochs, usually due to artifact rejection or bad channels.*
```python
epochs_intpl = epochs.interpolate_bads(reset_bads = True)
```
The `reset_bads` parameter, when set to True, will reset the list of bad channels for the interpolated epochs. This is helpful to ensure that the bad channels marked for rejection before interpolation are not carried over to the interpolated epochs.
### Averaging epochs

Subject ERPs are created by averaging epochs in each condition (powerful, powerless). 
```python
evoked_powerful = epochs.average(events = event_code_condition_1)
evoked_powerless = epochs.average(events = event_code_condition_2)
```

Subject ERPs are then averaged together to create the Grand Average ERPs. 
```python
evoked = epochs.average()
evoked.plot()
evoked.plot_topomap()
```
An `evoked` object in MNE is created from averaged epochs. An evoked object represents the average of the EEG data across multiple epochs, typically time-locked to a specific event. It provides a compact representation of the grand average response to the event of interest, making it easier to analyze and visualize the ERP (event-related potential) components. 
- `evoked.plot()` plots the average waveform for all channels. 
- `evoked.plot_topomap()` generates a topographic map of the ERP response at specific time points. 
### Extracting amplitude and latency values
*We obtain mean amplitudes within the predetermined time window for each ERP component.*

First, define time windows.
```python
tw_n2 = (0.2, 0.3)
tw_p3 = (0.35, 0.45)
tw_lpp = (0.5, 0.8)
```

Second, extract and calculate mean amplitudes,
```python
n2_epochs = epochs.copy().crop(tw_n2[0], tw_n2[1])
n2_mean_amp = epochs_n2.get_data().mean(axis = 2)

print(f'Mean Amplitude N2: {}'.format(n2_mean_amp))
```

Now, get the mean latency values. Select a representative channel for each component:
- N2, Fz electrode
- P3, Cz electrode
- LPP, Pz electrode
```python
n2_channel = 'Fz'
n2_peak, n2_latency = n2_epochs[n2_channel].get_peak(mode = 'abs')

print(f'Mean Latency for N2: {}'.format(n2_latency))
```
Select `neg` for mode when we are searching for a negative peak, or `pos` for a positive peak, or `abs` for a peak with the maximum absolute value within the range.