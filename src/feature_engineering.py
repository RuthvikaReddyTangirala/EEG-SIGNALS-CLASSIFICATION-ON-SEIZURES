import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch

# Feature Extraction

#### Extract relevant features from the EEG signals. You may consider time-domain and frequency-domain features.

### Extracting Time Domain Features

df = pd.read_csv("data/data_preprocessed.csv")

def extract_features_time_domain(signal):
    # Time-domain features
    mean_value = np.mean(signal)
    variance = np.var(signal)
    rms_value = np.sqrt(np.mean(signal**2))
    std_dev = np.std(signal)

    return [mean_value, variance, rms_value, std_dev]

# Apply the function to each row in the DataFrame
feature_columns = ['mean', 'variance', 'rms', 'std_dev']

# Create a new DataFrame for features
feature_df = pd.DataFrame(columns=['file_name', 'file_code', 'file_set'] + feature_columns)

# Iterate through rows in the original DataFrame
for index, row in df.iterrows():
    file_name = row['Filename']
    file_code = row['file_code']
    file_set = row['file_set']
    eeg_signal = row.iloc[4:]  # Assuming the EEG signal starts from the second column

    # Extract features from the EEG signal
    features = [file_name]+ [file_code] + [file_set] + extract_features_time_domain(eeg_signal)

    # Append the features to the new DataFrame
    feature_df = feature_df.append(pd.Series(features, index=feature_df.columns), ignore_index=True)

# Display the new DataFrame
print("Displaying the feature dataframe",feature_df)

"""### Extracting Frequency Domain Features"""

def calculate_psd(row, fs):
    """ Calculate the Power Spectral Density (PSD) for a row. """
    freqs, psd = welch(row, fs)
    return freqs, psd

def calculate_peak_frequency(row, fs):
    """ Find the peak frequency. """
    freqs, psd = calculate_psd(row, fs)
    peak_freq = freqs[np.argmax(np.abs(psd))]
    return peak_freq

def bandpower(row, fs, band):
    """ Calculate the Band Power within a specific frequency band for a row. """
    freqs, psd = welch(row, fs)
    psd_abs = np.abs(psd)  # Use the absolute value of the PSD
    band_freqs = np.logical_and(freqs >= band[0], freqs <= band[1])
    band_power = np.trapz(psd_abs[band_freqs], freqs[band_freqs])
    return band_power


fs = 173.61  # Sampling rate (in Hz)

# Initialize lists to store the features
peak_frequencies = []
delta_powers = []
theta_powers = []
alpha_powers = []
beta_powers = []
gamma_powers = []

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Calculate each feature
    peak_freq = calculate_peak_frequency(row[4:], fs)
    delta = bandpower(row[4:], fs, [0.5, 4])
    theta = bandpower(row[4:], fs, [4, 8])
    alpha = bandpower(row[4:], fs, [8, 13])
    beta = bandpower(row[4:], fs, [13, 30])
    gamma = bandpower(row[4:], fs, [30, 45])

    # Append the features to the lists
    peak_frequencies.append(peak_freq)
    delta_powers.append(delta)
    theta_powers.append(theta)
    alpha_powers.append(alpha)
    beta_powers.append(beta)
    gamma_powers.append(gamma)

# Add the features as new columns to the DataFrame
feature_df['Peak_Frequency'] = peak_frequencies
feature_df['Delta_Power'] = delta_powers
feature_df['Theta_Power'] = theta_powers
feature_df['Alpha_Power'] = alpha_powers
feature_df['Beta_Power'] = beta_powers
feature_df['Gamma_Power'] = gamma_powers

# Now df has the original data along with the new frequency domain features
print("Displaying the features dataframe",feature_df)

plt.figure(figsize=(30,20))
plt.subplot(3,1,1)
sns.scatterplot(x='variance', y = 'mean', data = feature_df, hue = 'file_code', palette=['red', 'blue', 'yellow', 'green',  'orange'])
plt.title("Variance vs mean")

plt.subplot(3,1,2)
sns.scatterplot(x='std_dev', y = 'mean', data = feature_df, hue = 'file_code', palette=['red', 'blue','yellow', 'green', 'orange'])
plt.title("Standard deviation vs mean")

plt.subplot(3,1,3)
sns.scatterplot(x='rms', y = 'mean', data = feature_df, hue = 'file_code', palette=['red', 'blue', 'yellow', 'green', 'orange'])
plt.title("RMS vs mean")

plt.legend()
plt.show()

"""From the Above graphs based on time based features it can be observed that values of Set S differ from all others significantly, Set D values has occasionally differ from other.

From the given paper:  
Volunteers
were relaxed in an awake state with eyes open ~A! and eyes
closed ~B!, respectively. Sets C, D, and E originated from our
EEG archive of presurgical diagnosis. For the present study
EEGs from five patients were selected, all of whom had
achieved complete seizure control after resection of one of
the hippocampal formations, which was therefore correctly
diagnosed to be the epileptogenic zone ~cf. Segments
in set D were recorded from within the epileptogenic zone,
and those in set C from the hippocampal formation of the
opposite hemisphere of the brain. While sets C and D contained only activity measured during seizure free intervals,
set E only contained seizure activity. Here segments were
selected from all recording sites exhibiting ictal activity.

Hence it is given that Set E has been collected during activity and it can also be observed in the above charts. Labelling the rows from Set E as 1 indicating it is a seizure.
"""

# Function to check if 'S' is present in the value
def labelling(value):
    if 'E' in value:
        return 1
    else:
        return 0


# Create a new column based on the condition
df['label'] = df['file_set'].apply(labelling)
feature_df['label'] = feature_df['file_set'].apply(labelling)
print("Displaying the dataframe\n", df)

feature_df.to_csv(r"data\feature_df.csv")