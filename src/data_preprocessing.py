
#importing all the necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import xgboost as xgb
from scipy.signal import welch
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch

warnings.filterwarnings('ignore')

"""# Data Preprocessing

#### Download and extract the datasets.
"""

# Specify the folder path where your text files are stored
folder_path = 'data\data'

# Initialize an empty list to store data
data = []

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".txt"):
        # Read the data from the file
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            # Read the content of the file and split it into samples
            content = file.read().split()
            # Convert ASCII data to integers (assuming each sample is a single integer)
            samples = [int(value) for value in content]
            # Append the data to the list with the filename as the first element
            data.append([filename] + samples)

# Create a DataFrame from the list of data
column_names = ['Filename'] + [f'Sample_{i}' for i in range(1, 4098)]  # Assuming 4096 samples
df = pd.DataFrame(data, columns=column_names)

# Display the DataFrame
print("Displaying the dataframe:\n",df)

"""The above dataset contains 500 rows and 4100 columns after extracting and encoding.

#### Explore the data to understand its structure and characteristics.
"""

#displaying information about the dataset
print("Displaying the information of the dataset\n", df.info())

#displaying statistics of the dataset
print("Displaying the statistics of the dataset\n", df.describe())

"""### If necessary, preprocess the EEG data, including handling missing values, noise reduction, and data augmentation.

#### Encoding the labels based on the file names
"""

def encode_filename(value):
    if 'F' in value:
        return 1
    elif 'N' in value:
        return 2
    elif 'O' in value:
        return 3
    elif 'S' in value:
        return 4
    elif 'Z' in value:
        return 5

# Create a file_code based on the Filename
df['file_code'] = df['Filename'].apply(encode_filename)

"""## Encoding the file set based on the file names

- File set has been taken from the official website
"""

def encode_filename_get_set(value):
    if 'F' in value:
        return 'D'
    elif 'N' in value:
        return 'C'
    elif 'O' in value:
        return 'B'
    elif 'S' in value:
        return 'E'
    elif 'Z' in value:
        return 'A'

# Create a file_set based on the Filename
df['file_set'] = df['Filename'].apply(encode_filename_get_set)

"""#### Encoding the data colletion and recording techniques given in the official paper"""

def encode_filename_get_recording_technique(value):
    if 'F' in value:
        return 'Epileptogenic Zone : Seizure Free'
    elif 'N' in value:
        return 'Hippocampal Formation : Seizure Free'
    elif 'O' in value:
        return 'Eyes Closed : Seizure Free'
    elif 'S' in value:
        return 'Seizure Activity'
    elif 'Z' in value:
        return 'Eyes Open : Seizure Free'

# Create a file_set based on the Filename
df['recording_technique'] = df['Filename'].apply(encode_filename_get_recording_technique)

"""### Bringing the categorical values to the begining"""

col = df.columns.tolist()
col = [col[0]] + col [-3:] + col[1:-3]
df = df[col]
print("The first five rows of the dataframe\n", df.head())

"""### Checking missing values"""

print("Checking if there are any null values\n",df.isnull().sum())

#displaying sample EEG graph from each set
plt.figure(figsize= (30,20))
print('data',df)
plt.subplot(3,2,1)
file_name = 'Z093.txt'
set_code = df[df['Filename'] == file_name].iloc[0,2]

x = [i for i in range(1,4098)]
y = df[df['Filename'] == file_name].iloc[0,4:]

plt.title(f'File Set : {set_code} File Name : {file_name}')
plt.plot(x, y )

plt.subplot(3,2,2)

file_name = 'O015.txt'
set_code = df[df['Filename'] == file_name].iloc[0,2]

x = [i for i in range(1,4098)]
y = df[df['Filename'] == file_name].iloc[0,4:]

plt.title(f'File Set : {set_code} File Name : {file_name}')
plt.plot(x, y )

plt.subplot(3,2,3)
file_name = 'N062.TXT'
set_code = df[df['Filename'] == file_name].iloc[0,2]

x = [i for i in range(1,4098)]
y = df[df['Filename'] == file_name].iloc[0,4:]

plt.title(f'File Set : {set_code} File Name : {file_name}')
plt.plot(x, y )

plt.subplot(3,2,4)
file_name = 'F021.txt'
set_code = df[df['Filename'] == file_name].iloc[0,2]

x = [i for i in range(1,4098)]
y = df[df['Filename'] == file_name].iloc[0,4:]

plt.title(f'File Set : {set_code} File Name : {file_name}')
plt.plot(x, y )


plt.subplot(3,2,5)
file_name = 'S056.txt'
set_code = df[df['Filename'] == file_name].iloc[0,2]

x = [i for i in range(1,4098)]
y = df[df['Filename'] == file_name].iloc[0,4:]

plt.title(f'File Set : {set_code} File Name : {file_name}')
plt.plot(x, y )

plt.show()

df.to_csv("data/data_preprocessed.csv") #saving the preprocessed dataset
"""From the samples timeseries from each set, it can be observed that  
A and B looks normal  
C has occasional spikes  
D has more occasional spikes  
where as E has very high number of hikes  

There is a chance that Set E contains the values during Seizure

The folders Contains numerous .txt files, each likely representing an individual EEG recording or data file. Each .txt file in these folders probably contains EEG data points. The sample data from each folder ('Z', 'S', 'F', 'N', 'O') appears to be in a similar format. Given this structure, it seems that each file represents a single EEG recording, with each row likely corresponding to a signal measurement at a specific time point. The data is univariate, meaning each file contains measurements from a single EEG channel or a specific feature extracted from the EEG signal.

All the files have the same number of samples (4,097), which indicates a consistency in data collection or recording length. However, the range of values and the mean values vary significantly between files. Such variations are expected in EEG data, as they reflect different brain activities and possibly different conditions (such as epileptic seizures versus normal brain function).

Each .txt file is encoded into 4097 samples and all F.txt files are coded as 1, N as 2, O as 3, S as 4, Z as 5. File set has also be named to F as D, N as C, O as B, S as E, Z as A.  

After performing all the steps the totals rows are set to be 500 and columns are 4100."""