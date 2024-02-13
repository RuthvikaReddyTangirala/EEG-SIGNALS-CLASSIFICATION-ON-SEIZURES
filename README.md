# Introduction
The project focuses on the analysis of EEG data using the Bonn EEG Dataset, a crucial resource 
in the field of neuroscience and medical diagnostics. Electroencephalography (EEG) is a non
invasive method that records the electrical activity of the brain. This technique is vital in 
studying brain functions and diagnosing conditions like epilepsy. The Bonn EEG Dataset 
specifically includes recordings related to epileptic seizures, offering a unique opportunity to 
explore and classify different patterns of brain activity associated with epilepsy. By applying 
advanced classification models to this dataset, the project aims to enhance the understanding 
and detection of epileptic seizures, which is crucial for improving patient care and treatment 
strategies. This research leverages the high-resolution temporal data provided by EEG 
recordings to identify distinctive patterns that differentiate between normal and abnormal 
brain activity, a step forward in the field of biomedical engineering and neuroscience. 

## Installation
Prerequisites
Python 3.x
pip or conda
Git (for cloning the repository)

## Steps
##### Clone the repository:
git clone https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES.git

##### Navigate to the project directory:
cd EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES

##### Install the required dependencies:
pip install -r requirements.txt




## Overview and Data Preprocessing of Dataset
Data preprocessing is done on electroencephalogram (EEG) data that is saved in text files. Each file's content is read, ASCII data is converted to integers, and the data is arranged into a DataFrame. Filenames, sample indices, and matching EEG data are all displayed in a column of the DataFrame. By presenting details about the DataFrame, descriptive statistics, and sample EEG graphs for each file, the code offers more insights into the dataset.  
 
The dataset contains 500 rows and 4100 columns after extracting and encoding. The raw data is cleaned, transformed, and organized to enhance its quality. 

Exploring the data is essential to unveil its inherent patterns and characteristics, providing valuable insights into its structure, distribution, and key features before initiating any analysis or modeling. 
![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/55bcf864-3168-4d2d-9574-951993973540)
![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/f64a8a1a-8e9f-4666-8621-b32bc93e03f6)

After preprocessing the EEG data, including handling missing values, noise reduction, and data augmentation: 
Matplotlib is utilized to show EEG data from files, which aids in comprehending the features and organization of the information. Sample EEG graph visualization improves comprehension of the distribution and substance of the dataset, providing the foundation for further research and model building. 
![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/fa88cf59-2834-40ee-8e30-d08e74c81166)

From the samples timeseries from each set, it can be observed that 
A and B looks normal 
C has occasional spikes D has more occasional spikes 
whereas E has very high number of hikes 
 
There is a chance that Set E contains the values during Seizure 
 
The folders Contains numerous .txt files, each likely representing an individual EEG recording or data file. Each .txt file in these folders probably contains EEG data points. The sample data from each folder ('Z', 'S', 'F', 'N', 'O') appears to be in a similar format. Given this structure, it seems that each file represents a single EEG recording, with each row likely corresponding to a signal measurement at a specific time point. The data is univariate, meaning each file contains measurements from a single EEG channel or a specific feature extracted from the EEG signal. 
 
All the files have the same number of samples (4,097), which indicates a consistency in data collection or recording length. However, the range of values and the mean values vary significantly between files. Such variations are expected in EEG data, as they reflect different brain activities and possibly different conditions (such as epileptic seizures versus normal brain function). 
 
Each .txt file is encoded into 4097 samples and all F.txt files are coded as 1, N as 2, O as 3, S as 4, Z as 5. File set has also been named to F as D, N as C, O as B, S as E, Z as A. 
 
After performing all the steps, the totals rows are set to be 500 and columns are 4100. 

## Feature Extraction 
 
Feature extraction involves selecting and transforming relevant information from raw data to create a concise set of features that capture essential patterns, reducing dimensionality and improving model performance 
Two distinct approaches are employed for feature extraction from EEG signals: time-domain and frequency-domain. 
 
##### Time Domain Features: 
From the time domain EEG data, the extract features time domain function computes statistical quantities including mean, variance, root mean square (RMS), and standard deviation. These characteristics record the fundamental amplitude and variability of the signal. 
 
![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/4aea883b-174c-4be7-95ec-96dbcf080936)

 
##### Frequency Domain Features: 
The Power Spectral Density (PSD) may be computed using functions like calculate_psd. For given frequency bands (delta, theta, alpha, beta, and gamma), functions like calculate_peak_frequency and bandpower can be used to extract features like peak frequency and power. Understanding the distribution of signal strength among various frequency components is made possible by these characteristics. 
  
![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/93c18522-9669-4ee3-8b87-9cd8b5fffb38)


Following their extraction, the features are arranged into a new DataFrame that includes properties in the temporal and frequency domains. The last piece of code examines the correlations between time-domain characteristics (variance, standard deviation, and RMS) and the mean value for each EEG signal is visualized using scatterplot. Because the scatterplots are color-coded according to the file code, it is possible to visually evaluate how various files or categories could display unique patterns in the feature space. 

 ![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/53e9d9ba-c34e-4ba8-8503-4bafc7c94600)

  
From the Above graphs based on time-based features it can be observed that values of Set S differ from all others significantly, Set D values has occasionally differed from other. 
 
From the given paper: 
 
Volunteers were relaxed in an awake state with eyes open ~A! and eyes closed ~B! respectively. Sets C, D, and E originated from our EEG archive of presurgical diagnosis. For the present study EEGs from five patients were selected, all of whom had achieved complete seizure control after resection of one of the hippocampal formations, which was therefore correctly diagnosed to be the epileptogenic zone ~cf. Segments in set D were recorded from within the epileptogenic zone, and those in set C from the hippocampal formation of the opposite hemisphere of the brain. While sets C and D contained only activity measured during seizure free intervals, set E only contained seizure activity. Here segments were selected from all recording sites exhibiting ictal activity. 
 
Hence it is given that Set E has been collected during activity and it can also be observed in the above charts. Labelling the rows from Set E as 1 indicating it is a seizure. 
 
![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/753070ce-8d1a-4c02-80fe-48e79e7c59ff)

 


