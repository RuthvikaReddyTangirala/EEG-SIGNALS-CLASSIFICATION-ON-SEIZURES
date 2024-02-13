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


