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

## Dataset Source

https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/

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

##### Storing the data
Unzip all the five folders and store it in the path data\data

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

## MODEL ARCHITECTURE AND TRAINING DETAILS 
 
### Model Selection, Model Training, and Model Evaluation with Hyperparameter Tuning. 
### Hyperparameter Tuning 
Process of optimizing a model's external configuration settings for improved performance. Unlike internal parameters adjusted during training, hyperparameters, such as learning rates and regularization strengths, are predefined and guide the learning process. Key considerations: 
•	Grid Search: A common method involves systematically trying hyperparameter combinations from a defined grid to find the best-performing set. 
•	Random Search: Alternatively, random combinations of hyperparameter values are sampled and evaluated, especially useful in high-dimensional spaces. 
•	Cross-Validation: Essential for assessing model generalization, cross-validation involves splitting the dataset, training on subsets, and evaluating on the remaining data. 
•	Overfitting and Underfitting: Hyperparameter tuning aims to balance overfitting (capturing noise) and underfitting (missing patterns) through techniques like regularization. 
•	Automated Tuning: Libraries like sci-kit-learn and TensorFlow offer tools for automated hyperparameter tuning, including grid search, random search, and advanced methods like Bayesian optimization. 
 
Model Selection 
Involves selecting the optimal machine learning algorithm for a task through the evaluation and comparison of different models' performance. 
Model Training 
A machine learning model learns patterns from labelled data, refining its parameters to accurately predict outcomes on new, unseen data. 

### Random Forest Model 
Objective: Efficiently handles classification tasks, especially in the context of data imbalance. 
Training: Trained on labeled data using the Random Forest algorithm. 
Hyperparameter Tuning: Utilized Grid Search CV for optimizing hyperparameters. 
Class Imbalance: Addressed by calculating class weights to ensure balanced learning. Outcome: Achieved robust performance in classification tasks. 
Calculate the weights as follows 
•	Weight for class 0 = Total Instances / (Number of Classes * Instances in Class 0) = 500 / (2 * 400) = 0.625 
•	Weight for class 1 = Total Instances / (Number of Classes * Instances in Class 1) = 500 / (2 * 100) = 2.5 
 
![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/9d7cd084-a993-4553-a3be-6f189cfac48d)

 
### XgBoost Model 
Objective: Optimized for predictive accuracy, particularly in datasets with class imbalance. 
Training: Trained using the XGBoost algorithm with a focus on boosting decision trees. 
Hyperparameter Tuning: Implemented Grid Search CV to fine-tune hyperparameters. 
Data Imbalance Handling: Utilized scale_pos_weight to address imbalanced class distribution. Outcome: Demonstrated strong predictive performance, especially in scenarios with imbalanced classes. 

![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/895598e9-db6b-4029-abbb-2a5c03aa7f1a)

 
### Recurrent Neural Network (RNN) Model 
Objective: Suited for sequential data, such as time series or natural language processing. 
Training: Trained using Keras Tuner, emphasizing the importance of sequence learning. 
Hyperparameter Tuning: Applied Keras Tuner for efficient tuning of RNN-specific parameters. 
Class Imbalance: Managed by calculating class weights to ensure fair representation. 
Validation Approach: Created validation sets from the training data to assess performance. 
Outcome: Effective in capturing temporal dependencies, making it suitable for sequential data and achieving competitive performance. 

![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/64390218-8f7a-44ba-81a5-3f89f82d0e81)

### Model Evaluation 
It encompasses analyzing the model’s accuracy on a distinct, unseen dataset to evaluate its capacity for generalization and accurate predictions.  
In the provided code, model evaluation and testing are conducted for three different models: Random Forest, XGBoost, and Recurrent Neural Network (RNN). Here's an explanation of the process: 

![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/585c8320-1e9a-45a2-ba09-890f7795e569)

 
 
### Random Forest Model Evaluation 
Predictions: The model's predictions (rf_y_pred) are generated using the test data (X_test). Efficiency Metrics: Accuracy, Precision, Recall, and F1 Score are calculated using the ground truth (y_test) and predicted values. 
Confusion Matrix: The confusion matrix for the Random Forest model (rf_conf_matrix) is also computed. 
Storage: Evaluation metrics are appended to the evaluation_df DataFrame for later comparison. 
 
### XGBoost Model Evaluation 
Predictions: The XGBoost model's predictions (xgb_y_pred) are generated on the test data. 
Efficiency Metrics: Accuracy, Precision, Recall, and F1 Score are calculated and stored. 
Confusion Matrix: The confusion matrix for the XGBoost model (xgb_conf_matrix) is computed and stored. 
 
### RNN Model Evaluation 
Predictions: The RNN model's predictions (rnn_y_pred) are generated on the test data, and probability scores (rnn_y_pred_probs) are obtained. 
Thresholding: A threshold of 0.5 is applied to convert probability scores into binary predictions. 
Efficiency Metrics: Accuracy, Precision, Recall, and F1 Score are calculated for the RNN model. 
Confusion Matrix: The confusion matrix for the RNN model (rnn_conf_matrix) is computed. 
 
 
### Random Forest Classifier 
For seizure detection using EEG data, we implemented a Random Forest classifier optimized via GridSearchCV. The search tested combinations of hyperparameters including the number of trees (n_estimators), the maximum depth of the trees (max_depth), the minimum number of samples required to split an internal node (min_samples_split), and the minimum number of samples required to be at a leaf node (min_samples_leaf). The best-performing model utilized 100 trees (n_estimators), a maximum depth of 3 (max_depth), and required at least one sample at each leaf (min_samples_leaf) and two samples to split (min_samples_split). This configuration achieved an accuracy score of 0.98, indicating an excellent fit to the training data. The model was balanced for class weights due to class imbalance, with weights of 0.625 for class 0 and 2.5 for class 1. This balancing improves the model's sensitivity to the minority class, which is critical in medical diagnostics. 
 
### XGBoost Classifier 
 
The XGBoost model was constructed with a binary logistic objective and log loss evaluation metric. The hyperparameter tuning was conducted via GridSearchCV over a diverse parameter grid, focusing on max_depth, learning_rate, n_estimators, subsample, and colsample_bytree. The optimal parameters included a learning rate of 0.1, a max_depth of 4, n_estimators at 
200, a subsample rate of 0.9, and a colsample_bytree of 0.8. These settings led to a best score of 0.9825, suggesting a highly effective model for classifying EEG data. The scale_pos_weight was set to 4 to address class imbalance, a vital adjustment to ensure model sensitivity towards the minority class. 
 
### RNN Model 
The RNN model for EEG data classification was optimized using Keras Tuner with a Sequential architecture. It comprises two SimpleRNN layers with a tunable number of units between 32 and 128 and relu activation. Dropout layers were included to prevent overfitting, with rates between 0.2 and 0.5. The model uses a sigmoid activation function in the output layer for binary classification. Adam optimizer with a learning rate selected from [1e-2, 1e-3, 1e-4] was employed, and binary_crossentropy was chosen as the loss function. The best model, after hyperparameter tuning, indicated a modest number of parameters, reflecting a balance between model complexity and computational efficiency. Training involved an early stopping callback on validation loss with patience set to 10 epochs to further mitigate overfitting. The class weights were adjusted to account for class imbalance, improving the model's performance on minority classes. 
 
### Multiclass Classification using Random Forest 
 
The Random Forest model was employed for multi-class classification of EEG data. Feature selection included statistical and power band values, standardized before training. A GridSearchCV approach optimized hyperparameters, resulting in a model with an accuracy score of 0.98. The model's ability to classify EEG data into various states was visualized via a confusion matrix, revealing its capacity to distinguish between different brain activity states with high precision, recall, and F1 scores. This analysis extends the utility of the Random Forest classifier beyond binary to multi-class EEG data classification. 

## EVALUATION RESULTS AND DISCUSSION 
 
### Evaluation of EEG Graphs 
 
![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/f64213d4-39bd-4cd3-8c36-ad8df7af069f)

![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/7d1b1d0a-f11f-459f-a31d-f5a12b0b6b44) 

![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/1ab24eef-fe77-4aad-ab2d-4e2c1354dd9a) 

![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/63a84908-69ed-476e-801f-d6d1419cd90c) 

![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/8c9ec057-8dae-4f5b-9f0d-430363ccb6a2)

The visualization of the EEG data from the Bonn EEG Dataset reveals significant variability across different sets, suggesting distinct brain activity patterns. Set E, with its high number of spikes, aligns with the expected characteristics of seizure activity, differentiating it from other sets. The consistency in sample size across the data points to standardized recording practices, essential for reliable analysis. 
The graphical analysis provides insights into the data sets’ dynamics, with Set S showing pronounced deviation, indicative of its unique brain state. Such distinctions are critical for the development of accurate classification models. 
By labelling Set E as seizure activity based on both the observed patterns and the paper's information, the project advances towards creating a model that can discern between seizure and non-seizure states. This step is key in devising effective diagnostic tools for epilepsy using machine learning techniques applied to EEG data.  
 
## MODEL EVALUATION AND TESTING 
 
![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/61871fb6-ebb3-4815-8ccb-a2a5eced0e22)


In the detailed evaluation of classification models for EEG data, the Random Forest model stands out with a 99% accuracy and a perfect recall score of 1.00, demonstrating exceptional proficiency in detecting seizure activity without any false negatives. Its F1 score of approximately 0.976 suggests an excellent balance between precision and recall, making it highly reliable for clinical applications where the cost of missing a true seizure is critical. 
The XgBoost model, with an accuracy of 97% and an F1 score of around 0.923, also performs admirably, striking a balance between precision and recall effectively. It shows a commendable ability to classify EEG data accurately, but with slightly less sensitivity compared to the Random Forest model. 
Lastly, the RNN model, while achieving a high accuracy of 99%, presents a contrast with a perfect precision score but a lower recall. This indicates that while the RNN model is excellent at correctly identifying seizure events when it predicts them, it may miss some seizure events, which could be a potential limitation in its deployment for seizure detection in a clinical setting. 
Overall, these results underscore the importance of model selection based on performance metrics that align with clinical priorities, such as minimizing false negatives for seizure detection. 
 
## F1 Score visualisation of different models 

![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/687ca3ea-7a39-4cf7-b125-a1ad8dcb8862)

In the evaluation of classification models for EEG data, the F1 score serves as a critical metric combining precision and recall into a single measure. The graph presents a comparative analysis of three models: Random Forest, XgBoost, and RNN. Random Forest and RNN emerge as the leading models, both achieving an F1 score of 0.98, suggesting an exceptional balance between false positives and false negatives. XgBoost, with a score of 0.92, although slightly lower, still indicates a high degree of model accuracy. These results are promising for the application of machine learning in clinical diagnostics, particularly in identifying seizurerelated activity in EEG recordings. The high performance of Random Forest and RNN models may be attributed to their ability to capture complex patterns within the data, which is crucial for distinguishing between normal brain activity and epileptic seizures. This analysis underscores the potential of advanced machine learning techniques in improving the precision of medical diagnostics.  
 
## Confusion matrix for different models 
  
![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/7a05eef4-8529-47ca-8580-37d816476b4e)

![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/feb9703b-2b75-4da1-ba1b-0a276a2f3f80) 

![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/16b651be-417b-4d73-97cb-d6ad2fc188f0)
 
The confusion matrices for the Random Forest, XgBoost, and RNN models reveal their performance in a binary classification task. Random Forest and RNN demonstrate impeccable specificity and sensitivity with only a single false positive and no false negatives, indicating a near-perfect classification of the seizure and non-seizure classes. XgBoost, while exhibiting a commendable performance, shows a slight increase in false negatives. This may suggest a tendency to under-predict the seizure class, which could be critical in clinical applications where missing a seizure event is more detrimental than a false alarm. The high accuracy of the Random Forest and RNN models suggests that they are better suited for scenarios where the cost of misclassification is high. These results reinforce the importance of choosing the right model based on the specific requirements of the task at hand, especially in medical diagnostics where the stakes are significant. 

## Confusion Matrix of Multiclass Classifier 
   
![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/33301fd0-9833-4091-8c29-34b69ac7d3d4)


The confusion matrix for the Random Forest model indicates a high degree of accuracy in classifying various EEG states. The model shows a strong ability to differentiate between seizure and non-seizure activity. Most misclassifications occur with the 'Epileptogenic Zone: Seizure Free' and 'Hippocampal Formation: Seizure Free' categories, suggesting similarities in EEG patterns. Overall, the model demonstrates effectiveness in EEG data classification, with potential implications for improving diagnostic procedures in neurology. 

## Visualizing the EEG data 
 
![image](https://github.com/RuthvikaReddyTangirala/EEG-SIGNALS-CLASSIFICATION-ON-SEIZURES/assets/113473457/8d384ee3-319e-446e-aaef-041c3b7b8f45)


The EEG data visualization includes two subplots that depict patterns during seizure and non-seizure states.  
The initial subplot illustrates distinctive EEG data patterns during a seizure, while the second subplot presents the EEG data during non-seizure periods, revealing a contrasting pattern.  
This visual representation contributes to a better understanding of the unique characteristics exhibited in EEG data during seizures and non-seizure conditions. 
 
 
## Conclusion and future work 
### Conclusion: 
The comprehensive analysis of EEG data classification has underscored the capabilities of advanced machine learning algorithms in distinguishing between normal and epileptic seizure states. The Random Forest model emerged as a notably effective classifier, with its high accuracy and recall rates being particularly promising for applications in medical diagnostics where the cost of false negatives is exceedingly high. XgBoost and RNN models also demonstrated significant potential, with robust overall performance metrics. 
 
### Future Work: 
Further investigation is warranted to enhance the accuracy and generalizability of these models across more diverse and larger datasets, which may encompass a broader spectrum of seizure types and patient backgrounds. Advancements in deep learning, particularly in convolutional neural networks (CNNs) and recurrent neural networks (RNNs), could be leveraged to handle multi-dimensional EEG data, providing a more nuanced understanding of the brain's electrical activity. Another promising direction is the real-time application of these models in clinical and ambulatory settings, which could lead to proactive seizure detection systems. This could not only improve patient outcomes but also catalyze the development of personalized treatment plans. Moreover, the intersection of EEG data analysis with other physiological data through multi-modal approaches may offer new insights into the mechanisms underlying epilepsy, facilitating the advent of holistic and patient-centric healthcare solutions. 


