
#importing all the necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import xgboost as xgb
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

warnings.filterwarnings('ignore')

"""In the process of selecting an appropriate model for our project, careful consideration was given to the size and nature of our dataset. With an initial dataset consisting of 500 samples, each with 4097 features, preprocessing efforts have resulted in a more streamlined dataset of 500 samples with only 4 features. This reduction in dimensionality raises concerns about the suitability of certain models, particularly Convolutional Neural Networks (CNNs). CNNs are renowned for their efficacy in handling image data with inherent spatial relationships, often represented as 2D grids. Given the transformed nature of our dataset, which lacks the grid-like structure associated with images, opting for a model tailored to tabular data or simpler structures appears more prudent.

# Model Evaluation and Testing
"""

def labelling(value):
    if 'E' in value:
        return 1
    else:
        return 0

df  = pd.read_csv(r"data\data_preprocessed.csv")

# Create a new column based on the condition
df['label'] = df['file_set'].apply(labelling)

feature_df = pd.read_csv(r"data\feature_df.csv")

# Assuming feature_df is your DataFrame with features
# 'file_name' column is dropped as it's not used for training
X = feature_df[['mean', 'variance', 'rms', 'std_dev','Peak_Frequency', 'Delta_Power','Theta_Power','Alpha_Power', 'Beta_Power', 'Gamma_Power']].values
y = feature_df['label'].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=40, stratify=y)

# load the model from disk
best_rf = pickle.load(open(r"model\random_forest.sav", 'rb'))
best_xgb = pickle.load(open(r"model\xg_boost.sav", 'rb'))
rnn_final_model = pickle.load(open(r"model\rnn.sav", 'rb'))
# Evaluation Metric

evaluation_df = pd.DataFrame(columns=['Model', 'Accuray', "Precision", 'Recall', 'F1 Score'])

# Predictions on test data
rf_y_pred = best_rf.predict(X_test)

# Efficiency metrics
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_precision = precision_score(y_test, rf_y_pred)
rf_recall = recall_score(y_test, rf_y_pred)
rf_f1 = f1_score(y_test, rf_y_pred)
rf_conf_matrix = confusion_matrix(y_test, rf_y_pred)

new_row = {'Model': 'Random Forest', 'Accuray': rf_accuracy, 'Precision' : rf_precision, 'Recall' : rf_recall, 'F1 Score' : rf_f1}
evaluation_df = evaluation_df.append(new_row, ignore_index= True)


# Predictions on test data
xgb_y_pred = best_xgb.predict(X_test)

# Efficiency metrics
xgb_accuracy = accuracy_score(y_test, xgb_y_pred)
xgb_precision = precision_score(y_test, xgb_y_pred)
xgb_recall = recall_score(y_test, xgb_y_pred)
xgb_f1 = f1_score(y_test, xgb_y_pred)
xgb_conf_matrix = confusion_matrix(y_test, xgb_y_pred)

new_row = {'Model': 'XgBoost', 'Accuray': xgb_accuracy, 'Precision' : xgb_precision, 'Recall' : xgb_recall, 'F1 Score' : xgb_f1}
evaluation_df = evaluation_df.append(new_row, ignore_index= True)


 # Evaluate the final model on test data
rnn_y_pred_probs = rnn_final_model.predict(X_test)
rnn_y_pred = (rnn_y_pred_probs > 0.5).astype(int)

# Efficiency metrics
rnn_accuracy = accuracy_score(y_test, rnn_y_pred)
rnn_precision = precision_score(y_test, rnn_y_pred)
rnn_recall = recall_score(y_test, rnn_y_pred)
rnn_f1 = f1_score(y_test, rnn_y_pred)
rnn_conf_matrix = confusion_matrix(y_test, rnn_y_pred)

new_row = {'Model': 'RNN', 'Accuray': rnn_accuracy, 'Precision' : rnn_precision, 'Recall' : rnn_recall, 'F1 Score' : rnn_f1}
evaluation_df = evaluation_df.append(new_row, ignore_index= True)



print("\nEvaluaion Metrics on Test Data \n")
print(evaluation_df)

"""The Random Forest model shows exceptional performance with an accuracy of 99% and a perfect recall of 1.00, indicating its proficiency in identifying all relevant cases. Its F1 score of approximately 0.976 suggests a strong balance between precision and recall. The XgBoost model, with an accuracy of 97% and an F1 score of around 0.923, also performs admirably, balancing precision and recall effectively. Lastly, the RNN model, while matching the 97% accuracy of XgBoost, stands out with a perfect precision of 1.00, though its recall is slightly lower at 0.85, as reflected in its F1 score of approximately 0.919.

# Results and Visualization

### Visualize the EEG data and model predictions. Create plots and graphs to illustrate your findings.

# Visualizing the F1 Score
"""

# F1 scores for each model
models = ['Random Forest', 'XgBoost', 'RNN']
scores = [rf_f1, xgb_f1, rnn_f1]

# Creating the bar chart
plt.bar(models, scores, color=['blue', 'green', 'red'])

# Annotating the F2 scores on the bars
for i, score in enumerate(scores):
    plt.text(i, score + 0.01, f'{score:.2f}', ha='center', va='bottom')

# Adjusting the y-axis limits
plt.ylim(0.8, 1)

# Adding title and labels
plt.title('F1 Scores of Different Models')
plt.xlabel('Models')
plt.ylabel('F1 Score')

# Display the chart
plt.show()

"""Random Forest and RNN have high scores around 0.98, while XgBoost has a slightly lower score of 0.92. The accompanying text indicates strong performance in precision-recall trade-off for the Random Forest and XgBoost models, with RNN slightly behind. With Random Forest and RNN outperforming XgBoost, all scoring above 0.90, indicating a high level of precision and recall in their performance.

# Visualizing the confusion matrix of different models
"""

# Creating the heatmap
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0', '1'],
            yticklabels=['0', '1'])

# Adding title and labels
plt.title('Confusion Matrix of Random Forest Model')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# Display the heatmap
plt.show()

# Creating the heatmap
sns.heatmap(xgb_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0', '1'],
            yticklabels=['0', '1'])

# Adding title and labels
plt.title('Confusion Matrix of XgBoost Model')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# Display the heatmap
plt.show()

# Creating the heatmap
sns.heatmap(rnn_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0', '1'],
            yticklabels=['0', '1'])

# Adding title and labels
plt.title('Confusion Matrix of RNN Model')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# Display the heatmap
plt.show()

"""Random Forest, XgBoost, and RNN, as part of a binary classification task. All three models show a significant number of true positives (79) and true negatives (20 for Random Forest and RNN, 18 for XgBoost), with very few false negatives and false positives, indicating high performance. The Random Forest and RNN models have no false negatives, while the XgBoost model has two. These results suggest that the Random Forest and RNN models are slightly better at correctly classifying both positive and negative classes than XgBoost in this particular task. These matrices underscore the models' robustness in correctly classifying the majority of the test data."""

#visualising EEG data
seizure_data = df[df['label'] == 1].iloc[0,4:-1].tolist()
non_seizure_data = df[df['label'] == 0].iloc[0,4:-1].tolist()

plt.figure(figsize= (30,20))

plt.subplot(2,1,1)

x = [i for i in range(1,4098)]
y = seizure_data

plt.title(f'Seizure Data')
plt.plot(x, y )

plt.subplot(2,1,2)

x = [i for i in range(1,4098)]
y = non_seizure_data

plt.title(f'non_seizure_data')
plt.plot(x, y )

plt.show()

"""# Multi Class Classification

- As random forest has performed best in the binary classification of Seizure or non seizure data.
- Random forest has been used for multi class clasification to classify data to multi classes bases on the EEG data
"""

# Assuming feature_df is your DataFrame with features
# 'file_name' column is dropped as it's not used for training
X = feature_df[['mean', 'variance', 'rms', 'std_dev','Peak_Frequency', 'Delta_Power','Theta_Power','Alpha_Power', 'Beta_Power', 'Gamma_Power']].values
y = df['file_code'].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=40, stratify=y)

#Random Forest
# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3]
}

# Create a RandomForest model
rf = RandomForestClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

# Perform grid search
grid_search.fit(X_train, y_train)

# Best parameters and best score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

# Use the best estimator for making predictions
best_rf = grid_search.best_estimator_

# Predictions on test data
rf_y_pred = best_rf.predict(X_test)

# Efficiency metrics
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_precision = precision_score(y_test, rf_y_pred, average='weighted')
rf_recall = recall_score(y_test, rf_y_pred, average='weighted')
rf_f1 = f1_score(y_test, rf_y_pred, average='weighted')
rf_conf_matrix = confusion_matrix(y_test, rf_y_pred)

print(f'Accuracy: {rf_accuracy:.4f}')
print(f'Precision: {rf_precision:.4f}')
print(f'Recall: {rf_recall:.4f}')
print(f'F1 Score: {rf_f1:.4f}')

# Define class labels
class_labels = ['Epileptogenic Zone : Seizure Free', 'Hippocampal Formation : Seizure Free', 'Eyes Closed : Seizure Free', 'Seizure Activity', 'Eyes Open : Seizure Free']

# Plot the confusion matrix using Seaborn heatmap
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Random Forest Confusion Matrix')
plt.show()

"""The Random Forest model, used for multi-class classification of EEG data, leveraged features like mean and power bands. After standardizing the data, the model was fine-tuned using GridSearchCV. This resulted in an accurate classifier capable of differentiating between various brain activity states. The effectiveness of the model was demonstrated through a confusion matrix."""
