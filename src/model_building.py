#importing all the necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import warnings
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from feature_engineering import feature_df
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch

warnings.filterwarnings('ignore')
"""# Data Splitting

### Split the data into training, validation, and test sets.
"""

# Assuming feature_df is your DataFrame with features
# 'file_name' column is dropped as it's not used for training
X = feature_df[['mean', 'variance', 'rms', 'std_dev','Peak_Frequency', 'Delta_Power','Theta_Power','Alpha_Power', 'Beta_Power', 'Gamma_Power']].values
y = feature_df['label'].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=40, stratify=y)

"""# Model Selection and Model Training and Model Evaluation with Hyper Parameter Tuning.

## Training the Random Forest Model

- Hyper parameter tuning is being performed using Grid Search CV

- Class weights has been calculated to balance the imbalance in the data

## Calculate the weights as follows:

- Weight for class 0 = Total Instances / (Number of Classes * Instances in Class 0) = 500 / (2 * 400) = 0.625
- Weight for class 1 = Total Instances / (Number of Classes * Instances in Class 1) = 500 / (2 * 100) = 2.5
"""

#Random Forest
# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3]
}

# Create a RandomForest model
rf = RandomForestClassifier(random_state=42, class_weight={0: 0.625, 1: 2.5})

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

# Perform grid search
grid_search.fit(X_train, y_train)

# Best parameters and best score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

# Use the best estimator for making predictions
best_rf = grid_search.best_estimator_

"""# Training the XgBoost Model

- Hyper parameter tuning is being performed using Grid Search CV

- Imbalance in the data has been dealt using scale_pos_weight
"""

#XGBoost

# Define the parameter grid
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Create a XGBClassifier model
xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=4  # Set the scale_pos_weight parameter as in your original model
)

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy'
)

# Perform grid search
grid_search.fit(X_train, y_train)

# Best parameters and best score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

# Use the best estimator for making predictions
best_xgb = grid_search.best_estimator_

"""# Training the RNN Model

- Hyper parameter tuning is being performed using Kears Tuner

- Class weights has been calculated to balance the imbalance in the data

- RNN creates a set of Validation set from training set and calculates the validation loss and validation accuracy
"""

#RNN with keras tuner

def build_model(hp):
    model = Sequential()
    model.add(SimpleRNN(units=hp.Int('units', min_value=32, max_value=128, step=32),
                        input_shape=(X_train.shape[1], 1),
                        activation='relu',
                        return_sequences=True))
    model.add(Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(SimpleRNN(units=hp.Int('units', min_value=32, max_value=128, step=32),
                        activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
class_weight = {0 : 0.625, 1: 2.5}
# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Define the Keras Tuner RandomSearch
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,  # You can increase this based on your computational resources
    directory='keras_tuner_dir',
    project_name='rnn_hyperparameter_tuning'
)

# Search for the best hyperparameters
tuner.search(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping], class_weight = class_weight)

# Get the best hyperparameters
rnn_best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:", rnn_best_hps)

# Build the final model with the best hyperparameters
rnn_final_model = tuner.hypermodel.build(rnn_best_hps)
print("Summary of RNN model\n", rnn_final_model.summary())

# Train the final model
rnn_final_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# save the model to disk
filename = r'model\random_forest.sav'

pickle.dump(best_rf, open(filename, 'wb'))

filename = r'model\xg_boost.sav'

pickle.dump(best_xgb, open(filename, 'wb'))

filename = r'model\rnn.sav'

pickle.dump(rnn_final_model, open(filename, 'wb'))

print("successfully ran")