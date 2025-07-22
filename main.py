import models.random_forest as rf
import models.xgboost as xgb
import os
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)
dataset_folder = os.getenv("DATASET_FOLDER")
model_output = os.getenv("MODEL_OUTPUT")

# Define the Random Forest model
rf_model = rf.RandomForestModelTrainer(seed=42)

# Train the model on MAL_1 dataset
# Best parameters found:  {'ccp_alpha': np.float64(0.0), 'class_weight': 'balanced', 'max_depth': 30, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
# Accuracy: 0.8456, macro recall: 0.56, weight recall: 0.85
rf_model.train(os.path.join(dataset_folder, 'MAL_1.pkl'), model_output, parameters={
    'ccp_alpha': [0.0],
    'class_weight': ['balanced'],
    'max_depth': [30],
    'max_features': ['log2'],
    'min_samples_leaf': [1],
    'min_samples_split': [2],
    'n_estimators': [300]
})

# Train the model on MAL_2 dataset
# Best parameters found:  {'ccp_alpha': np.float64(0.0), 'class_weight': None, 'max_depth': 20, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
# Accuracy: 0.9726, macro recall: 0.68, weight recall: 0.97
rf_model.train(os.path.join(dataset_folder, 'MAL_2.pkl'), model_output, parameters={
    'ccp_alpha': [0.0],
    'class_weight': [None],
    'max_depth': [20],
    'max_features': ['log2'],
    'min_samples_leaf': [1],
    'min_samples_split': [2],
    'n_estimators': [200]
})

# Train the model on CAP_1 dataset
# Best parameters found:  {'ccp_alpha': np.float64(0.0), 'class_weight': 'balanced', 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 300}
# Accuracy: 0.8074, macro recall: 0.65, weight recall: 0.80
rf_model.train(os.path.join(dataset_folder, 'CAP_1.pkl'), model_output, parameters={
    'ccp_alpha': [0.0],
    'class_weight': ['balanced'],
    'max_depth': [10],
    'max_features': ['sqrt'],
    'min_samples_leaf': [1],
    'min_samples_split': [10],
    'n_estimators': [300]
})

# Train the model on CAP_2 dataset
# Best parameters found:  {'ccp_alpha': np.float64(0.0), 'class_weight': 'balanced', 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200}
# Accuracy: 0.8, macro recall: 0.62, weight recall: 0.80
rf_model.train(os.path.join(dataset_folder, 'CAP_2.pkl'), model_output, parameters={
    'ccp_alpha': [0.0],
    'class_weight': ['balanced'],
    'max_depth': [30],
    'max_features': ['sqrt'],
    'min_samples_leaf': [4],
    'min_samples_split': [2],
    'n_estimators': [200]
})

# Define the XGBoost model
xgb_model = xgb.XGBoostModelTrainer(seed=42)

# Train the model on MAL_1 dataset
# Best parameters found: {'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.8}
# Accuracy: 0.8600, macro recall: 0.66, weight recall: 0.86
xgb_model.train(os.path.join(dataset_folder, 'MAL_1.pkl'), model_output, parameters={
    'colsample_bytree': [1],
    'learning_rate': [0.1],
    'max_depth': [7],
    'n_estimators': [200],
    'subsample': [0.8]
})

# Train the model on MAL_2 dataset
# Best parameters found: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 100, 'subsample': 1}
# Accuracy: 0.9769, macro recall: 0.76, weight recall: 0.98
xgb_model.train(os.path.join(dataset_folder, 'MAL_2.pkl'), model_output, parameters={
    'colsample_bytree': [0.8],
    'learning_rate': [0.1],
    'max_depth': [10],
    'n_estimators': [100],
    'subsample': [1]
})

# Best parameters found: {'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 200, 'subsample': 0.6}
# Accuracy: 0.8088, macro recall: 0.62, weight recall: 0.81
xgb_model.train(os.path.join(dataset_folder, 'CAP_1.pkl'), model_output, parameters={
    'colsample_bytree': [1],
    'learning_rate': [0.1],
    'max_depth': [10],
    'n_estimators': [200],
    'subsample': [0.6]
})

# Best parameters found: {'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 200, 'subsample': 0.8}
# Accuracy: 0.8059, macro recall: 0.62, weight recall: 0.81
xgb_model.train(os.path.join(dataset_folder, 'CAP_2.pkl'), model_output, parameters={
    'colsample_bytree': [1],
    'learning_rate': [0.1],
    'max_depth': [10],
    'n_estimators': [200],
    'subsample': [0.8]
})