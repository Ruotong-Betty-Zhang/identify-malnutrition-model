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

# # Train the model on MAL_1 dataset
# # Train the model on MAL_1 dataset
# # Best parameters found:  {'ccp_alpha': np.float64(0.0), 'class_weight': None, 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
# rf_model.train(os.path.join(dataset_folder, 'MAL_1.pkl'), model_output, parameters={
#     'ccp_alpha': [0.0],
#     'class_weight': [None],
#     'max_depth': [None],
#     'max_features': ['log2'],
#     'min_samples_leaf': [1],
#     'min_samples_split': [2],
#     'n_estimators': [300]
# })

# # Train the model on MAL_2 dataset
# # Best parameters found:  {'ccp_alpha': np.float64(0.0), 'class_weight': 'balanced', 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}
# rf_model.train(os.path.join(dataset_folder, 'MAL_2.pkl'), model_output, parameters={
#     'ccp_alpha': [0.0],
#     'class_weight': ['balanced'],
#     'max_depth': [30],
#     'max_features': ['sqrt'],
#     'min_samples_leaf': [2],
#     'min_samples_split': [2],
#     'n_estimators': [200]
# })

# # Train the model on CAP_1 dataset
# # Best parameters found:  {'ccp_alpha': np.float64(0.0), 'class_weight': None, 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
# rf_model.train(os.path.join(dataset_folder, 'CAP_1.pkl'), model_output, parameters={
#     'ccp_alpha': [0.0],
#     'class_weight': [None],
#     'max_depth': [30],
#     'max_features': ['sqrt'],
#     'min_samples_leaf': [1],
#     'min_samples_split': [2],
#     'n_estimators': [100]
# })

# # Train the model on CAP_2 dataset
# # Best parameters found:  {'ccp_alpha': np.float64(0.0), 'class_weight': 'balanced', 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 100}
# rf_model.train(os.path.join(dataset_folder, 'CAP_2.pkl'), model_output, parameters={
#     'ccp_alpha': [0.0],
#     'class_weight': ['balanced'],
#     'max_depth': [30],
#     'max_features': ['sqrt'],
#     'min_samples_leaf': [2],
#     'min_samples_split': [10],
#     'n_estimators': [100]
# })

# # Train the model on MAL_L dataset
# # Best parameters found:  {'ccp_alpha': np.float64(0.0), 'class_weight': 'balanced', 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
# rf_model.train(os.path.join(dataset_folder, 'MAL_L.pkl'), model_output, parameters={
#     'ccp_alpha': [0.0],
#     'class_weight': ['balanced'],
#     'max_depth': [10],
#     'max_features': ['sqrt'],
#     'min_samples_leaf': [1],
#     'min_samples_split': [2],
#     'n_estimators': [100]
# })

# # Train the model on CAP_L dataset
# # Best parameters found:  {'ccp_alpha': np.float64(0.0), 'class_weight': None, 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
# rf_model.train(os.path.join(dataset_folder, 'CAP_L.pkl'), model_output, parameters={
#     'ccp_alpha': [0.0],
#     'class_weight': [None],
#     'max_depth': [30],
#     'max_features': ['sqrt'],
#     'min_samples_leaf': [1],
#     'min_samples_split': [2],
#     'n_estimators': [100]
# })

# Define the XGBoost model
xgb_model = xgb.XGBoostModelTrainer(seed=42)

# # Train the model on MAL_1 dataset
# # Best parameters found: {'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.8}
# # Accuracy: 0.8616, macro recall: 0.58, weight recall: 0.86
# # print("Training XGBoost model on MAL_1 dataset")
# # xgb_model.train(os.path.join(dataset_folder, 'MAL_1.pkl'), model_output)
# xgb_model.train(os.path.join(dataset_folder, 'MAL_1.pkl'), model_output)

# # Train the model on MAL_2 dataset
# # Best parameters found: {'colsample_bytree': 0.6, 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 100, 'subsample': 0.8}
# # Accuracy: 0.9705, macro recall: 0.71, weight recall: 0.97
# # print("Training XGBoost model on MAL_2 dataset")
# xgb_model.train(os.path.join(dataset_folder, 'MAL_2.pkl'), model_output)

# # Train the model on MAL_L dataset
# # Best parameters found: {'colsample_bytree': 0.6, 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 100, 'subsample': 0.8}
# # Accuracy: 0.9882, macro recall: 0.81, weight recall: 0.99
# # print("Training XGBoost model on MAL_L dataset")
# xgb_model.train(os.path.join(dataset_folder, 'MAL_L.pkl'), model_output)

# # Train the model on CAP_1 dataset
# # Best parameters found: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 200, 'subsample': 1}
# # Accuracy: 0.8085, macro recall: 0.64, weight recall: 0.81
# # print("Training XGBoost model on CAP_1 dataset")
# xgb_model.train(os.path.join(dataset_folder, 'CAP_1.pkl'), model_output)


# # # Train the model on CAP_2 dataset
# # Best parameters found: {'colsample_bytree': 0.6, 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 200, 'subsample': 1}
# # Accuracy: 0.8041, macro recall: 0.63, weight recall: 0.8
# # print("Training XGBoost model on CAP_2 dataset")
# xgb_model.train(os.path.join(dataset_folder, 'CAP_2.pkl'), model_output)

# # # Train the model on CAP_L dataset
# # print("Training XGBoost model on CAP_L dataset")
# # Best parameters found: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.6}
# # # Accuracy: 0.8188, macro recall: 0.62, weight recall: 0.82
# xgb_model.train(os.path.join(dataset_folder, 'CAP_L.pkl'), model_output)





# Train the model on MAL_2_AUG dataset
# Best parameters found: {'colsample_bytree': 0.6, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.6}
# xgb_model.train(os.path.join(dataset_folder, 'MAL_2_AUG.pkl'), model_output, test_on_original=True)

xgb_model.train(os.path.join(dataset_folder, 'CAP_2_AUG.pkl'), model_output, test_on_original=True)