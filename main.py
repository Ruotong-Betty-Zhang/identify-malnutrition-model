import models.random_forest as rf
import models.xgboost as xgb
import models.xgboost_aug as xgb_aug
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)
dataset_folder = os.getenv("DATASET_FOLDER")
model_output = os.getenv("MODEL_OUTPUT")

# Define the Random Forest model
rf_model = rf.RandomForestModelTrainer(seed=42)

# Train the model on MAL_1 dataset
# Best parameters found:  {'ccp_alpha': 0.0, 'class_weight': 'balanced', 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 100}
rf_model.train(os.path.join(dataset_folder, 'MAL_1.pkl'), model_output)

# Train the model on MAL_2 dataset
# Best parameters found:  {'ccp_alpha': 0.025, 'class_weight': 'balanced', 'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 200}
rf_model.train(os.path.join(dataset_folder, 'MAL_2.pkl'), model_output)

# Train the model on CAP_1 dataset
# Best parameters found:  {'ccp_alpha': 0.0, 'class_weight': 'balanced', 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 300}
rf_model.train(os.path.join(dataset_folder, 'CAP_1.pkl'), model_output)

# Train the model on CAP_2 dataset
# Best parameters found:  {'ccp_alpha': 0.0, 'class_weight': 'balanced', 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 300}
rf_model.train(os.path.join(dataset_folder, 'CAP_2.pkl'), model_output)

# Train the model on MAL_L dataset
# Best parameters found:  {'ccp_alpha': 0.05, 'class_weight': 'balanced', 'max_depth': 5, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 100}
rf_model.train(os.path.join(dataset_folder, 'MAL_L.pkl'), model_output)

# Train the model on CAP_L dataset
# Best parameters found:  {'ccp_alpha': 0.0, 'class_weight': 'balanced', 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 300}
rf_model.train(os.path.join(dataset_folder, 'CAP_L.pkl'), model_output)



# Define the XGBoost model
xgb_model = xgb.XGBoostModelTrainer(seed=42)

# Train the model on MAL_1 dataset
# Best parameters found: {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 200, 'subsample': 0.6}
xgb_model.train(os.path.join(dataset_folder, 'MAL_1.pkl'), model_output)

# Train the model on MAL_2 dataset
# Best parameters found: {'colsample_bytree': 1.0, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 200, 'subsample': 1.0}
xgb_model.train(os.path.join(dataset_folder, 'MAL_2.pkl'), model_output)

# Train the model on MAL_L dataset
# Best parameters found: {'colsample_bytree': 1.0, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 50, 'subsample': 1.0}
xgb_model.train(os.path.join(dataset_folder, 'MAL_L.pkl'), model_output)

# Train the model on CAP_1 dataset
# Best parameters found: {'colsample_bytree': 0.6, 'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 200, 'subsample': 1.0}
xgb_model.train(os.path.join(dataset_folder, 'CAP_1.pkl'), model_output)

# Train the model on CAP_2 dataset
# Best parameters found: {'colsample_bytree': 0.6, 'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 50, 'subsample': 1.0}
xgb_model.train(os.path.join(dataset_folder, 'CAP_2.pkl'), model_output)

# Train the model on CAP_L dataset
# Best parameters found: {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'subsample': 0.8}
xgb_model.train(os.path.join(dataset_folder, 'CAP_L.pkl'), model_output)

# Augmentation
xgb_model_aug = xgb_aug.XGBoostModelTrainer()
# Best parameters found: {'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 50, 'subsample': 0.8}
xgb_model_aug.train(os.path.join(dataset_folder, 'MAL_2_AUG.pkl'), model_output, test_on_original=True)