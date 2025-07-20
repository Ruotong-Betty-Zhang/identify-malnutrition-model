import models.random_forest as rf
import models.xgboost as xgb
import os
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)
dataset_folder = os.getenv("DATASET_FOLDER")
model_output = os.getenv("MODEL_OUTPUT")

# # Define the model
# rf_model = rf.RandomForestModelTrainer(seed=42)

# # Train the model on MAL_1 dataset
# rf_model.train(os.path.join(dataset_folder, 'MAL_1.pkl'), model_output)

# Define the XGBoost model
xgb_model = xgb.XGBoostModelTrainer(seed=42)

# Train the model on MAL_1 dataset
xgb_model.train(os.path.join(dataset_folder, 'MAL_1.pkl'), model_output)
