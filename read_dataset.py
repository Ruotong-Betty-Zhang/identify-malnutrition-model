import msoffcrypto
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import locale
from utils import calculate_age, drop_columns, combine_malnutrition_labels, calculate_malnutrition, generate_gender_column_and_carelevel, check_missing_values, fill_missing_by_idno_and_mode, calculate_feature_changes, knn_impute_missing_values,restore_integer_columns,smote_augment
from dotenv import load_dotenv
import os

# Use password in .env
load_dotenv(dotenv_path=".env", override=True)
# Reading an encrypted Excel file
password = os.getenv("PASSWORD")
encrypted_file_path = os.getenv("CSV_PATH")
dataset_folder = os.getenv("DATASET_FOLDER")
print(f"Using dataset folder: {dataset_folder}")

# If the dataset folder doesn't exist, create it
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# Open the encrypted Excel file
with open(encrypted_file_path, "rb") as f:
    office_file = msoffcrypto.OfficeFile(f)
    office_file.load_key(password=password)
    # Decrypt the file
    decrypted = io.BytesIO()
    office_file.decrypt(decrypted)

    dfs = pd.read_excel(decrypted, sheet_name=None, engine="openpyxl")

basic_info_df = dfs["Staying Upright InterRAI"]
extra_info_df = dfs["Staying Upright demo2"]

# Preprocess the DataFrames
df = calculate_age(basic_info_df, extra_info_df)
df = generate_gender_column_and_carelevel(df, extra_info_df)
df = drop_columns(df)
df = df.dropna(subset=["CAP_Nutrition"])
temp_df = df.copy(deep=True)
df = knn_impute_missing_values(df)
df = restore_integer_columns(df, original_df=temp_df)

# Generate cap df
# CAP_1: Calculate CAP_Nutrition and drop Scale_BMI
cap_1 = df.copy(deep=True)
cap_1 = cap_1.drop(columns=["Scale_BMI"])
cap_1.to_pickle(os.path.join(dataset_folder, 'CAP_1.pkl'))

# CAP_2: Add malnutrition label from mini nutrition assessment
cap_2 = calculate_malnutrition(cap_1)
cap_2.to_pickle(os.path.join(dataset_folder, 'CAP_2.pkl'))

# CAP_L: Longitudinal dataset with feature changes
cap_long = calculate_feature_changes(cap_1)
cap_long.to_pickle(os.path.join(dataset_folder, 'CAP_L.pkl'))

# Generate mal df
# MAL_1: Calculate Malnutrition and drop nutrition related features
mal_1 = calculate_malnutrition(df)
mal_1 = mal_1.drop(columns=["iK2a", "iK2g", "iG3", "iE2a", "iE2b", "iE2c", "iI1c", "iI1d", "CAP_Nutrition"])
mal_1.to_pickle(os.path.join(dataset_folder, 'MAL_1.pkl'))

# MAL_2: Combine malnutrition labels, only 0 and 1
mal_2 = combine_malnutrition_labels(mal_1)
mal_2.to_pickle(os.path.join(dataset_folder, 'MAL_2.pkl'))

# MAL_L: Longitudinal dataset with feature changes
mal_long = calculate_feature_changes(mal_2)
mal_long.to_pickle(os.path.join(dataset_folder, 'MAL_L.pkl'))

# Data augmentation for MAL_2 using SMOTE
mal_2_aug = smote_augment(mal_2, target_col="Malnutrition", random_state=42)
mal_2_aug.to_pickle(os.path.join(dataset_folder, 'MAL_2_AUG.pkl'))