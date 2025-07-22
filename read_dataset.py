import msoffcrypto
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import locale
from utils import calculate_age, drop_columns, combine_malnutrition_labels, calculate_malnutrition, generate_gender_column_and_carelevel, check_missing_values, fill_missing_by_idno_and_mode, calculate_feature_changes, knn_impute_missing_values,restore_integer_columns
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
# # Check for missing values
# columns_to_check = ['iJ1g', 'iJ1h', 'iJ1i', 'iJ12', 'iK1ab', 'iK1bb']
# check_missing_values(columns_to_check, df)
df = drop_columns(df)
df = df.dropna(subset=["CAP_Nutrition"])
temp_df = df.copy(deep=True)
df = knn_impute_missing_values(df)
df = restore_integer_columns(df, original_df=temp_df)

# Generate cap df
cap_1 = df.copy(deep=True)
cap_1 = cap_1.drop(columns=["Scale_BMI"])
# cap_df = fill_missing_by_idno_and_mode(cap_df)
cap_1.to_pickle(os.path.join(dataset_folder, 'CAP_1.pkl'))

cap_2 = calculate_malnutrition(cap_1)
cap_2.to_pickle(os.path.join(dataset_folder, 'CAP_2.pkl'))

cap_long = calculate_feature_changes(cap_1)
cap_long.to_pickle(os.path.join(dataset_folder, 'CAP_L.pkl'))

# Generate mal df
mal_1 = calculate_malnutrition(df)
mal_1 = mal_1.drop(columns=["iK2a", "iK2g", "iG3", "iE2a", "iE2b", "iE2c", "iI1c", "iI1d", "CAP_Nutrition"])
# mal_df = fill_missing_by_idno_and_mode(mal_df)

mal_1.to_pickle(os.path.join(dataset_folder, 'MAL_1.pkl'))

mal_2 = combine_malnutrition_labels(mal_1)
mal_2.to_pickle(os.path.join(dataset_folder, 'MAL_2.pkl'))

mal_long = calculate_feature_changes(mal_2)
mal_long.to_pickle(os.path.join(dataset_folder, 'MAL_L.pkl'))

# # 生成df的年龄分布图
# plt.figure(figsize=(10, 6))
# df['Age'].hist(bins=30, edgecolor='black')
# plt.title('Age Distribution')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.grid(False)
# plt.savefig(os.path.join(dataset_folder, 'age_distribution.png'))
# plt.close()