import msoffcrypto
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import locale
from utils import calculate_age, drop_columns, average_fill_empty, calculate_malnutrition, drop_cap_nutrition_rows, generate_gender_column_and_carelevel, check_missing_values, fill_missing_by_idno_and_mode, calculate_feature_changes, knn_impute_missing_values,restore_integer_columns
from dotenv import load_dotenv
import os

# Reading an encrypted Excel file
encrypted_file_path = "./datasets/InterRAI Pt 4 2505.xlsx"
target_folder = "./datasets/"
# Use password in .env
load_dotenv()
# Reading an encrypted Excel file
# encrypted_file_path = os.getenv("CSV_PATH")
password = os.getenv("PASSWORD")


with open(encrypted_file_path, "rb") as f:
    office_file = msoffcrypto.OfficeFile(f)
    office_file.load_key(password=password)
    # Decrypt the file
    decrypted = io.BytesIO()
    office_file.decrypt(decrypted)

    dfs = pd.read_excel(decrypted, sheet_name=None, engine="openpyxl")


print(dfs.keys()) 
basic_info_df = dfs["Staying Upright InterRAI"]
extra_info_df = dfs["Staying Upright demo2"]

# Preprocess the DataFrames
df = calculate_age(basic_info_df, extra_info_df)
df = generate_gender_column_and_carelevel(basic_info_df, extra_info_df)
# # Check for missing values
# columns_to_check = ['iJ1g', 'iJ1h', 'iJ1i', 'iJ12', 'iK1ab', 'iK1bb']
# check_missing_values(columns_to_check, df)
df = drop_columns(df)

# Generate cap df
cap_df = drop_cap_nutrition_rows(df)
# cap_df = fill_missing_by_idno_and_mode(cap_df)
cap_df = knn_impute_missing_values(cap_df)
cap_df = restore_integer_columns(cap_df, original_df=df)
cap_df = calculate_malnutrition(cap_df)
cap_df.to_pickle(target_folder + 'cap_data.pkl')
# df = pd.read_pickle(target_folder + 'cap_data.pkl')

# Generate mal df
mal_df = calculate_malnutrition(df)
mal_df = df.drop(columns=["iK2a", "iK2g", "iG3", "iE2a", "iE2b", "iE2c", "iI1c", "iI1d", "CAP_Nutrition"])
# mal_df = fill_missing_by_idno_and_mode(mal_df)
mal_df = knn_impute_missing_values(mal_df)
mal_df = restore_integer_columns(mal_df,original_df=df)
mal_df.to_pickle(target_folder + 'mal_data.pkl')

# cap_df = pd.read_pickle(target_folder + 'cap_data.pkl')
# mal_df = pd.read_pickle(target_folder + 'mal_data.pkl')

# Generate longitudinal df
cap_long_df = calculate_feature_changes(cap_df)
cap_long_df.to_pickle(target_folder + 'cap_long_data.pkl')
mal_long_df = calculate_feature_changes(mal_df)
mal_long_df.to_pickle(target_folder + 'mal_long_data.pkl')