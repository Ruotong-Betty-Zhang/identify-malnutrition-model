import msoffcrypto
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import locale
from utils import calculate_age, drop_columns, average_fill_empty, calculate_malnutrition, drop_cap_nutrition_rows
from dotenv import load_dotenv
import os

target_folder = "./datasets/"
# Use password in .env
load_dotenv()
# Reading an encrypted Excel file
encrypted_file_path = os.getenv("CSV_PATH")
password = os.getenv("PASSWORD")

# 解密并读取
with open(encrypted_file_path, "rb") as f:
    office_file = msoffcrypto.OfficeFile(f)
    office_file.load_key(password=password)

    decrypted = io.BytesIO()
    office_file.decrypt(decrypted)

    # 读取解密后的 Excel 内容
    dfs = pd.read_excel(decrypted, sheet_name=None, engine="openpyxl")


print(dfs.keys()) 
basic_info_df = dfs["Staying Upright InterRAI"]
extra_info_df = dfs["Staying Upright demo2"]

# Preprocess the DataFrames
df = calculate_age(basic_info_df, extra_info_df)
df = drop_columns(df)

# Generate cap df
cap_df = drop_cap_nutrition_rows(df)
cap_df = average_fill_empty(cap_df)
cap_df = calculate_malnutrition(cap_df)
cap_df.to_pickle(target_folder + 'cap_data.pkl')
# df = pd.read_pickle(target_folder + 'cap_data.pkl')

# Generate mal df
mal_df = average_fill_empty(df)
mal_df = calculate_malnutrition(df)
mal_df = mal_df.drop(columns=["iK2a", "iK2g", "iG3", "iE2a", "iE2b", "iE2c", "iI1c", "iI1d", "CAP_Nutrition"])
mal_df.to_pickle(target_folder + 'mal_data.pkl')
# mal_df = pd.read_pickle(target_folder + 'mal_data.pkl')