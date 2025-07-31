import pandas as pd
from datetime import datetime
import locale
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from pandas.api.types import is_numeric_dtype

def calculate_age(basic_info, extra_info):
    """
    Calculate and assign Age to basic_info based on Date of Birth (Q3DoB) or baseline age (Q4Age_BL).
    
    Priority:
    1. If Q3DoB is available, calculate age using Q3DoB and Assessment_Date.
    2. If Q3DoB is missing, estimate age using Q4Age_BL + time difference from baseline (Date1).
    """
    # Ensure date columns are in datetime format
    extra_info["Date1"] = pd.to_datetime(extra_info["Date1"], format="%d%b%Y", errors="coerce")
    extra_info["Q3DoB"] = pd.to_datetime(extra_info["Q3DoB"], format="%d%b%Y", errors="coerce")
    basic_info["Assessment_Date"] = pd.to_datetime(basic_info["Assessment_Date"], format="%d%b%Y", errors="coerce")

    # Merge extra info into basic_info
    merged_df = pd.merge(
        basic_info,
        extra_info[["IDNo", "Q3DoB", "Q4Age_BL", "Date1"]],
        how="left",
        left_on="IDno",
        right_on="IDNo"
    )

    # Age calculation
    def compute_age(row):
        if pd.notnull(row["Q3DoB"]):
            # If Date of Birth is available, use it directly
            delta = row["Assessment_Date"] - row["Q3DoB"]
            return round(delta.days / 365.25, 2)
        elif pd.notnull(row["Q4Age_BL"]) and pd.notnull(row["Date1"]):
            # If only baseline age is available, estimate using years from Date1
            delta = row["Assessment_Date"] - row["Date1"]
            return round(row["Q4Age_BL"] + delta.days / 365.25, 2)
        else:
            return np.nan

    merged_df["Age"] = merged_df.apply(compute_age, axis=1)

    # Drop helper columns if needed
    result_df = merged_df.drop(columns=["IDNo", "Q3DoB", "Q4Age_BL", "Date1"])

    # Drop all assessments where Age is NaN
    result_df = result_df.dropna(subset=["Age"])

    # Optional: Preview
    # print(result_df[["IDno", "Assessment_Date", "Age"]].head(10))

    return result_df
# def generate_gender_column_and_carelevel(basic_info, extra_info):
#     # For every IDno, find the gender in extra_info['Q2Gender'] and care level in extra_info['CareLevel']
#     # Initialize the Gender and CareLevel columns in basic_info
#     basic_info['Gender'] = None
#     basic_info['CareLevel'] = None
#     for index, row in basic_info.iterrows():
#         idno = row["IDno"]
#         # Find the row in extra_info with the same IDno
#         extra_row = extra_info[extra_info["IDNo"] == idno]
#         if not extra_row.empty:
#             # Set the gender and care level
#             basic_info.at[index, 'Gender'] = extra_row.iloc[0]['Q2Gender']
#             care_level = extra_row.iloc[0]['CareLevel']
#             mapping = {'Hospital': 3, 'Dementia Unit': 2, 'Rest Home': 1, 'RH': 1}
#             basic_info.at[index, 'CareLevel'] = mapping.get(care_level, None)

#     print(basic_info[['IDno','CareLevel', 'Gender']].head(10))
#     return basic_info

def generate_gender_column_and_carelevel(basic_info: pd.DataFrame, extra_info: pd.DataFrame) -> pd.DataFrame:
    # CareLevel 映射规则
    carelevel_map = {'Hospital': 3, 'Dementia Unit': 2, 'Rest Home': 1, 'RH': 1}

    # 选择需要的列并重命名
    extra_info_subset = extra_info[['IDNo', 'Q2Gender', 'CareLevel']].copy()
    extra_info_subset = extra_info_subset.rename(columns={
        'IDNo': 'IDno',
        'Q2Gender': 'Gender'
    })

    # 合并 basic_info 和 extra_info（按 IDno）
    merged = basic_info.merge(extra_info_subset, on='IDno', how='left')

    # 映射 CareLevel 为数字
    merged['CareLevel'] = merged['CareLevel'].map(carelevel_map)

    # 补全缺失值并转为整数（可选，也可以不填）
    merged['Gender'] = pd.to_numeric(merged['Gender'], errors='coerce').fillna(-1).astype(int)
    merged['CareLevel'] = merged['CareLevel'].fillna(-1).astype(int)

    print(merged[['IDno', 'Gender', 'CareLevel']].head(10))
    return merged

def drop_columns(df):
    """Return a DataFrame with unnecessary columns dropped."""
    result_df = df.copy(deep=True)
    # Drop all cols after iJ1g
    columns_to_keep = ['iJ1g', 'iJ1h', 'iJ1i', 'iJ12', 'iJ1', 'Age', 'Gender', 'CareLevel']
    start_index = result_df.columns.get_loc('iJ1g')
    cols_from_start = result_df.columns[start_index:]
    cols_to_drop = [col for col in cols_from_start if col not in columns_to_keep]

    # drop iK4d: Dry mouth 
    # drop iNN2: Not in dictionary, and no data
    # drop iJ1(miss 63% of the data, 2190 data missing): Falls
    additional_cols_to_drop = ["iK4d", "iNN2", "iJ1"]
    cols_to_drop.extend(additional_cols_to_drop)
    
    result_df = result_df.drop(columns=cols_to_drop)
    return result_df

def average_fill_empty(df):
    """Fill NaN values in the DataFrame with the mean of each column."""
    # Fill NaN values with the mean of each column
    copied_df = df.copy()
    copied_df.fillna(copied_df.mean(numeric_only=True), inplace=True)
    return copied_df

def calculate_malnutrition(df):
    """Calculate malnutrition status based on specific columns and return a DataFrame with the new column."""
    result_df = df.copy(deep=True)
    result_df["Malnutrition"] = result_df.apply(get_malnutrition_status, axis=1)
    return result_df

def get_malnutrition_status(row):
    if row["iK2a"] is None or row["iK2g"] is None or row["iG3"] is None or row["iE2a"] is None or row["iE2b"] is None or row["iE2c"] is None or row["iI1c"] is None:
        return -1
    score = 0
    mood = False
    dementia = False
    if row["iK2a"] == 1:
        score += 1
    if row["iK2g"] == 1:
        score += 1
    if row["iG3"] in [2, 3]:
        score += 1
    if row["iE2a"] in [3, 8]:
        score += 1
        mood = True
    if row["iE2b"] in [3, 8] and not mood:
        score += 1
        mood = True
    if row["iE2c"] in [3, 8] and not mood:
        score += 1
    if row["iI1c"] in [1, 2, 3]:
        score += 1
        dementia = True
    if row["iI1d"] in [1, 2, 3] and not dementia:
        score += 1
    return score

def check_missing_values(columns_to_check, df):
    """Check for missing values in the specified columns of the DataFrame."""
    missing_values = df[columns_to_check].isnull().sum()
    # Get the total number of rows in the DataFrame
    total_count = df[columns_to_check].shape[0]

    # Calculate the loss rate (percentage of missing values)
    loss_rate = (missing_values / total_count).round(4) * 100  # Convert to percentage with 2 decimal places

    # Create a DataFrame to display the results
    result_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Total Values': total_count,
        'Data Loss Rate (%)': loss_rate
    })

    # Display the results
    print(result_df)
    
def fill_missing_by_idno_and_mode(df, id_column='IDno'):
    """Fill missing values in specified columns by the mode of each IDno group."""
    columns = df.columns[df.isnull().any()].tolist()
    result_df = df.copy()
    for col in columns:
        global_mode = df[col].mode().iloc[0] if not df[col].mode().empty else None
        
        grouped = df.groupby(id_column)
        result_df[col] = grouped[col].transform(
        lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else global_mode)
    )
    return result_df

# For every patient with the same IDno, calculate the change in each features compare to the last assessment divided by the day passed
def calculate_feature_changes(df, exclude_columns=['IDno', 'Assessment_Date', 'Malnutrition', 'CAP_Nutrition'], include_columns=[
    'iK1ab',
    'iK1bb',
    "Scale_ADLHierarchy",
    "Scale_ADLLongForm",
    "Scale_ADLShortForm",
    "Scale_AggressiveBehaviour",
    "Scale_BMI",
    "Scale_CHESS",
    "Scale_Communication",
    "Scale_CPS",
    "Scale_DRS",
    "Scale_IADLCapacity",
    "Scale_IADLPerformance",
    "Scale_MAPLE",
    "Scale_Pain",
    "Scale_PressureUlcerRisk",
    "OverallUrgencyScale"
]):
    """
    For every patient with the same IDno, calculate the change in each features compare to the last assessment divided by the day passed\n
    :param df: DataFrame containing the patient data\n
    :param exclude_columns: List of columns to exclude from change calculation\n
    """
    changed_df = df.copy()
    changed_df['Assessment_Date'] = pd.to_datetime(changed_df['Assessment_Date'])

    # Create new columns for the change in each feature
    # 创建空的 DataFrame 来存储 change 列
    change_cols = [
        f"{col}_change" for col in changed_df.columns
        if col not in exclude_columns and '_change' not in col
    ]
    change_df = pd.DataFrame(np.nan, index=changed_df.index, columns=change_cols)

    # 一次性拼接，避免碎片化
    changed_df = pd.concat([changed_df, change_df], axis=1)

    # Go through each patient
    for patient_id in changed_df['IDno'].unique():
        # Find the assessment dates for the patient
        patient_data = changed_df[changed_df['IDno'] == patient_id]
        patient_data = patient_data.sort_values(by='Assessment_Date', ascending=False)

        # If there is only one assessment, remove and skip this patient
        if len(patient_data) < 2:
            changed_df = changed_df[changed_df['IDno'] != patient_id]
            continue

        # For each assessment, calculate the change in each feature compared to the last assessment
        for i in range(1, len(patient_data)):
            current_row = patient_data.iloc[i]
            previous_row = patient_data.iloc[i - 1]

            # Calculate the change in each feature
            for column in changed_df.columns:
                if column not in ['IDno', 'Assessment_Date', 'Malnutrition'] and '_change' not in column and (include_columns == None or column in include_columns):
                    column_name = f"{column}_change"

                    # date_diff = (current_row['Assessment_Date'] - previous_row['Assessment_Date']).days
                    # if date_diff != 0:
                    #     change = (current_row[column] - previous_row[column]) / date_diff
                    #     column_name = f"{column}_change"
                    #     changed_df.loc[(changed_df['IDno'] == patient_id) & (changed_df['Assessment_Date'] == current_row['Assessment_Date']), column_name] = change
                    
                    change = (current_row[column] - previous_row[column])
                    column_name = f"{column}_change"
                    changed_df.loc[(changed_df['IDno'] == patient_id) & (changed_df['Assessment_Date'] == current_row['Assessment_Date']), column_name] = change

        # Remove the first assessment for each patient, as it has no previous assessment to compare to
        changed_df = changed_df[changed_df['Assessment_Date'] != patient_data.iloc[0]['Assessment_Date']] 

        # Remove the empty columns that were created for change
        changed_df = changed_df.drop(columns=[col for col in changed_df.columns if '_change' in col and changed_df[col].isnull().all()])
    return changed_df

def knn_impute_missing_values(df, exclude_cols=['IDno', 'Assessment_Date'], n_neighbors=5):
    df = df.copy(deep=True)
    # Only fill iJ12 (Recent Falls) based on iJ1g (Falls - In last 30 days)
    print("Missing values in iJ12 before filling based on iJ1g:")
    print(df['iJ12'].isna().sum())
    mask = df['iJ12'].isna()  
    df.loc[mask & df['iJ1g'].isin([1, 2]), 'iJ12'] = 1
    df.loc[mask & df['iJ1g'].isin([0]), 'iJ12'] = 0
    print("Number of missing values in iJ12 after filling based on iJ1g:")
    print(df['iJ12'].isna().sum()) 


    df_copy = df.copy(deep=True)

    # Step 1: select columns to impute
    impute_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]

    # Step 2: standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_copy[impute_cols])

    # Step 3: fit KNN imputer and transform the data
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = imputer.fit_transform(scaled_data)

    # Step 4: inverse transform the imputed data and update the DataFrame
    imputed_data_unscaled = scaler.inverse_transform(imputed_data)
    df_copy[impute_cols] = imputed_data_unscaled

    print(f"✅ KNN imputation completed for {len(impute_cols)} columns with {n_neighbors} neighbors.")
    return df_copy

def knn_Classifier(df, exclude_cols=['IDno', 'Assessment_Date'], n_neighbors=5):
    df_copy = df.copy(deep=True)
    # Only fill iJ12 (Recent Falls) based on iJ1g (Falls - In last 30 days)
    print("Missing values in iJ12 before filling based on iJ1g:")
    print(df_copy['iJ12'].isna().sum())
    mask = df_copy['iJ12'].isna()  
    df_copy.loc[mask & df_copy['iJ1g'].isin([1, 2]), 'iJ12'] = 1
    df_copy.loc[mask & df_copy['iJ1g'].isin([0]), 'iJ12'] = 0
    print("Number of missing values in iJ12 after filling based on iJ1g:")
    print(df_copy['iJ12'].isna().sum()) 

    target_cols = [
        col for col in df_copy.columns
        if col not in exclude_cols 
        and is_numeric_dtype(df_copy[col]) 
        and df_copy[col].isna().sum() > 0
    ]

    print(f"🧩 Target columns to impute with KNN: {target_cols}")

    for target_col in target_cols:
        # Use only complete features (no NaNs) as input for KNN
        feature_cols = [
            col for col in df_copy.columns 
            if col not in exclude_cols + [target_col]
            and is_numeric_dtype(df_copy[col])
            and df_copy[col].isna().sum() == 0
        ]

        if not feature_cols:
            print(f"⚠️ No complete features to impute '{target_col}', skipping.")
            continue

        train_df = df_copy[df_copy[target_col].notna()]
        test_df = df_copy[df_copy[target_col].isna()]

        if test_df.empty:
            print(f"✅ No missing in '{target_col}', skipping.")
            continue

        # Fit KNN classifier
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(train_df[feature_cols], train_df[target_col].astype(int))

        predicted = knn.predict(test_df[feature_cols])
        df_copy.loc[test_df.index, target_col] = predicted

        print(f"✅ Filled {len(predicted)} values in '{target_col}' using KNN Classifier.")

    return df_copy

def restore_integer_columns(df, original_df=None, manual_cols=None):

    df_restored = df.copy()

    auto_int_cols = []
    if original_df is not None:
        for col in df.columns:
            if col in original_df.columns:
                orig_col = original_df[col]
                if pd.api.types.is_integer_dtype(orig_col) or (
                    pd.api.types.is_numeric_dtype(orig_col) and
                    orig_col.dropna().apply(float.is_integer).mean() > 0.95
                ):
                    auto_int_cols.append(col)

    target_cols = set(auto_int_cols)
    if manual_cols:
        target_cols.update(manual_cols)

    for col in target_cols:
        if col in df_restored.columns:
            df_restored[col] = df_restored[col].round().astype('Int64') 

    return df_restored

def generate_model_input_list(df, save_path=None):
    """Generate a list of columns in the DataFrame in form of json."""
    # Save the column names in a list
    column_list = df.columns.tolist()
    # Convert the list to a JSON string
    column_list_json = ', '.join(f'"{col}"' for col in column_list)

    # Save to a file
    if save_path:
        with open(save_path, 'w') as f:
            f.write(f'[{column_list_json}]')
        print(f"Column list saved to {save_path}")
    else:
        print("Column list not saved, no path provided.")
    # Return the JSON string
    return f'[{column_list_json}]'

def combine_malnutrition_labels(mal_df):
    temp_df = mal_df.copy()
    # Combine the malnutrition labels into binary values
    temp_df['Malnutrition'] = temp_df['Malnutrition'].apply(lambda x: 0 if x in [0, 1, 2] else 1)
    return temp_df
