import pandas as pd
from datetime import datetime
import locale
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def calculate_age(basic_info, extra_info):
    """Add Age column to basic_info DataFrame based on extra_info DataFrame."""
	# Use the IDNo, Date1, Q4Age_BL columns to create a new DataFrame
    tempDf = extra_info[["IDNo", "Date1", "Q4Age_BL"]]
    locale.setlocale(locale.LC_TIME, 'C')

    # Go through each row in the DataFrame
    for index, row in tempDf.iterrows():
        # Convert the date string to a datetime object
        date_str = row["Date1"]
        date_obj = datetime.strptime(date_str, "%d%b%Y")

        # Extract the year and month
        year = date_obj.year
        month = date_obj.month
        age = row["Q4Age_BL"]
        if not age or age <= 0:
            basic_info.at[index, "Age"] = None
            continue

        seen = False
        for index2, row2 in basic_info.iterrows():
            # Check if the IDNo and month are the same
            if seen and row["IDNo"] != row2["IDno"]:
                break
            if row["IDNo"] == row2["IDno"]:
                date_str2 = row2["Assessment_Date"]
                date_obj2 = datetime.strptime(date_str2, "%d%b%Y")
                year2 = date_obj2.year
                month2 = date_obj2.month

                yearDiff = year2 - year
                monthDiff = month2 - month
                # Use the years and months to calculate age in df1
                age2 = age + yearDiff + monthDiff / 12
                # Add the age to the DataFrame
                basic_info.at[index2, "Age"] = age2
                seen = True

    # If age is below 0, set it to None
    basic_info.loc[basic_info["Age"] < 0, "Age"] = None

    # Print the age of the first 10 patients
    print(basic_info[["IDno", "Age"]].head(10))
    return basic_info

def generate_gender_column_and_carelevel(basic_info, extra_info):
    basic_info['Gender'] = extra_info['Q2Gender']
    mapping = {'Hospital': 3, 'Dementia Unit': 2, 'Rest Home': 1}
    basic_info['CareLevel'] = extra_info['CareLevel'].map(mapping)
    print(basic_info[['CareLevel', 'Gender']].head(10))
    return basic_info


def drop_columns(df):
    """Return a DataFrame with unnecessary columns dropped."""
    # Drop all cols after iJ1g
    columns_to_keep = ['iJ1g', 'iJ1h', 'iJ1i', 'iJ12']
    start_index = df.columns.get_loc('iJ1g')
    cols_from_start = df.columns[start_index:]
    cols_to_drop = [col for col in cols_from_start if col not in columns_to_keep]

    # drop iK4d: Dry mouth 
    # drop iNN2: Not in dictionary, and no data
    # drop iJ1(miss 63% of the data, 2190 data missing): Falls
    additional_cols_to_drop = ["iK4d", "iNN2", "iJ1"]
    cols_to_drop.extend(additional_cols_to_drop)
    
    df = df.drop(columns=cols_to_drop)
    return df

def average_fill_empty(df):
    """Fill NaN values in the DataFrame with the mean of each column."""
    # Fill NaN values with the mean of each column
    copied_df = df.copy()
    copied_df.fillna(copied_df.mean(numeric_only=True), inplace=True)
    return copied_df

def calculate_malnutrition(df):
    """Calculate malnutrition status based on specific columns and return a DataFrame with the new column."""
    df["Malnutrition"] = df.apply(get_malnutrition_status, axis=1)
    return df

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

def drop_cap_nutrition_rows(df):
    """Drop rows where CAP_Nutrition is NaN and return the DataFrame."""
    return df.dropna(subset=["CAP_Nutrition"])


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
def calculate_feature_changes(df, exclude_columns=['IDno', 'Assessment_Date', 'Malnutrition', 'CAP_Nutrition']):
    """
    For every patient with the same IDno, calculate the change in each features compare to the last assessment divided by the day passed\n
    :param df: DataFrame containing the patient data\n
    :param exclude_columns: List of columns to exclude from change calculation\n
    """
    changed_df = df.copy()
    changed_df['Assessment_Date'] = pd.to_datetime(changed_df['Assessment_Date'])

    # Create new columns for the change in each feature
    for column in changed_df.columns:
        if column not in exclude_columns and '_change' not in column:
            changed_df[f"{column}_change"] = np.nan  # Initialize with NaN

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
                if column not in ['IDno', 'Assessment_Date', 'Malnutrition'] and '_change' not in column:
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
