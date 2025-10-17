# What is This Project About
This project aims to use machine learning tools to identify novel key predictors of malnutrition in elderly people in the New Zealand health care service using the InterRAI-LTCF dataset. This study is approved by the NZ Health and Disability Ethics Committee (HDEC) (Approval number 18/NTB/151).

This project uses the existing CAP_Nutrition from the InterRAI-LTCF dataset and the calculated Mini Nutrition Assessment scores as labels, trains separately on two major model structures, including Random Forest and XGBoost. A grid search on the major hyperparameters is done to automatically configure the models. The result models are evaluated through a confusion matrix to find the best performing ones. The top 12 most important features of all trained models and their importance scores are used to identify predictors of malnutrition.

A model evaluation and comparison interface is developed to provide an easy-to-use graphical interface for researchers who are not familiar with code and wish to compare the performance/feature importance of the two different models.

# Setup
## Python
This project runs on Python 3.12 - Python 3.13
The following code sets up a virtual environment for Python 3.12
```bash
py -3.12 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## .env
Create a `.env` file in the `/` directory with the following content
```
PASSWORD = "The password of the dataset CSV file"
CSV_PATH = "The path of the dataset CSV file"
DATASET_FOLDER = "The output folder where the processed dataset .pkl file will be stored"
MODEL_OUTPUT = "The output folder of trained models"
```

# How to Run
