# What is This Project About


# Setup
## Python
This project runs on Python 3.12 - Python 3.13
The following code sets up a virtual environment of Python 3.12
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
