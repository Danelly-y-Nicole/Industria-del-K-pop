import pandas as pd
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

def extract_data(file_path):
    data = pd.read_csv(file_path)
    return data

def check_missing_ratio(data):
    data_na = (data.isnull().sum() / len(data)) * 100
    data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio' :data_na})
    return missing_data

def handle_duplicates(data):
    duplicate_rows_data = data[data.duplicated()]
    return duplicate_rows_data

def preprocess_dates(data):
    data['Debut'] = data['Debut'].replace('0/01/1900', pd.NA)
    data['Date of Birth'] = pd.to_datetime(data['Date of Birth'], format='%d/%m/%Y')
    data['Debut'] = pd.to_datetime(data['Debut'], format='%d/%m/%Y')

    data['age'] = (datetime.now() - data['Date of Birth']).astype('<m8[Y]')
    data['Debut Age'] = (data['Debut'] - data['Date of Birth']).astype('<m8[Y]')

    data['year'], data['month'], data['day'] = data['Date of Birth'].apply(lambda x:x.year), data['Date of Birth'].apply(lambda x:x.month), data['Date of Birth'].apply(lambda x:x.day)

    return data

def map_gender_to_numeric(data):
    gender_mapping = {'M': 1, 'F': 0}
    data['Gender_numeric'] = data['Gender'].map(gender_mapping)
    return data

def save_preprocessed_data(data, output_file):
    data.to_csv(output_file, index=False)
