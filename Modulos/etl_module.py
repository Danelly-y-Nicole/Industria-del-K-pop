import pandas as pd

def extract_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Example preprocessing: selecting columns and imputing missing values
    selected_columns = ['Height', 'Weight', 'age', 'Debut Age', 'year', 'month', 'day', 'Gender_numeric']
    data_selected = data[selected_columns]
    
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data_selected)
    
    preprocessed_data = pd.DataFrame(data_imputed, columns=selected_columns)
    return preprocessed_data
