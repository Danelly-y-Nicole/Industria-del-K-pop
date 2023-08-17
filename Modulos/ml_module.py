from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def prepare_data(data):
    selected_columns = ['Height', 'Weight', 'age', 'Debut Age', 'year', 'month', 'day', 'Gender_numeric']
    data_selected = data[selected_columns]
    return data_selected

def impute_missing_values(data):
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data)
    return data_imputed

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def preprocess_data(data):
    # Elimina columnas no relevantes
    data = data.drop(['Stage Name', 'Full Name', 'Korean Name', 'K Stage Name', 'Gender'], axis=1)

    # Codificación de variables categóricas
    categorical_columns = ['Group', 'Company', 'Country', 'Second Country', 'Birthplace', 'Other Group', 'Former Group']
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_columns = pd.DataFrame(encoder.fit_transform(data[categorical_columns]))
    encoded_columns.columns = encoder.get_feature_names_out(categorical_columns)
    data = pd.concat([data.drop(categorical_columns, axis=1), encoded_columns], axis=1)

    # Transformación de fechas
    data['Date of Birth'] = pd.to_datetime(data['Date of Birth'])
    data['Debut'] = pd.to_datetime(data['Debut'])
    reference_date = pd.to_datetime('2023-01-01')
    data['Age_at_Debut'] = (data['Debut'] - data['Date of Birth']).dt.days
    data = data.drop(['Date of Birth', 'Debut'], axis=1)

    # Elimina filas con valores nulos
    data = data.dropna()

    return data

def train_model(X_train, y_train):
    # Entrena el modelo de RandomForestRegressor
    regression_model = RandomForestRegressor(random_state=42)
    regression_model.fit(X_train, y_train)
    return regression_model

def evaluate_model(model, X_test, y_test):
    # Evalúa el modelo
    mse = mean_squared_error(y_test, model.predict(X_test))
    return mse
