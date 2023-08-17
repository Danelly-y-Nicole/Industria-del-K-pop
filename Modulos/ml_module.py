from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

def train_classification_model(data):
    X = data.drop('Gender_numeric', axis=1)
    y = data['Gender_numeric']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_classification_model(model, data):
    X = data.drop('Gender_numeric', axis=1)
    y = data['Gender_numeric']
    y_pred = model.predict(X)
    
    accuracy = accuracy_score(y, y_pred)
    return accuracy

def train_regression_model(data):
    X = data.drop('Height', axis=1)
    y = data['Height']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

def evaluate_regression_model(model, data):
    X = data.drop('Height', axis=1)
    y = data['Height']
    y_pred = model.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    return mse
