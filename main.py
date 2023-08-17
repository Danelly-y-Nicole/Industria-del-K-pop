# Load data
file_path = 'kpopidolsv3.csv'
data = extract_data(file_path)

# Data preprocessing
missing_data = check_missing_ratio(data)
duplicate_rows_data = handle_duplicates(data)
data_preprocessed = preprocess_dates(data)

# Map Gender values to numeric values
data_preprocessed = map_gender_to_numeric(data_preprocessed)

# Save preprocessed data
output_file = 'kpopidols_preprocessed.csv'
save_preprocessed_data(data_preprocessed, output_file)

# Carga los datos
data = pd.read_csv('kpopidols_preprocessed.csv')

# Clasificación de Género de los Idols
# Prepara los datos
prepared_data = prepare_data(data)

# Manejo de valores nulos (imputación)
imputed_data = impute_missing_values(prepared_data)

# Divide los datos en características (X) y etiquetas (y)
X = imputed_data[:, :-1]
y = imputed_data[:, -1]

# Divide los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrena el modelo de regresión logística
classification_model = train_logistic_regression(X_train, y_train)

# Evalúa el modelo
accuracy = evaluate_model(classification_model, X_test, y_test)
print("Accuracy:", accuracy)

# Predicción de la Estatura de los Idols
data = preprocess_data(data)

# División de características (X) y etiquetas (y)
X = data.drop('Height', axis=1)
y = data['Height']

# División de datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = train_model(X_train, y_train)

mse = evaluate_model(model, X_test, y_test)
print("Mean Squared Error:", mse)
