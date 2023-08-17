# Industria-del-K-pop
Analisis de la Industria del K-pop.
# Predicción de Género y Estatura de Ídolos del K-pop

## Introducción

Este proyecto aborda el emocionante desafío de predecir el género y la estatura de los ídolos del K-pop mediante técnicas de programación en Python y modelos de aprendizaje automático. El K-pop, o pop coreano, es un género musical que ha ganado una gran popularidad en todo el mundo y ha dado lugar a numerosos grupos de ídolos con seguidores apasionados. Este proyecto busca proporcionar una comprensión más profunda de los atributos de estos ídolos, lo que podría tener aplicaciones en la industria del entretenimiento y el marketing.

## Problema a Resolver

El problema central que aborda este proyecto es la creación de modelos de aprendizaje automático capaces de predecir dos aspectos fundamentales de los ídolos del K-pop:

1. **Género:** La clasificación del género (masculino o femenino) de los ídolos basada en sus características personales y profesionales. Esto puede contribuir a una mejor comprensión de la distribución de géneros en la industria y ayudar en la toma de decisiones relacionadas con estrategias de promoción y marketing.

2. **Estatura:** La predicción de la estatura de los ídolos utilizando diversos atributos, como edad, lugar de nacimiento, compañía discográfica y más. Esto podría proporcionar información valiosa para la gestión de grupos y la planificación de presentaciones en vivo.

## Pasos Clave

### 1. Carga y Preprocesamiento de Datos

En el archivo `data_preprocessing.ipynb`, se lleva a cabo una serie de pasos para preparar los datos para su análisis y modelado. Los pasos incluyen:

- **Extracción de Datos:** Se carga el conjunto de datos original desde un archivo CSV utilizando la librería `pandas`.

- **Exploración y Limpieza de Datos:** Se realiza un análisis exploratorio inicial para comprender la estructura y características de los datos. Se identifican y manejan valores faltantes, y se eliminan filas duplicadas.

- **Manipulación de Fechas:** Se convierten las fechas de nacimiento y debut en formatos adecuados y se calculan edades y edades en el debut.

- **Codificación de Variables Categóricas:** Se codifican las variables categóricas relevantes para poder utilizarlas en los modelos.

- **Creación de Columnas Adicionales:** Se crean nuevas columnas derivadas, como el año, mes y día de nacimiento, para un análisis más detallado.

### 2. Clasificación de Género de los Ídolos

En el archivo `gender_classification.ipynb`, se aborda la tarea de clasificar el género de los ídolos utilizando un modelo de Regresión Logística. Los pasos clave incluyen:

- **Selección de Características:** Se eligen las características más relevantes, como altura, peso, edad y edad en el debut.

- **Imputación de Valores Faltantes:** Se utilizan técnicas de imputación para manejar los valores faltantes en el conjunto de datos.

- **División de Datos:** Se dividen los datos en conjuntos de entrenamiento y prueba.

- **Entrenamiento del Modelo:** Se entrena un modelo de Regresión Logística utilizando la librería `scikit-learn`.

- **Evaluación del Modelo:** Se evalúa el rendimiento del modelo en función de su precisión en la clasificación del género.

### 3. Predicción de Estatura de los Ídolos

En el archivo `height_prediction.ipynb`, se aborda la tarea de predecir la estatura de los ídolos utilizando un modelo de RandomForestRegressor. Los pasos principales son:

- **Preprocesamiento Adicional:** Se realizan más pasos de preprocesamiento, incluyendo la eliminación de columnas no relevantes y la codificación de variables categóricas.

- **Transformación de Fechas:** Se transforman las fechas de nacimiento y debut en características numéricas relevantes.

- **División de Datos:** Se dividen los datos en conjuntos de entrenamiento y prueba.

- **Entrenamiento del Modelo:** Se entrena un modelo de RandomForestRegressor utilizando la librería `scikit-learn`.

- **Evaluación del Modelo:** Se evalúa el rendimiento del modelo utilizando el Mean Squared Error (MSE) como métrica.

### 4. Visualización de Resultados

En los archivos `data_visualization.ipynb` y `height_visualization.ipynb`, se crean visualizaciones para representar los resultados y características de los ídolos del K-pop. Esto incluye:

- **Visualización de Distribuciones:** Se utilizan librerías como `matplotlib` y `seaborn` para crear gráficos que muestran las distribuciones de características como la altura y el género.

- **Visualización de Métricas de Evaluación:** Se crean gráficos que representan las métricas de evaluación de los modelos, como la precisión y el MSE.


## Enfoque de la Solución

El proyecto sigue un enfoque estructurado para abordar estos desafíos:

1. **Carga y Preprocesamiento de Datos:** Se inicia con la carga del conjunto de datos que contiene información detallada sobre los ídolos del K-pop. Luego, se realizan tareas de limpieza, transformación y manipulación de datos para prepararlos adecuadamente para el análisis y modelado.

2. **Clasificación de Género:** Se emplea un modelo de Regresión Logística para predecir el género de los ídolos basado en características específicas. Se utilizan métricas de evaluación para medir la precisión del modelo en la clasificación del género.

3. **Predicción de Estatura:** Se implementa un modelo de RandomForestRegressor para predecir la estatura de los ídolos, utilizando tanto atributos numéricos como categóricos. El rendimiento del modelo se evalúa mediante el cálculo del Mean Squared Error (MSE).

4. **Visualización de Resultados:** Se crean visualizaciones gráficas para presentar las distribuciones de características, métricas de evaluación y otros aspectos relevantes de los resultados obtenidos.

## Requisitos y Uso del Repositorio

Para ejecutar este proyecto, se requiere Python 3.x y las siguientes librerías de Python: pandas, scikit-learn, matplotlib, seaborn y plotly. El repositorio contiene una estructura organizada que incluye directorios para los datos, el código fuente y los cuadernos de Jupyter utilizados en cada etapa del proyecto.

## Uso del Repositorio

- `data`: Contiene el conjunto de datos original y el archivo preprocesado.
- `src`: Contiene los módulos de funciones utilizados en el proceso de preprocesamiento y modelado de datos.
- `notebooks`: Contiene el cuaderno utilizados¿ en el análisis exploratorio, la implementación de modelos y la visualización de resultados.
- `README.md`: El archivo que estás leyendo, que proporciona una descripción general del proyecto y su estructura.

## Resultados Esperados

Al finalizar este proyecto, se espera obtener modelos de aprendizaje automático que puedan predecir el género y la estatura de los ídolos del K-pop con una precisión y eficacia razonables. Estos modelos podrían proporcionar información valiosa para la industria del entretenimiento y contribuir a una mejor comprensión de los atributos que definen a los ídolos del K-pop.

## Autoras

[Nicole Peralta y Danelly Ureña]


## Contribuciones

Se agradecen las contribuciones y comentarios de la comunidad de programación y amantes del K-pop para mejorar y ampliar este proyecto.
