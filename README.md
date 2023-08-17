# Industria-del-K-pop
Analisis de la Industria del K-pop.
# Predicción de Género y Estatura de Ídolos del K-pop

## Introducción

Este proyecto aborda el emocionante desafío de predecir el género y la estatura de los ídolos del K-pop mediante técnicas de programación en Python y modelos de aprendizaje automático. El K-pop, o pop coreano, es un género musical que ha ganado una gran popularidad en todo el mundo y ha dado lugar a numerosos grupos de ídolos con seguidores apasionados. Este proyecto busca proporcionar una comprensión más profunda de los atributos de estos ídolos, lo que podría tener aplicaciones en la industria del entretenimiento y el marketing.

## Problema a Resolver

El problema central que aborda este proyecto es la creación de modelos de aprendizaje automático capaces de predecir dos aspectos fundamentales de los ídolos del K-pop:

1. **Género:** La clasificación del género (masculino o femenino) de los ídolos basada en sus características personales y profesionales. Esto puede contribuir a una mejor comprensión de la distribución de géneros en la industria y ayudar en la toma de decisiones relacionadas con estrategias de promoción y marketing.

2. **Estatura:** La predicción de la estatura de los ídolos utilizando diversos atributos, como edad, lugar de nacimiento, compañía discográfica y más. Esto podría proporcionar información valiosa para la gestión de grupos y la planificación de presentaciones en vivo.

## Archivos

- `etl_module.py`: Este módulo contiene las funciones y procesos relacionados con la extracción, transformación y carga de datos.
- `ml_module.py`: Este módulo contiene funciones y procesos relacionados con la construcción y evaluación de modelos de Aprendizaje Automático.
- `main.py`: Este es el archivo principal desde el cual se ejecutan las tuberías ETL y ML del proyecto.
- `Notebook_de_exploración_K_pop.ipynb`: Este cuaderno de Jupyter contiene la exploración inicial de los datos y el análisis de la industria del K-pop.

## Pasos Clave

### 1. Extracción, Transformación y Carga (ETL)

En `etl_module.py`, se realizan los siguientes pasos:

- Extracción de datos desde fuentes externas.
- Limpieza y transformación de datos para su análisis y modelado.
- Generación de características adicionales para un análisis más profundo.

### 2. Modelado de Aprendizaje Automático (ML)

En `ml_module.py`, se llevan a cabo los siguientes pasos:

- Preparación de datos para el entrenamiento y evaluación de modelos.
- Construcción y entrenamiento de modelos de Aprendizaje Automático.
- Evaluación del rendimiento de los modelos utilizando métricas apropiadas.

### 3. Ejecución del Proyecto

El archivo `main.py` ejecuta las tuberías ETL y ML en secuencia, permitiendo la ejecución del proyecto completo.

### 3. Predicción de Estatura de los Ídolos

En el archivo `height_prediction.ipynb`, se aborda la tarea de predecir la estatura de los ídolos utilizando un modelo de RandomForestRegressor. Los pasos principales son:

- **Preprocesamiento Adicional:** Se realizan más pasos de preprocesamiento, incluyendo la eliminación de columnas no relevantes y la codificación de variables categóricas.

- **Transformación de Fechas:** Se transforman las fechas de nacimiento y debut en características numéricas relevantes.

- **División de Datos:** Se dividen los datos en conjuntos de entrenamiento y prueba.

- **Entrenamiento del Modelo:** Se entrena un modelo de RandomForestRegressor utilizando la librería `scikit-learn`.

- **Evaluación del Modelo:** Se evalúa el rendimiento del modelo utilizando el Mean Squared Error (MSE) como métrica.


## Enfoque de la Solución

El proyecto sigue un enfoque estructurado para abordar estos desafíos:

1. **Carga y Preprocesamiento de Datos:** Se inicia con la carga del conjunto de datos que contiene información detallada sobre los ídolos del K-pop. Luego, se realizan tareas de limpieza, transformación y manipulación de datos para prepararlos adecuadamente para el análisis y modelado.

2. **Clasificación de Género:** Se emplea un modelo de Regresión Logística para predecir el género de los ídolos basado en características específicas. Se utilizan métricas de evaluación para medir la precisión del modelo en la clasificación del género.

3. **Predicción de Estatura:** Se implementa un modelo de RandomForestRegressor para predecir la estatura de los ídolos, utilizando tanto atributos numéricos como categóricos. El rendimiento del modelo se evalúa mediante el cálculo del Mean Squared Error (MSE).


## Requisitos y Uso del Repositorio

Para ejecutar este proyecto, se requiere Python 3.x y las siguientes librerías de Python: pandas, scikit-learn, matplotlib, seaborn y plotly. El repositorio contiene una estructura organizada que incluye directorios para los datos, el código fuente y los cuadernos de Jupyter utilizados en cada etapa del proyecto.

## Uso del Repositorio

- `data`: Contiene el conjunto de datos original y el archivo preprocesado.
- `src`: Contiene los módulos de funciones utilizados en el proceso de preprocesamiento y modelado de datos.
- `notebook`: Contiene el cuaderno utilizados¿ en el análisis exploratorio, la implementación de modelos y la visualización de resultados.
- `README.md`: El archivo que estás leyendo, que proporciona una descripción general del proyecto y su estructura.

## Resultados Esperados

Al finalizar este proyecto, se espera obtener modelos de aprendizaje automático que puedan predecir el género y la estatura de los ídolos del K-pop con una precisión y eficacia razonables. Estos modelos podrían proporcionar información valiosa para la industria del entretenimiento y contribuir a una mejor comprensión de los atributos que definen a los ídolos del K-pop.

## Autoras

[Nicole Peralta y Danelly Ureña]


## Contribuciones

Se agradecen las contribuciones y comentarios de la comunidad de programación y amantes del K-pop para mejorar y ampliar este proyecto.
