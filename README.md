# Análisis de Rendimiento Estudiantil con PCA y Máquinas de Vectores de Soporte

Este proyecto es una aplicación web interactiva construida con Streamlit para analizar el rendimiento académico de estudiantes utilizando Componentes Principales (PCA) y Máquinas de Vectores de Soporte (SVM). Permite explorar datos de rendimiento en Matemáticas y Portugués, reducir la dimensionalidad de los datos, predecir si un estudiante aprobará o no, y comprender los conceptos de álgebra lineal subyacentes.

## Funcionalidades Principales

* **Selección de Fuente de Datos:** Permite al usuario elegir entre datasets de rendimiento estudiantil en Matemáticas (`student-mat.csv`) y Portugués (`student-por.csv`).
* **Selección de Características:** El usuario puede seleccionar las variables numéricas del dataset que se utilizarán para el análisis PCA. Se requiere la selección de al menos dos características.
* **Reducción de Dimensionalidad con PCA:**
    * Aplica PCA para reducir la cantidad de variables manteniendo la varianza importante.
    * Permite ajustar el número de componentes principales.
    * Visualiza los datos en las dos primeras componentes principales.
    * Opcionalmente, muestra la importancia de las características originales en los componentes principales.
* **Clasificación con SVM:**
    * Utiliza un modelo SVM para predecir si un estudiante aprobará o no (basado en un umbral configurable para la nota final G3).
    * Permite seleccionar el kernel del SVM y el parámetro de regularización C.
    * Ofrece la opción de optimizar automáticamente los hiperparámetros del SVM mediante búsqueda de cuadrícula.
* **Visualización de Resultados:** Muestra la precisión del modelo, el reporte de clasificación y la matriz de confusión.
* **Interpretación de PCA y SVM:** Proporciona explicaciones sobre los conceptos de álgebra lineal involucrados en PCA y SVM.
* **Información Adicional del Dataset:** Permite explorar estadísticas descriptivas, valores únicos y distribuciones de las variables.
* **Validación de Entrada:** Asegura que se seleccionen al menos dos características para el análisis PCA.

## Estructura del Proyecto
