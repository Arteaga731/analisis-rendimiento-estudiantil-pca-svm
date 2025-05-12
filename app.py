# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="Análisis PCA + SVM Avanzado", layout="wide")

st.title("Análisis de Rendimiento Estudiantil con PCA y Máquinas de Vectores de Soporte")
st.markdown("Este análisis utiliza Componentes Principales (PCA) para reducir la dimensionalidad de los datos de rendimiento estudiantil y Máquinas de Vectores de Soporte (SVM) para predecir si un estudiante aprobará o no.")

# 1. Cargar datos
@st.cache_data
def load_data(data_source):
    if data_source == "Matemáticas":
        df = pd.read_csv("student-mat.csv", sep=";")
    elif data_source == "Portugués":
        df = pd.read_csv("student-por.csv", sep=";")
    return df.sample(n=150, random_state=42)

st.sidebar.subheader("Fuente de Datos")
data_source = st.sidebar.radio("Seleccionar dataset:", ["Matemáticas", "Portugués"])
df = load_data(data_source)

st.subheader(f"Dataset Cargado: {data_source}")
st.write(f"Tamaño del dataset: {df.shape[0]} filas, {df.shape[1]} columnas")
st.write("Primeras filas del dataset:", df.head())

# 2. Selección de características relevantes
st.sidebar.header("Configuración del Análisis")
all_features = df.columns.tolist()
categorical_features = [col for col in all_features if df[col].dtype == 'object']
numerical_features_for_select = [f for f in all_features if df[f].dtype != 'object' and f not in ['G3']]
default_features = [f for f in ['studytime', 'failures', 'absences', 'G1', 'G2'] if f in numerical_features_for_select]

selected_features = st.sidebar.multiselect("Seleccionar características para PCA:", numerical_features_for_select, default=default_features)
target_variable = "G3"
grade_threshold = st.sidebar.slider("Umbral para aprobar (G3):", min_value=0, max_value=20, value=10)
y = df[target_variable].apply(lambda x: 1 if x >= grade_threshold else 0)

if len(selected_features) < 2:
    st.error("TIENES QUE ELEGIR AL MENOS 2 CARACTERÍSTICAS PARA REALIZAR EL ANÁLISIS PCA.")
else:
    X = df[selected_features]

    # 3. Preprocesamiento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.subheader("Datos Escalares:")
    st.write("Los datos numéricos han sido escalados para tener media 0 y desviación estándar 1.")

    # 4. PCA
    n_components = st.sidebar.slider("Número de Componentes Principales:", min_value=1, max_value=X_scaled.shape[1], value=2)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    st.subheader(f"Reducción de Dimensionalidad con PCA (n_componentes={n_components})")
    st.write(f"Varianza explicada por los {n_components} componentes principales: {np.sum(pca.explained_variance_ratio_):.2f}")
    st.write("Los Componentes Principales son combinaciones lineales de las características originales, capturando la mayor variabilidad en los datos.")

    if n_components >= 2:
        fig_pca, ax_pca = plt.subplots()
        scatter = ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", alpha=0.7)
        ax_pca.set_xlabel("Componente Principal 1")
        ax_pca.set_ylabel("Componente Principal 2")
        ax_pca.set_title("Visualización de los Datos Tras PCA")
        legend = ax_pca.legend(*scatter.legend_elements(), title="Estado")
        ax_pca.add_artist(legend)
        st.pyplot(fig_pca)

    # Mostrar la importancia de las características en los componentes principales
    if st.sidebar.checkbox("Mostrar importancia de las características en PCA"):
        st.subheader("Importancia de las Características en los Componentes Principales")
        components_df = pd.DataFrame(pca.components_, columns=selected_features)
        for i in range(n_components):
            st.write(f"**Componente Principal {i+1}:**")
            importance = components_df.iloc[i].sort_values(ascending=False)
            st.write(importance)

    # 5. Clasificador SVM
    st.sidebar.subheader("Configuración del Modelo SVM")
    test_size = st.sidebar.slider("Tamaño del conjunto de prueba (%):", min_value=0.1, max_value=0.9, value=0.3)

    optimize_hyperparameters = st.sidebar.checkbox("Optimizar hiperparámetros del SVM")
    kernel = st.sidebar.selectbox("Kernel para SVM (si no se optimiza):", ["linear", "rbf", "poly"])
    C = st.sidebar.slider("Parámetro de Regularización C (si no se optimiza):", min_value=0.1, max_value=10.0, value=1.0)
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=test_size, random_state=random_state)

    if optimize_hyperparameters:
        st.subheader("Optimización de Hiperparámetros del SVM")
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}
        grid_search = GridSearchCV(SVC(random_state=random_state), param_grid, cv=2) # cv para validación cruzada
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        st.write(f"Mejores hiperparámetros encontrados: {best_params}")
        st.write(f"Precisión (validación cruzada) con los mejores hiperparámetros: {best_score:.2f}")
        model = grid_search.best_estimator_
    else:
        model = SVC(kernel=kernel, C=C, random_state=random_state)
        model.fit(X_train, y_train)

    # 6. Resultados del Modelo
    y_pred = model.predict(X_test)
    st.subheader("Resultados del Modelo de Máquinas de Vectores de Soporte (SVM)")
    st.write(f"Precisión del modelo: *{accuracy_score(y_test, y_pred):.2f}*")

    st.subheader("Reporte de Clasificación:")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Matriz de Confusión:")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Aprueba", "Aprueba"], yticklabels=["No Aprueba", "Aprueba"])
    ax_cm.set_xlabel("Predicción")
    ax_cm.set_ylabel("Real")
    st.pyplot(fig_cm)

    # 7. Interpretación de PCA y SVM (Aspectos de Álgebra Lineal)
    st.subheader("Aspectos de Álgebra Lineal en PCA y SVM")
    st.markdown(
        """
        **Componentes Principales (PCA):**
        - PCA se basa en la descomposición de la matriz de covarianza de los datos.
        - Busca los **autovectores** (direcciones principales) de esta matriz, que representan las direcciones de máxima varianza en los datos.
        - Los **autovalores** asociados a estos autovectores indican la cantidad de varianza explicada por cada componente principal.
        - La transformación de los datos originales a las nuevas componentes principales se realiza mediante una **proyección lineal** utilizando estos autovectores.

        **Máquinas de Vectores de Soporte (SVM):**
        - El objetivo de SVM es encontrar un **hiperplano** en el espacio de características que mejor separe las clases.
        - Para datos linealmente separables, esto implica resolver un problema de optimización para encontrar el hiperplano con el **máximo margen** (la distancia entre el hiperplano y los puntos más cercanos de cada clase, llamados vectores de soporte).
        - En el caso de kernels no lineales (como 'rbf' o 'poly'), los datos se transforman a un espacio de mayor dimensión donde se busca un hiperplano lineal. Esta transformación se realiza mediante funciones de **productos internos** (kernels) entre los vectores de datos.
        - La decisión de clasificación para un nuevo punto se basa en el signo de la función de decisión, que es una combinación lineal de los productos internos entre el nuevo punto y los vectores de soporte.
        """
    )

    # 8. Información Adicional del Dataset
    st.sidebar.subheader("Información del Dataset")
    if st.sidebar.checkbox("Mostrar información detallada del dataset"):
        st.write("Descripción estadística del dataset:")
        st.write(df.describe())

        st.write("Valores únicos por columna:")
        for col in df.columns:
            st.write(f"- **{col}:** {df[col].nunique()} valores únicos")

        if st.sidebar.checkbox("Mostrar distribución de la variable objetivo (G3)"):
            fig_hist_g3, ax_hist_g3 = plt.subplots()
            sns.histplot(df['G3'], bins=range(0, 21), kde=True, ax=ax_hist_g3)
            ax_hist_g3.set_title(f"Distribución de la Nota Final (G3) - {data_source}")
            ax_hist_g3.set_xlabel("Nota Final (G3)")
            ax_hist_g3.set_ylabel("Frecuencia")
            st.pyplot(fig_hist_g3)

        if st.sidebar.checkbox("Mostrar distribución de las características seleccionadas"):
            for feature in selected_features:
                fig_hist_feature, ax_hist_feature = plt.subplots()
                sns.histplot(df[feature], kde=True, ax=ax_hist_feature)
                ax_hist_feature.set_title(f"Distribución de {feature} - {data_source}")
                ax_hist_feature.set_xlabel(feature)
                ax_hist_feature.set_ylabel("Frecuencia")
                st.pyplot(fig_hist_feature)