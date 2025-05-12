👋 ¡Hola! Y bienvenido/a al repositorio del **Análisis de Rendimiento Estudiantil con PCA y SVM** 🚀

Este proyecto es una aplicación web hecha con Streamlit para echarle un ojo al rendimiento académico de estudiantes usando Componentes Principales (PCA) y Máquinas de Vectores de Soporte (SVM). La idea es ver si podemos predecir si un estudiante aprobará o no, ¡y de paso entender un poco de álgebra lineal! 😉

## ⚙️ ¿Qué puedes hacer con esto?

* **Elige los datos:** Puedes seleccionar si quieres analizar los datos de **Matemáticas** (`student-mat.csv`) o de **Portugués** (`student-por.csv`). ¡Tú decides! 📊
* **Selecciona qué analizar:** Marca las variables (características) que te interesan para el análisis PCA. Recuerda que **necesitas elegir al menos dos** para que funcione bien. 🤔
* **Reduce la complejidad con PCA:** PCA nos ayuda a simplificar los datos sin perder mucha información importante. Puedes ajustar cuántas "dimensiones" queremos conservar. ✨
* **Visualiza los datos:** Si eliges al menos dos componentes principales, verás un gráfico de cómo se ven los datos después de aplicar PCA. 🗺️
* **Descubre qué es importante:** Opcionalmente, puedes ver qué tanto influye cada variable original en estas nuevas "dimensiones" que crea PCA. 🔍
* **Predice con SVM:** Usamos un modelo de Machine Learning llamado SVM para intentar predecir si un estudiante aprobará o no. Puedes jugar con el tipo de "kernel" y un parámetro llamado "C". 🧠
* **¡Optimiza el modelo!** Si quieres, puedes activar una opción para que el software intente encontrar la mejor configuración para el modelo SVM automáticamente. 🏆
* **Mira los resultados:** Verás la precisión del modelo, un reporte con más detalles y una "matriz de confusión" para entender dónde acierta y dónde se equivoca el modelo. 📈
* **Aprende un poquito:** Hay una sección que explica de forma sencilla los conceptos de álgebra lineal que usamos en PCA y SVM. 🤓
* **Explora los datos:** Puedes activar opciones para ver estadísticas del dataset, cuántos valores únicos hay en cada columna y cómo se distribuyen las notas y las variables que elegiste. 🧐
* `app.py`: Aquí está todo el código de la magia de Streamlit. ✨
* `requirements.txt`: Este archivo le dice a tu computadora qué librerías de Python necesita para que esto funcione. 📦
* `student-mat.csv`: Los datos de rendimiento en la clase de Matemáticas. 🍎
* `student-por.csv`: Los datos de rendimiento en la clase de Portugués. 🇵🇹

## 🙏 ¡Gracias por los datos!

* Los datasets de `student-mat.csv` y `student-por.csv` son cortesía del UCI Machine Learning Repository ([https://archive.ics.uci.edu/dataset/320/student+performance](https://archive.ics.uci.edu/dataset/320/student+performance)). 


