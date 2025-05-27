# 🎯 Machine Learning

Esta carpeta contiene implementaciones y apuntes de algoritmos fundamentales de Machine Learning tradicional.

## 📁 Estructura de Contenidos

### 🔍 Aprendizaje Supervisado

#### Regresión
- **[📈 Linear Regression](linear_regression/)** - Regresión lineal simple
  - Conceptos básicos de regresión
  - Implementación desde cero y con scikit-learn
  - Análisis de residuos y métricas de evaluación

- **[📊 Multiple Linear Regression](multiple_linear_regression/)** - Regresión lineal múltiple
  - Regresión con múltiples variables
  - Selección de características
  - Multicolinealidad y regularización

- **[🔄 Polynomial Regression](polynomial_regression/)** - Regresión polinomial
  - Relaciones no lineales
  - Overfitting y underfitting
  - Validación cruzada

#### Clasificación
- **[🎲 Logistic Regression](logistic_regression/)** - Regresión logística
  - Clasificación binaria y multiclase
  - Función sigmoide y odds ratio
  - Regularización L1 y L2

- **[🔍 K-Nearest Neighbors](knn/)** - K-Vecinos más cercanos
  - Algoritmo basado en instancias
  - Selección del valor k óptimo
  - Métricas de distancia

- **[🧠 Perceptron](perceptron/)** - Perceptrón
  - Fundamentos de redes neuronales
  - Algoritmo de aprendizaje
  - Limitaciones y extensiones

### 🔍 Aprendizaje No Supervisado

- **[🎯 Clustering](clustering/)** - Algoritmos de agrupamiento
  - K-Means clustering
  - Evaluación de clusters
  - Selección del número óptimo de clusters

- **[📉 PCA](pca/)** - Análisis de Componentes Principales
  - Reducción de dimensionalidad
  - Varianza explicada
  - Visualización de datos de alta dimensión

- **[🔗 Hierarchical Clustering](hierarchical_clustering/)** - Clustering jerárquico
  - Dendrogramas
  - Linkage methods
  - Comparación con K-Means

### 🎮 Aprendizaje por Refuerzo

- **[🎮 Reinforcement Learning](reinforcement_learning/)** - Fundamentos de RL
  - Q-Learning básico
  - Política vs valor
  - Exploración vs explotación

## 🛠️ Herramientas Utilizadas

- **Python 3.8+**
- **NumPy** - Computación numérica
- **Pandas** - Manipulación de datos
- **Scikit-learn** - Algoritmos de ML
- **Matplotlib/Seaborn** - Visualización
- **Jupyter Notebooks** - Desarrollo interactivo

## 📚 Conceptos Clave Cubiertos

### Fundamentos
- Diferencia entre aprendizaje supervisado y no supervisado
- Overfitting vs Underfitting
- Bias-Variance tradeoff
- Validación cruzada

### Preprocesamiento
- Limpieza de datos
- Normalización y estandarización
- Codificación de variables categóricas
- Manejo de valores faltantes

### Evaluación de Modelos
- Métricas para regresión (MSE, MAE, R²)
- Métricas para clasificación (Accuracy, Precision, Recall, F1)
- Curvas ROC y AUC
- Matrices de confusión

### Optimización
- Gradient Descent
- Regularización (L1, L2)
- Selección de hiperparámetros
- Grid Search y Random Search

## 🎯 Objetivos de Aprendizaje

Al completar esta sección, deberías ser capaz de:

- [ ] Implementar algoritmos básicos de ML desde cero
- [ ] Usar scikit-learn para problemas reales
- [ ] Evaluar y comparar diferentes modelos
- [ ] Aplicar técnicas de preprocesamiento adecuadas
- [ ] Interpretar resultados y métricas
- [ ] Identificar y solucionar problemas comunes

## 📖 Recursos Recomendados

### Libros
- "Hands-On Machine Learning" - Aurélien Géron
- "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman
- "Pattern Recognition and Machine Learning" - Christopher Bishop

### Cursos
- Andrew Ng's Machine Learning Course (Coursera)
- CS229 Machine Learning (Stanford)

### Documentación
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## 🚀 Próximos Pasos

1. Completa los notebooks en orden secuencial
2. Practica con datasets reales de Kaggle
3. Implementa algoritmos desde cero para entender mejor
4. Avanza a Deep Learning una vez dominados estos conceptos

---

💡 **Tip**: La práctica constante es clave. Intenta aplicar cada algoritmo a diferentes tipos de problemas para entender sus fortalezas y limitaciones. 