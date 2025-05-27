# 📊 Scikit-learn Cheat Sheet

## 🔄 Flujo de Trabajo Típico

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Entrenar modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Predecir y evaluar
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
```

## 📈 Algoritmos de Regresión

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Regresión Lineal
lr = LinearRegression()
lr.fit(X_train, y_train)

# Ridge Regression (L2)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso Regression (L1)
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Support Vector Regression
svr = SVR(kernel='rbf', C=1.0)
svr.fit(X_train, y_train)
```

## 🎯 Algoritmos de Clasificación

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Regresión Logística
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Support Vector Machine
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
```

## 🔍 Clustering

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X)

# Hierarchical Clustering
agg = AgglomerativeClustering(n_clusters=3)
clusters = agg.fit_predict(X)

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)
clusters = gmm.fit_predict(X)
```

## 🔧 Preprocesamiento

```python
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, PolynomialFeatures
)
from sklearn.impute import SimpleImputer

# Escalado de características
scaler = StandardScaler()  # Media 0, desviación 1
scaler = MinMaxScaler()    # Rango [0, 1]
scaler = RobustScaler()    # Robusto a outliers

X_scaled = scaler.fit_transform(X_train)

# Imputación de valores faltantes
imputer = SimpleImputer(strategy='mean')  # 'median', 'most_frequent'
X_imputed = imputer.fit_transform(X)

# Codificación de variables categóricas
le = LabelEncoder()
y_encoded = le.fit_transform(y)

ohe = OneHotEncoder(sparse=False)
X_encoded = ohe.fit_transform(X_categorical)

# Características polinomiales
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

## 📊 Selección de Características

```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier

# Selección univariada
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Eliminación recursiva de características
estimator = RandomForestClassifier()
rfe = RFE(estimator, n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)

# Selección basada en modelo
selector = SelectFromModel(RandomForestClassifier())
X_selected = selector.fit_transform(X, y)
```

## 🔄 Validación de Modelos

```python
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, KFold
)

# Validación cruzada
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Random Search
from scipy.stats import randint
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 10)
}
random_search = RandomizedSearchCV(
    RandomForestClassifier(), param_dist, n_iter=20, cv=5
)
random_search.fit(X_train, y_train)
```

## 📏 Métricas de Evaluación

### Clasificación
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# Métricas básicas
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Reporte completo
report = classification_report(y_true, y_pred)
print(report)

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)

# ROC AUC
auc = roc_auc_score(y_true, y_pred_proba[:, 1])  # Para clasificación binaria
```

### Regresión
```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

# Métricas de regresión
mse = mean_squared_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

## 🔗 Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Pipeline simple
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Pipeline con transformaciones diferentes por columna
numeric_features = ['age', 'income']
categorical_features = ['gender', 'education']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])
```

## 📉 Reducción de Dimensionalidad

```python
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Truncated SVD (para matrices sparse)
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)
```

## 🎯 Consejos Rápidos

```python
# Guardar y cargar modelos
import joblib
joblib.dump(model, 'modelo.pkl')
model = joblib.load('modelo.pkl')

# Obtener nombres de características
feature_names = model.feature_names_in_  # sklearn >= 1.0

# Importancia de características (para modelos basados en árboles)
importances = model.feature_importances_

# Probabilidades de predicción
probas = model.predict_proba(X_test)

# Configurar random_state para reproducibilidad
model = RandomForestClassifier(random_state=42)
```

## 🚨 Errores Comunes

1. **No escalar datos antes de SVM/KNN**
2. **Aplicar fit_transform en test set** (usar solo transform)
3. **No usar random_state** para reproducibilidad
4. **Data leakage** en preprocesamiento
5. **No validar hiperparámetros** correctamente

---

💡 **Tip**: Siempre usa pipelines para evitar data leakage y hacer tu código más limpio y reproducible. 