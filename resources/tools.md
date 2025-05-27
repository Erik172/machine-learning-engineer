# 🛠️ Herramientas y Librerías

## 🐍 Python - Ecosistema Principal

### Manipulación de Datos
```python
import pandas as pd          # Manipulación de datos estructurados
import numpy as np           # Computación numérica
import polars as pl          # Alternativa rápida a pandas
import dask.dataframe as dd  # Procesamiento paralelo de datos grandes
```

### Visualización
```python
import matplotlib.pyplot as plt  # Gráficos básicos
import seaborn as sns           # Visualización estadística
import plotly.express as px     # Gráficos interactivos
import altair as alt            # Gramática de gráficos
import bokeh                    # Visualizaciones web interactivas
```

### Machine Learning Tradicional
```python
import sklearn                  # Scikit-learn - ML clásico
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb          # Gradient boosting
import lightgbm as lgb         # Gradient boosting rápido
import catboost as cb          # Gradient boosting para datos categóricos
```

### Deep Learning
```python
import tensorflow as tf        # TensorFlow
import keras                   # API de alto nivel
import torch                   # PyTorch
import torch.nn as nn
import transformers            # Hugging Face Transformers
import fastai                  # Fast.ai - DL simplificado
```

## 📊 Análisis Exploratorio de Datos (EDA)

### Herramientas Automáticas
- **pandas-profiling**: Reportes automáticos de EDA
- **sweetviz**: Comparación de datasets
- **autoviz**: Visualización automática
- **dataprep**: Preparación y exploración de datos

```python
# Ejemplo de uso
import pandas_profiling as pp
profile = pp.ProfileReport(df)
profile.to_file("reporte_eda.html")
```

### Detección de Anomalías
- **PyOD**: Outlier Detection
- **isolation-forest**: Detección de anomalías
- **DBSCAN**: Clustering para detectar outliers

## 🧠 Deep Learning Frameworks

### TensorFlow/Keras
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Modelo simple
model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### PyTorch
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Modelo simple
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)
```

### Herramientas Especializadas
- **Hugging Face Transformers**: Modelos preentrenados de NLP
- **Detectron2**: Computer Vision (Facebook)
- **MMDetection**: Detección de objetos
- **OpenMMLab**: Suite completa de CV

## 🗣️ Procesamiento de Lenguaje Natural

### Librerías Principales
```python
import nltk                    # Natural Language Toolkit
import spacy                   # NLP industrial
import transformers            # Modelos transformer
import gensim                  # Topic modeling y word embeddings
import textblob               # Análisis de sentimientos simple
```

### Herramientas Específicas
- **Hugging Face Hub**: Modelos preentrenados
- **spaCy**: Procesamiento rápido y preciso
- **NLTK**: Herramientas académicas completas
- **Gensim**: Word2Vec, Doc2Vec, LDA

## 👁️ Computer Vision

### OpenCV
```python
import cv2                     # Computer Vision básico
import PIL                     # Python Imaging Library
from skimage import filters    # Procesamiento de imágenes científico
```

### Deep Learning para CV
- **torchvision**: Modelos y datasets para PyTorch
- **tensorflow-datasets**: Datasets para TensorFlow
- **albumentations**: Augmentación de imágenes
- **imgaug**: Augmentación avanzada

## 📈 Optimización y AutoML

### Optimización de Hiperparámetros
```python
import optuna                  # Optimización bayesiana
import hyperopt               # Optimización de hiperparámetros
from sklearn.model_selection import GridSearchCV
```

### AutoML
- **Auto-sklearn**: AutoML para scikit-learn
- **TPOT**: Programación genética para ML
- **H2O.ai**: Plataforma AutoML
- **PyCaret**: ML de bajo código

## 🔄 MLOps y Producción

### Experiment Tracking
```python
import mlflow                  # Tracking de experimentos
import wandb                   # Weights & Biases
import neptune                 # Experiment management
```

### Deployment
- **FastAPI**: APIs rápidas para modelos
- **Flask**: Framework web ligero
- **Streamlit**: Apps web para ML
- **Gradio**: Interfaces rápidas para modelos

### Containerización
```dockerfile
# Docker para ML
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

## 🚀 Aceleración y Escalabilidad

### GPU Computing
```python
import cupy                    # NumPy en GPU
import cudf                    # Pandas en GPU (RAPIDS)
import torch.cuda             # PyTorch GPU
```

### Computación Distribuida
- **Dask**: Computación paralela en Python
- **Ray**: Computación distribuida
- **Spark**: Big Data processing
- **Horovod**: Entrenamiento distribuido de DL

## 📊 Bases de Datos y Big Data

### Conexiones a BD
```python
import sqlalchemy              # ORM para bases de datos
import pymongo                 # MongoDB
import psycopg2               # PostgreSQL
import sqlite3                # SQLite
```

### Big Data
- **Apache Spark**: Procesamiento distribuido
- **Hadoop**: Ecosistema de big data
- **Kafka**: Streaming de datos
- **Airflow**: Orquestación de workflows

## 🧪 Testing y Validación

### Testing de Modelos
```python
import pytest                  # Testing framework
import great_expectations     # Validación de datos
import deepchecks             # Testing de modelos ML
```

### Interpretabilidad
```python
import shap                    # SHAP values
import lime                    # Local interpretability
import eli5                    # Explain ML models
import interpret              # Interpretable ML
```

## 🔧 Utilidades Generales

### Configuración y Logging
```python
import hydra                   # Gestión de configuraciones
import logging                 # Logging estándar
import loguru                  # Logging mejorado
import click                   # CLI interfaces
```

### Procesamiento de Datos
```python
import joblib                  # Serialización de modelos
import pickle                  # Serialización Python
import h5py                    # Archivos HDF5
import zarr                    # Arrays comprimidos
```

## 📱 Herramientas de Desarrollo

### IDEs Recomendados
- **Jupyter Lab**: Notebooks avanzados
- **VS Code**: Editor con extensiones ML
- **PyCharm**: IDE completo para Python
- **Google Colab**: Notebooks en la nube con GPU

### Extensiones Útiles
- **Jupyter Extensions**: nbextensions
- **VS Code Python**: Extensión oficial
- **GitHub Copilot**: Asistente de código IA

## 🌐 APIs y Servicios Cloud

### Cloud ML Services
- **Google Cloud AI**: AutoML, Vertex AI
- **AWS SageMaker**: Plataforma ML completa
- **Azure ML**: Machine Learning en Azure
- **Databricks**: Plataforma de datos unificada

### APIs Útiles
- **OpenAI API**: GPT y otros modelos
- **Google Vision API**: Computer Vision
- **AWS Rekognition**: Análisis de imágenes
- **Hugging Face Inference API**: Modelos NLP

---

💡 **Tip**: Mantén un entorno virtual separado para cada proyecto y documenta las versiones de las librerías en `requirements.txt`. 