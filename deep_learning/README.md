# 🧠 Deep Learning

Esta carpeta contiene implementaciones y apuntes de algoritmos y arquitecturas de Deep Learning, desde fundamentos hasta técnicas avanzadas.

## 📁 Estructura de Contenidos

### ⚡ Fundamentos
- **[⚡ Gradient Descent](gradient_descent/)** - Optimización y backpropagation
  - Algoritmos de optimización (SGD, Adam, RMSprop)
  - Backpropagation paso a paso
  - Learning rate scheduling
  - Momentum y técnicas de aceleración

- **[🕸️ Neural Networks](neuronal_networks/)** - Redes neuronales básicas
  - Perceptrón multicapa (MLP)
  - Funciones de activación
  - Inicialización de pesos
  - Regularización (Dropout, Batch Normalization)

### 🖼️ Computer Vision
- **[🖼️ Convolutional Neural Networks](cnn/)** - CNNs fundamentales
  - Operaciones de convolución y pooling
  - Arquitecturas clásicas (LeNet, AlexNet, VGG)
  - Transfer Learning
  - Data Augmentation

- **[🏗️ Arquitecturas Avanzadas](../AlexNet.ipynb)** - Modelos específicos
  - AlexNet - Primera CNN exitosa en ImageNet
  - ResNet - Redes residuales y skip connections
  - Análisis de arquitecturas modernas

### 🗣️ Natural Language Processing
- **[📝 NLP Fundamentals](NLP/)** - Fundamentos de NLP con DL
  - Word embeddings (Word2Vec, GloVe)
  - Redes neuronales recurrentes (RNN, LSTM, GRU)
  - Attention mechanisms
  - Transformers básicos

- **[🤗 Hugging Face](../hugging-face/)** - Modelos preentrenados
  - BERT, GPT, T5 y otros transformers
  - Fine-tuning de modelos
  - Pipelines de NLP
  - Tokenización avanzada

## 🛠️ Frameworks y Herramientas

### Principales
- **TensorFlow/Keras** - Framework principal para prototipos rápidos
- **PyTorch** - Framework para investigación y producción
- **Hugging Face Transformers** - Modelos de NLP preentrenados

### Utilidades
- **TensorBoard** - Visualización de entrenamientos
- **Weights & Biases** - Experiment tracking
- **CUDA** - Aceleración GPU
- **Docker** - Containerización de modelos

## 📚 Conceptos Clave Cubiertos

### Arquitecturas Fundamentales
- **Feedforward Networks** - Redes completamente conectadas
- **Convolutional Networks** - Para datos con estructura espacial
- **Recurrent Networks** - Para datos secuenciales
- **Transformer Networks** - Attention-based models

### Técnicas de Optimización
- **Gradient Descent Variants** - SGD, Adam, AdaGrad, RMSprop
- **Learning Rate Scheduling** - Decay, warm-up, cyclic
- **Regularization** - Dropout, Batch Norm, Weight Decay
- **Advanced Optimizers** - AdamW, LAMB, RAdam

### Técnicas Avanzadas
- **Transfer Learning** - Aprovechamiento de modelos preentrenados
- **Fine-tuning** - Adaptación a tareas específicas
- **Multi-task Learning** - Entrenamiento conjunto
- **Meta-learning** - Aprender a aprender

### Evaluación y Debugging
- **Loss Functions** - Cross-entropy, MSE, custom losses
- **Metrics** - Accuracy, F1, BLEU, ROUGE
- **Visualization** - Feature maps, attention weights
- **Debugging** - Gradient flow, overfitting detection

## 🎯 Objetivos de Aprendizaje

Al completar esta sección, deberías ser capaz de:

- [ ] Implementar redes neuronales desde cero
- [ ] Usar TensorFlow/Keras y PyTorch efectivamente
- [ ] Diseñar arquitecturas para problemas específicos
- [ ] Aplicar transfer learning y fine-tuning
- [ ] Optimizar hiperparámetros y debugging
- [ ] Implementar modelos state-of-the-art
- [ ] Desplegar modelos en producción

## 📊 Datasets Utilizados

### Computer Vision
- **MNIST** - Dígitos escritos a mano
- **CIFAR-10/100** - Clasificación de objetos
- **ImageNet** - Dataset masivo de imágenes
- **COCO** - Detección y segmentación

### Natural Language Processing
- **IMDB** - Análisis de sentimientos
- **WikiText** - Modelado de lenguaje
- **SQuAD** - Question answering
- **GLUE** - Benchmark de NLP

## 🚀 Proyectos Prácticos

### Nivel Principiante
1. **Clasificador de dígitos MNIST** con MLP
2. **Clasificador CIFAR-10** con CNN básica
3. **Análisis de sentimientos** con RNN

### Nivel Intermedio
1. **Transfer learning** con modelos preentrenados
2. **Generador de texto** con LSTM
3. **Autoencoder** para reducción de dimensionalidad

### Nivel Avanzado
1. **Implementación de ResNet** desde cero
2. **Fine-tuning de BERT** para clasificación
3. **GAN** para generación de imágenes

## 📖 Recursos Recomendados

### Libros
- "Deep Learning" - Ian Goodfellow, Yoshua Bengio, Aaron Courville
- "Deep Learning with Python" - François Chollet
- "Hands-On Machine Learning" - Aurélien Géron

### Cursos
- CS231n: Convolutional Neural Networks (Stanford)
- CS224n: Natural Language Processing (Stanford)
- Fast.ai Practical Deep Learning for Coders

### Papers Fundamentales
- "ImageNet Classification with Deep CNNs" (AlexNet)
- "Deep Residual Learning for Image Recognition" (ResNet)
- "Attention Is All You Need" (Transformer)
- "BERT: Pre-training of Deep Bidirectional Transformers"

## 🔧 Setup y Configuración

### Instalación Básica
```bash
# TensorFlow
pip install tensorflow

# PyTorch
pip install torch torchvision torchaudio

# Hugging Face
pip install transformers datasets

# Utilidades
pip install wandb tensorboard matplotlib seaborn
```

### GPU Setup
```bash
# Verificar GPU disponible
python -c "import torch; print(torch.cuda.is_available())"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 🚨 Consideraciones Importantes

### Recursos Computacionales
- **GPU recomendada** para entrenamientos eficientes
- **Google Colab** para experimentación gratuita
- **Kaggle Kernels** como alternativa
- **Cloud platforms** para proyectos grandes

### Mejores Prácticas
- **Versionado de experimentos** con MLflow/W&B
- **Reproducibilidad** con seeds fijos
- **Monitoreo** de métricas durante entrenamiento
- **Checkpointing** para entrenamientos largos

## 🔄 Flujo de Trabajo Típico

1. **Exploración de datos** y análisis inicial
2. **Preprocesamiento** y augmentación
3. **Diseño de arquitectura** o selección de modelo
4. **Entrenamiento** con validación
5. **Evaluación** y análisis de resultados
6. **Fine-tuning** e iteración
7. **Deployment** y monitoreo

---

💡 **Tip**: El Deep Learning requiere mucha experimentación. Mantén un registro detallado de tus experimentos y no tengas miedo de iterar múltiples veces. 