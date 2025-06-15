# Generando datos reducidos de entrenamiento mediante técnicas inteligentes para modelos de deep learning

Este repositorio consiste en el Trabajo de Fin de Grado de **“Generando datos reducidos de entrenamiento mediante técnicas inteligentes para modelos de deep learnin”**. Aquí se encuentra el código y los resultados sobre la destilación de datos, cuantización y selección de ellas.

---

## 📖 Descripción

Los modelos de aprendizaje profundo requieren grandes volúmenes de datos y recursos de cómputo para entrenar con alta precisión en tareas de clasificación, segmentación y detección de objetos. Este proyecto propone un pipeline para:

1. **Destilación de datos**: generar ejemplares “sintéticos” o resumidos.  
2. **Cuantización**: reducir la precisión numérica de los datos y el espacio que ocupa.   
3. **Selección de muestras**: elegir subconjuntos adecuados para maximizar la representatividad y diversidad.

El objetivo es demostrar que es posible reducir drásticamente el tamaño del conjunto de entrenamiento sin perjudicar la precisión del modelo.

---

## 📂 Estructura del repositorio

```text
├── 01_Colab/                     # Experimentos realizados en google colab 
│   ├── MNIST/           
│   ├── FMNIST/           
|   └── CIFAR10/ 
│  
├── 02_Destilacion y Cuantizacion/ # Resultados de los experimentos sobre destilacion y cuantizacion  
│  
├── 03_Select lenet/               # Resultados de los experimentos sobre selección partiendo de una lenet
|
├── 03_Select eff/                 # Resultados de los experimentos sobre selección partiendo de una efficientnet
│  
├── requirements.txt               # Dependencias de Python  
├── README.md                      # Este fichero  
└── LICENSE                        # Licencia MIT  
