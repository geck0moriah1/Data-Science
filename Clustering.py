#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
df = pd.read_csv("C:/Users/PC/MACHINE LEARNING/DATASETS/df_clientes.csv")
pd.set_option('display.max_columns', None)
df.head(20)


# In[2]:


df.shape


# In[2]:


# Seleccionar solo las columnas numéricas
columns = df.select_dtypes(include='number').columns

# Dividir las columnas en grupos de 3
groups = [columns[i:i+3] for i in range(0, len(columns), 3)]

# Crear boxplots para cada grupo de columnas
for i, group in enumerate(groups):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[group])
    plt.title(f"Boxplot for Group {i+1}: {', '.join(group)}")
    plt.xticks(rotation=45)  # Rotar las etiquetas del eje X para mayor claridad
    plt.tight_layout()  # Ajustar el diseño
    plt.show()


# In[6]:


# Función para detectar y eliminar outliers usando IQR
def eliminar_outliers_iqr(df, columna):
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_sin_outliers = df[(df[columna] >= lower_bound) & (df[columna] <= upper_bound)]
    return df_sin_outliers

# Crear DataFrame limpio sin outliers para todas las columnas cuantitativas
columnas_cuantitativas = ['numProductos', 'numCestas', 'numMarcas', 'numCategorias', 'numSegmentos', 'gastoTotal','gastoDesv','ofertas','sumFrescos','sumAlimentacion']


# Crear DataFrames limpios por cada columna
df_clean = df.copy()
for columna in columnas_cuantitativas:
    df_clean = eliminar_outliers_iqr(df_clean, columna)
    
# Gráficos: Histogramas antes y después para todas las columnas cuantitativas
fig, axes = plt.subplots(len(columnas_cuantitativas), 2, figsize=(12, 5 * len(columnas_cuantitativas)))
    
for i, columna in enumerate(columnas_cuantitativas):
    # Histograma antes de eliminar outliers
    axes[i, 0].hist(df[columna], bins=50, edgecolor='black')
    axes[i, 0].set_title(f'{columna} - Antes de eliminar outliers')
    axes[i, 0].set_xlabel(columna)
    axes[i, 0].set_ylabel('Frecuencia')

    # Histograma después de eliminar outliers
    axes[i, 1].hist(df_clean[columna], bins=50, edgecolor='black')
    axes[i, 1].set_title(f'{columna} - Después de eliminar outliers')
    axes[i, 1].set_xlabel(columna)
    axes[i, 1].set_ylabel('Frecuencia')

plt.tight_layout()
plt.show()


# In[8]:


from sklearn.preprocessing import StandardScaler

# Estandarizar las variables numéricas
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_clean[['numProductos', 'numCestas', 'numMarcas', 'numCategorias', 'numSegmentos', 'gastoTotal','gastoDesv','ofertas','sumFrescos','sumAlimentacion']])
# Verificar los datos escalados
pd.DataFrame(data_scaled)


# In[9]:


# Importar librerías necesarias
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Método del Codo y Silhouette Score
inercias = []  # Almacenar valores de inercia (método del codo)
silhouette_scores = []  # Almacenar valores de Silhouette Score

K = range(2, 11)  # Probar valores de k entre 2 y 10

# Iterar por cada número de clusters
for k in K:
    kmeans = KMeans(n_clusters=k).fit(data_scaled)
    inercias.append(kmeans.inertia_)  # Guardar la inercia
    silhouette_scores.append(silhouette_score(data_scaled, kmeans.labels_))  # Calcular Silhouette Score

# Gráficos de comparación
plt.figure(figsize=(12, 5))

# Gráfico del Método del Codo
plt.subplot(1, 2, 1)
plt.plot(K, inercias, 'o-', color='blue')
plt.title('Método del Codo')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.grid()

# Gráfico del Silhouette Score
plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'o-', color='green')
plt.title('Silhouette Score')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid()

# Mostrar los gráficos
plt.tight_layout()
plt.show()

# Mostrar los valores de inercia y silhouette score
print("Inercia:", inercias)
print("Silhouette Score:", silhouette_scores)

