#kmeans.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, rand_score, jaccard_score

# Carregar os embeddings do arquivo
embeddings = np.load('embeddings.npy')

# Verificar se os embeddings têm dimensão 3 e remover a dimensão extra se necessário
if embeddings.ndim == 3:
    embeddings = np.squeeze(embeddings, axis=1)

# Definir o número de clusters (por exemplo, 6)
num_clusters = 6

# Inicializar o modelo K-means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Ajustar o modelo aos embeddings
kmeans.fit(embeddings)

# Obter os rótulos dos clusters
labels = kmeans.labels_

# Carregar os textos originais com rótulos verdadeiros
df = pd.read_csv('corpus.csv')

# Mapear valores de texto para números inteiros
category_map = {category: idx for idx, category in enumerate(df['categoria'].unique())}
df['categoria'] = df['categoria'].map(category_map)

# Verificar o tipo dos rótulos verdadeiros e converter para números inteiros se necessário
true_labels = df['categoria']

# Adicionar os rótulos dos clusters ao DataFrame
df['categoria'] = labels

# Salvar os resultados em um novo arquivo CSV
df.to_csv('kmeans_results.csv', index=False)

# Calcular as métricas de avaliação
accuracy = accuracy_score(true_labels, labels)
f1 = f1_score(true_labels, labels, average='weighted')
rand_index = rand_score(true_labels, labels)
jaccard = jaccard_score(true_labels, labels, average='weighted')

# Imprimir as métricas de avaliação
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Rand Index: {rand_index:.4f}")
print(f"Jaccard Score: {jaccard:.4f}")
