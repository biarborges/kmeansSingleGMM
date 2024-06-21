#gmm.py

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, f1_score, rand_score, jaccard_score

embeddings = np.load('embeddings.npy')

# Verificar e remover a dimensão extra se necessário
if embeddings.ndim == 3:
    embeddings = np.squeeze(embeddings, axis=1)

num_components = 6  # Número de componentes para o GMM

gmm = GaussianMixture(n_components=num_components, random_state=42, init_params='random_from_data')

# Ajustar o modelo aos embeddings
gmm.fit(embeddings)

# Obter os rótulos dos clusters
labels = gmm.predict(embeddings)

df = pd.read_csv('corpus.csv')

# Mapear valores de texto para números inteiros
category_map = {category: idx for idx, category in enumerate(df['categoria'].unique())}
df['categoria'] = df['categoria'].map(category_map)

# Rótulos verdadeiros
true_labels = df['categoria']

# Adicionar os rótulos dos clusters ao DataFrame
df['categoria'] = labels

# Salvar os resultados em um novo arquivo CSV
df.to_csv('gmm_results.csv', index=False)

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
