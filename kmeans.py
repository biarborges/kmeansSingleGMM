#kmeans.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, rand_score, jaccard_score

embeddings = np.load('embeddings.npy')

if embeddings.ndim == 3:
    embeddings = np.squeeze(embeddings, axis=1)

num_clusters = 6

kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(embeddings)

labels = kmeans.labels_

df = pd.read_csv('corpus.csv')

category_map = {category: idx for idx, category in enumerate(df['categoria'].unique())}
df['categoria'] = df['categoria'].map(category_map)

true_labels = df['categoria']

df['categoria'] = labels

df.to_csv('kmeans_results.csv', index=False)

accuracy = accuracy_score(true_labels, labels)
f1 = f1_score(true_labels, labels, average='weighted')
rand_index = rand_score(true_labels, labels)
jaccard = jaccard_score(true_labels, labels, average='weighted')


print("K-Means")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Rand Index: {rand_index:.4f}")
print(f"Jaccard Score: {jaccard:.4f}")
