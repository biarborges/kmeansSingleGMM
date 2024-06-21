# Comparison of Clustering Techniques in Text Documents in Portuguese

This project performs text clustering using BERT embeddings and various clustering algorithms in text documentos in portuguese. 
The code is divided into four Python files:

* text_processing.py: For text preprocessing and vectorization.
* kmeans.py: For clustering using K-means.
* singleLinkage.py: For clustering using Single Linkage.
* gmm.py: For clustering using Gaussian Mixture Model (GMM).

* corpus.csv: Contains the corpus used. It consists of 3600 news articles divided into 6 categories. The file contains the category and the text respectively.

* agglomerative_single_linkage_results.csv, gmm_results.csv, kmeans_results.csv: Contain the categories after clustering and the text for each algorithm.

* embeddings.npy: Contains the embeddings after processing with BERT.
