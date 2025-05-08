# Comparison of Clustering Techniques in Text Documents in Portuguese

This project performs text clustering using BERT embeddings and various clustering algorithms in text documents in portuguese. 
The code is divided into four Python files:

* text_processing.py: For text preprocessing and vectorization.
* kmeans.py: For clustering using K-means.
* singleLinkage.py: For clustering using Single Linkage.
* gmm.py: For clustering using Gaussian Mixture Model (GMM).

* corpus.csv: Contains the corpus used. It consists of 3600 news articles divided into 6 categories. The file contains the category and the text respectively.

* agglomerative_single_linkage_results.csv, gmm_results.csv, kmeans_results.csv: Contain the categories after clustering and the text for each algorithm.

* embeddings.npy: Contains the embeddings after processing with BERT.

Citation:

@article{Borges_2025,  
title={Comparison of Clustering Techniques in Text Documents in Portuguese}, 

volume={18}, 

url={https://journals-sol.sbc.org.br/index.php/isys/article/view/5029}, 

DOI={10.5753/isys.2025.5029}, 

abstractNote={&amp;lt;p&amp;gt;Managing the vast amount of text data in the digital world is a complex challenge. An effective approach to tackle it is through the technique of text document clustering. This study evaluated the performance of three clustering algorithms — K-Means, Single Linkage, and Gaussian Mixture Model (GMM) — in clustering Brazilian Portuguese news articles using BERTimBau, a Portuguese variant of the BERT model, for preprocessing. Metrics such as accuracy, F1-score, Rand index, and Jaccard coefficient were used for evaluation. The results of these metrics indicated that Single Linkage achieved the best overall performance, surpassing K-Means and GMM in most of the evaluated criteria.&amp;lt;/p&amp;gt;}, 

number={1}, 

journal={iSys - Brazilian Journal of Information Systems}, 

author={Borges, Beatriz Ribeiro}, 

year={2025}, 

month={Mar.}, 

pages={4:1 – 4:17} }
