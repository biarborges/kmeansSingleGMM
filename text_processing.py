#text_processing.py

import pandas as pd
import numpy as np
import re
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm

def load_data():
    df = pd.read_csv('corpus.csv')
    return df['texto'].tolist(), df['categoria'].tolist()

def preprocess_text(texts):
    preprocessed_texts = []
    for text in texts:
        # Remover caracteres especiais, transformar para minúsculas e remover múltiplos espaços
        cleaned_text = re.sub(r'\s+', ' ', text.replace('*', '').lower()).strip()
        preprocessed_texts.append(cleaned_text)
    return preprocessed_texts

def vectorize_with_bertimbau(texts, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    model = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

    try:
        embeddings = np.load('embeddings.npy', allow_pickle=True)
        print("Embeddings carregados do arquivo.")
        return embeddings
    except FileNotFoundError:
        print("Arquivo de embeddings não encontrado. Vetorizando textos...")

    embeddings = []
    with torch.no_grad():
        for text in tqdm(texts, desc="Vetorizando textos"):
            inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].numpy()  # Pegar a representação do token CLS
            embeddings.append(embedding) 

    embeddings = np.stack(embeddings, axis=0)  # Empilhar ao longo do eixo 0 para formar uma matriz 2D

    np.save('embeddings.npy', embeddings)  # Salvar como matriz 2D
    print("Embeddings salvos em embeddings.npy.")

    return embeddings

if __name__ == "__main__":
    texts, categories = load_data()
    preprocessed_texts = preprocess_text(texts)
    embeddings = vectorize_with_bertimbau(preprocessed_texts)
