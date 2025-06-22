# modules/rag_system/embedding_manager.py

import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import List

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.logger = logging.getLogger("EmbeddingManager")
        self.model = SentenceTransformer(model_name)
        self.logger.info(f"Embedding model loaded: {model_name}")

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        return self.batch_encode(texts)

    def batch_encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return np.array(embeddings, dtype='float32')

    def encode_single_text(self, text: str) -> np.ndarray:
        return np.array(self.model.encode(text), dtype='float32')

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.logger.info(f"FAISS index built with {embeddings.shape[0]} vectors")
        return index

    def save_index(self, index: faiss.Index, path: str) -> None:
        faiss.write_index(index, path)
        self.logger.info(f"Index saved to {path}")

    def load_index(self, path: str) -> faiss.Index:
        return faiss.read_index(path)

    def search_similar(self, index: faiss.Index, query_emb: np.ndarray, k: int =5):
        D, I = index.search(np.array([query_emb]), k)
        return I[0], D[0]
