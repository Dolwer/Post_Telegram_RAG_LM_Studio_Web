# modules/rag_system/rag_retriever.py

import os
import json
import logging
from pathlib import Path
from time import time
from .rag_file_utils import FileProcessor
from .rag_chunk_tracker import ChunkTracker
from .embedding_manager import EmbeddingManager

class RAGRetriever:
    def __init__(self, config: dict):
        self.logger = logging.getLogger("RAGRetriever")
        self.file_processor = FileProcessor()
        self.embed_mgr = EmbeddingManager(config.get("embedding_model", "all-MiniLM-L6-v2"))
        self.chunk_tracker = ChunkTracker()
        self.config = config

        self.chunks = []
        self.chunk_ids = []
        self.embeddings = None
        self.index = None
        self.index_path = config.get("index_path", "faiss_index.idx")

    def process_inform_folder(self, folder_path: str):
        texts = []
        for root, _, files in os.walk(folder_path):
            for f in files:
                full = os.path.join(root, f)
                if self.file_processor.validate_file(full):
                    txt = self.file_processor.extract_text_from_file(full)
                    texts.append(txt)
        chunked = []
        for t in texts:
            chunked += self.chunk_text(t, self.config.get("chunk_size", 512))
        self.chunks = chunked
        self.chunk_ids = list(range(len(chunked)))

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = None):
        overlap = overlap or self.config.get("chunk_overlap", 50)
        tokens = text.split()
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk = " ".join(tokens[i:i+chunk_size])
            chunks.append(chunk)
        return chunks

    def build_knowledge_base(self):
        self.embeddings = self.embed_mgr.encode_texts(self.chunks)
        self.index = self.embed_mgr.build_faiss_index(self.embeddings)
        self.embed_mgr.save_index(self.index, self.index_path)
        self.chunk_tracker.reset_usage_stats()
        self.logger.info("RAG knowledge base initialized with usage tracker.")

    def retrieve_context(self, query: str, max_length: int = None) -> str:
        q_emb = self.embed_mgr.encode_single_text(query)
        ids, sims = self.embed_mgr.search_similar(self.index, q_emb, k=20)

        candidate_chunks = [(i, self.chunks[i]) for i in ids]
        diverse = self.chunk_tracker.get_diverse_chunks(candidate_chunks)
        selected = diverse[:self.config.get("max_context_chunks", 5)]

        for chunk_id, _ in selected:
            self.chunk_tracker.track_usage(chunk_id=chunk_id, topic=query)

        context = "\n\n".join([chunk for _, chunk in selected])
        self.logger.debug(f"Selected {len(selected)} diverse chunks for topic: {query}")
        return context

    def update_knowledge_base(self, new_content: str, source: str = None):
        new_chunks = self.chunk_text(new_content, self.config.get("chunk_size",512))
        new_embs = self.embed_mgr.encode_texts(new_chunks)
        self.chunks.extend(new_chunks)
        self.embeddings = np.vstack([self.embeddings, new_embs])
        self.index.add(new_embs)
        self.embed_mgr.save_index(self.index, self.index_path)

        start_id = len(self.chunks) - len(new_chunks)
        for i in range(len(new_chunks)):
            self.chunk_tracker.track_usage(chunk_id=start_id + i, topic=source or "update")

    def get_relevant_chunks(self, topic: str, limit: int = 10):
        return self.retrieve_context(topic)

    def rerank_results(self, query: str, candidates: list):
        return candidates

    def get_index_stats(self) -> dict:
        return {
            "total_chunks": len(self.chunks),
        }
