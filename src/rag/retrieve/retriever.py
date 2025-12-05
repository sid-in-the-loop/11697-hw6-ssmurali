from rag.embed.embedding import Embedding
from rag.db import DB
from config import RAGConfig, RetrieverType, RetrieverConfig

from typing import List

from typing import List, Sequence, Dict, Any
from rag.embed.embedding import Embedding, Chunk

import faiss
import json
import numpy as np
import re
import os

from rank_bm25 import BM25Okapi  # chosen sparse retrieval backend

import logging
# Avoid setting global logging configuration here to prevent test log pollution.
logger = logging.getLogger(__name__)

class Retriever():
    def __init__(self, config: RetrieverConfig):
        self.config: RetrieverConfig = config
        self.is_built = False
    
    def build(self, db) -> None:
        """Create internal structures (FAISS index, BM25 object, etc.) from DB chunks."""
        pass

    def retrieve(self, query: str, embedder, db) -> List[Chunk]:
        """Return top_k chunks for a query."""
        raise NotImplementedError()

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

class SparseRetriever(Retriever):
    """BM25-based sparse retriever using rank-bm25's BM25Okapi.

    build(): tokenizes all chunk strings and fits BM25.
    retrieve(): scores query tokens and returns top-k chunks.
    save()/load(): persist tokenized corpus + id ordering for fast reload.
    """

    _TOKEN_SPLIT_RE = re.compile(r"\w+")

    @staticmethod
    def _tokenize(text: str):
        return [t.lower() for t in SparseRetriever._TOKEN_SPLIT_RE.findall(text)]

    def __init__(self, config: RetrieverConfig):
        super().__init__(config)
        self.bm25 = None
        self.tokenized_corpus = None  # List[List[str]]
        self.id_order = None  # list of chunk ids aligned with tokenized_corpus rows

    def build(self, db: DB):
        if not db.chunks:
            raise ValueError("DB has no chunks; cannot build sparse retriever.")

        # Maintain deterministic order (insertion order of dict is fine, but make explicit)
        items = list(db.chunks.values())
        self.tokenized_corpus = [SparseRetriever._tokenize(c.string) for c in items]
        self.id_order = [c.id for c in items]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.is_built = True
        logger.info(f"Sparse BM25 index built over {len(self.tokenized_corpus)} chunks")

    def retrieve(self, query: str, embedder, db: DB) -> List[Chunk]:  # embedder unused but kept for interface parity
        if not self.is_built or self.bm25 is None:
            raise RuntimeError("SparseRetriever used before build().")
        assert self.id_order is not None
        tokens = SparseRetriever._tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)  # np.array shape (N,)
        k = self.config.topk
        assert k is not None
        # argsort descending
        top_idx = np.argsort(scores)[::-1][:k]
        results = []
        for i in top_idx:
            cid = self.id_order[i]
            results.append(db.get_chunk_by_id(cid))
        return results

    def save(self, path: str):
        if not self.is_built:
            raise RuntimeError("Cannot save sparse retriever before build().")
        os.makedirs(path, exist_ok=True)
        meta = {
            "id_order": self.id_order,
            "tokenized_corpus": self.tokenized_corpus,
            "type": "bm25-okapi"
        }
        with open(os.path.join(path, "sparse_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    def load(self, path: str):
        meta_path = os.path.join(path, "sparse_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No sparse_meta.json at {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.id_order = meta["id_order"]
        self.tokenized_corpus = meta["tokenized_corpus"]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.is_built = True
        logger.info(f"Sparse BM25 index loaded for {len(self.tokenized_corpus)} chunks")

class DenseRetriever(Retriever):
    def __init__(self, config: RetrieverConfig):
        super().__init__(config)
        self.index = None
        self.id_order = None  # list mapping row index -> chunk_id

    def build(self, db: DB):
        logger.info("Building faiss index from db")
        embeddings = db.embeddings_matrix()  # shape (N, dim)

        if self.config.normalize_embeddings:
            faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        factory_str = self.config.faiss_factory  # e.g., "Flat"
        self.index = faiss.index_factory(dim, factory_str)

        # If index type requires training (IVF, PQ, HNSW sometimes):
        if self.index.is_trained is False:
            self.index.train(embeddings)

        self.id_order = list(db.chunks.keys())
        self.index.add(embeddings)  # add all vectors

        logger.info("Moving faiss from cpu to gpu")

        # res = faiss.StandardGpuResources()
        # self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        self.is_built = True
        logger.info(f"Faiss index built of size {self.index.ntotal} vectors")

    def retrieve(self, query: str, embedder, db) -> list[Chunk]:
        assert self.is_built
        assert self.index is not None
        assert self.id_order is not None

        q_vec = embedder.embed(query).get_vector().astype('float32')
        if self.config.normalize_embeddings:
            faiss.normalize_L2(q_vec.reshape(1, -1))

        D, I = self.index.search(q_vec.reshape(1, -1), self.config.topk)
        # D: distances (if inner product + normalized → cosine similarity equivalent)
        # I: indices into self.id_order
        result_chunks = []
        for local_idx in I[0]:
            chunk_id = self.id_order[local_idx]
            result_chunks.append(db.get_chunk_by_id(chunk_id))
        return result_chunks

    def save(self, path: str):
        faiss.write_index(self.index, path + "/dense.faiss")
        # Save id_order
        with open(path + "/dense_meta.json", "w") as f:
            json.dump({"id_order": self.id_order}, f)

    def load(self, path: str):
        self.index = faiss.read_index(path + "/dense.faiss")
        with open(path + "/dense_meta.json") as f:
            meta = json.load(f)
        self.id_order = meta["id_order"]
        self.is_built = True


from .rrf import RRF

def retriever_factory(rag_config: RAGConfig):
    config = rag_config.retriever_config
    if config.retriever_type is RetrieverType.BASE:
        return Retriever(config)
    elif config.retriever_type is RetrieverType.SPARSE:
        return SparseRetriever(config)
    elif config.retriever_type is RetrieverType.DENSE:
        return DenseRetriever(config)
    elif config.retriever_type is RetrieverType.RRF:
        return RRF(config)
    else:
        raise TypeError(f"Weird retriever type: {config.retriever_type}")


