import os
import json
from typing import List, Dict, Set, Optional

from config import RetrieverConfig, RetrieverType
from rag.db import DB
from rag.embed.embedding import Chunk
from .retriever import Retriever, SparseRetriever, DenseRetriever

import logging
logger = logging.getLogger(__name__)


class RRF(Retriever):
    """Reciprocal Rank Fusion (RRF) Hybrid Retriever.

    Strategy:
      1. Build both sparse (BM25) and dense (FAISS) indices.
      2. For a query, obtain top-k' results from each subsystem (k' usually > final topk).
      3. Fuse using RRF: score(d) = Σ 1/(k_rrf + rank_system(d)), rank is 1-based.
      4. Return top-k fused chunks.

    Notes:
      * Keeps internal SparseRetriever & DenseRetriever instances with their own configs.
      * Final returned list length <= self.config.topk
    """

    def __init__(self, config: RetrieverConfig):
        super().__init__(config)
        # internal retrievers
        self.sparse: Optional[SparseRetriever] = None
        self.dense: Optional[DenseRetriever] = None
        # component cutoffs (can be larger than final topk)
        self.k_subsystem = config.rrf_subsystem_k
        self.k_rrf = config.k_rrf

    # --- build ---
    def build(self, db: DB):
        dense_cfg = RetrieverConfig(
            retriever_type=RetrieverType.DENSE,
            normalize_embeddings=self.config.normalize_embeddings,
            faiss_factory=self.config.faiss_factory,
            topk=self.k_subsystem,
        )
        sparse_cfg = RetrieverConfig(
            retriever_type=RetrieverType.SPARSE,
            topk=self.k_subsystem,
            normalize_embeddings=None,
            faiss_factory=None,
        )
        self.sparse = SparseRetriever(sparse_cfg)
        self.dense = DenseRetriever(dense_cfg)
        self.sparse.build(db)
        self.dense.build(db)
        self.is_built = True
        logger.info(
            f"Hybrid RRF retriever built (sparse_k={self.k_subsystem}, dense_k={self.k_subsystem}, final_topk={self.config.topk})"
        )

    # --- retrieval ---
    def retrieve(self, query: str, embedder, db: DB) -> List[Chunk]:
        if not self.is_built:
            raise RuntimeError("Hybrid retriever used before build().")
        assert self.sparse and self.dense
        assert self.config.topk is not None, "Final topk must be set in RetrieverConfig for hybrid retriever"

        sparse_chunks: List[Chunk] = self.sparse.retrieve(query=query, embedder=embedder, db=db)
        dense_chunks: List[Chunk] = self.dense.retrieve(query=query, embedder=embedder, db=db)

        sparse_rank: Dict[int, int] = {c.id: i + 1 for i, c in enumerate(sparse_chunks)}
        dense_rank: Dict[int, int] = {c.id: i + 1 for i, c in enumerate(dense_chunks)}

        # pool unique chunk ids
        pool_ids: Set[int] = set(sparse_rank.keys()) | set(dense_rank.keys())

        # compute RRF scores
        scored: List[tuple[int, float]] = []  # (chunk_id, score)
        k_rrf = self.k_rrf
        for cid in pool_ids:
            score = 0.0
            if cid in sparse_rank:
                score += 1.0 / (k_rrf + sparse_rank[cid])
            if cid in dense_rank:
                score += 1.0 / (k_rrf + dense_rank[cid])
            scored.append((cid, score))

        # sort by score desc, tie-break deterministically by cid
        scored.sort(key=lambda x: (-x[1], x[0]))

        final_ids = [cid for cid, _ in scored[: self.config.topk]]
        # Preserve fused order
        fused_chunks = [db.get_chunk_by_id(cid) for cid in final_ids]
        return fused_chunks


