from rag.embed.embedder import embedder_factory
from rag.retrieve.retriever import retriever_factory
from rag.db import DB
from rag.embed.embedding import Chunk

from config import RAGConfig, Mode

from typing import List, Optional

import logging
from sentence_transformers import CrossEncoder
logger = logging.getLogger(__name__)

class RAG:
    def __init__(self, config: RAGConfig): 
        logger.info(f"Creating RAG")
        self.embedder = embedder_factory(config)
        self.retriever = retriever_factory(config)
        self.db = DB(config=config)
        self.cross_encoder: Optional[CrossEncoder] = None

        self.config = config


        if(config.mode == Mode.EMBED):
            assert config.data_path is not None, f"Must provide data_path for documents in embed mode"
            self.db.from_embedder(self.embedder, config.data_path)
            self.db.to_disk(db_path=config.db_path)
        elif(config.mode is Mode.INFER):
            self.db.from_disk(db_path=config.db_path)
            self.retriever.build(self.db)
            if config.rerank:
                assert config.rerank_model is not None
                model_name: str = config.rerank_model
                self.cross_encoder = CrossEncoder(model_name)
                logger.info(f"Loaded cross-encoder reranker '{model_name}'")
        else:
            raise TypeError("idk this rag mode")

    def retrieve(self, query: str) -> List[str]:
        chunks: List[Chunk] = self.retriever.retrieve(query=query, db=self.db, embedder=self.embedder)
        
        if not self.config.rerank:
            return [c.string for c in chunks]
        
        logger.info("Starting reranking")
        # If reranking enabled and cross-encoder loaded, first pull a shortlist (may temporarily upsize retriever topk)
        assert self.config.retriever_config.topk is not None
        shortlist_k = min(self.config.rerank_shortlist, self.config.retriever_config.topk)

        candidates = chunks[:shortlist_k]
        pair_texts = [(query, c.string) for c in candidates]

        assert self.cross_encoder is not None
        scores = self.cross_encoder.predict(pair_texts)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        reranked = [c for c, _ in ranked]

        logger.info("Rerank completed")
        return [c.string for c in reranked]

     


