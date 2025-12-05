from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Mode(str, Enum):
    EMBED = 'embed'
    INFER = 'infer'


class EmbedderType(str, Enum):
    BASE = 'base'
    MINILM6 = 'minilm6'
    MINILM12 = 'minilm12'
    MXBAI = 'mxbai'
    AZURE = 'azure'  # API-based Azure embeddings


class RetrieverType(str, Enum):
    BASE = 'base'
    SPARSE = 'sparse'
    DENSE = 'dense'
    RRF = 'rrf'


@dataclass
class RetrieverConfig:
    retriever_type: RetrieverType
    normalize_embeddings: Optional[bool] = None
    faiss_factory: Optional[str] = None
    topk: Optional[int] = None
    rrf_subsystem_k: int = 100
    k_rrf: int = 60

    def __post_init__(self):
        assert self.faiss_factory in [None, "Flat"]
        if self.retriever_type is RetrieverType.RRF:
            assert self.rrf_subsystem_k is not None
            assert self.k_rrf is not None


@dataclass
class RAGConfig:
    embedder_type: EmbedderType
    retriever_config: RetrieverConfig
    db_path: str
    data_path: Optional[str] = None
    mode: Mode = Mode.INFER
    chunk_size: int = 1000
    chunk_overlap: int = 100
    rerank: bool = False
    rerank_model: Optional[str] = None
    rerank_shortlist: int = 50


@dataclass
class LLMConfig:
    model: str = "gpt-5-mini"
    key: str = ""

