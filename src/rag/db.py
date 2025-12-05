from rag.embed.embedding import Chunk, Embedding
from rag.embed.embedder import Embedder

from config import RAGConfig, EmbedderType
import json, os, hashlib
import numpy as np
from typing import Dict


import logging
logger = logging.getLogger(__name__)

def _sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class DB:
    MANIFEST_NAME = "manifest.json"
    META_NAME = "meta.json"
    EMB_NAME = "embeddings.npy"
    VERSION = 1

    def __init__(self, config: RAGConfig):
        self.config: RAGConfig = config
        self.chunks: Dict[int, Chunk] = {}

    def to_disk(self, db_path: str):
        logger.info(f"Saving DB to disk path {db_path}")
        _ensure_dir(db_path)

        # 1) Freeze a deterministic order (sorted by chunk.id)
        items = sorted(self.chunks.values(), key=lambda c: c.id)

        # 2) Build meta and embeddings aligned to that order
        meta = []
        embs = []
        id_to_row = {}

        for row_idx, c in enumerate(items):
            vec = c.embedding.get_vector()
            if vec is None:
                raise ValueError(f"Chunk {c.id} has no embedding vector.")
            # force float32 and 1D
            vec = np.asarray(vec, dtype=np.float32).reshape(-1)
            embs.append(vec)
            meta.append({"id": c.id, "doc_id": c.doc_id, "string": c.string})
            id_to_row[c.id] = row_idx

        if not embs:
            # nothing to save; write empty files and a minimal manifest
            emb_mat = np.zeros((0, 0), dtype=np.float32)
        else:
            # Verify all dims match
            dim = embs[0].shape[0]
            for i, e in enumerate(embs):
                if e.shape[0] != dim:
                    raise ValueError(f"Embedding dimension mismatch at row {i}: {e.shape[0]} vs {dim}")
            emb_mat = np.vstack(embs)  # (N, D), float32

        # 3) Write meta + embeddings
        meta_path = os.path.join(db_path, self.META_NAME)
        emb_path  = os.path.join(db_path, self.EMB_NAME)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        np.save(emb_path, emb_mat)

        # 4) Compute checksums
        meta_sha256 = _sha256_file(meta_path)
        emb_sha256  = _sha256_file(emb_path)

        # 5) Write manifest with verification info & id_to_row mapping
        manifest = {
            "version": self.VERSION,
            "count": int(emb_mat.shape[0]),
            "dim": int(emb_mat.shape[1] if emb_mat.ndim == 2 else 0),
            "dtype": str(emb_mat.dtype),
            "config": str(self.config),
            "embedder_type": self.config.embedder_type.value, 
            "files": {
                "meta": {"name": self.META_NAME, "sha256": meta_sha256, "size": os.path.getsize(meta_path)},
                "embeddings": {"name": self.EMB_NAME, "sha256": emb_sha256, "size": os.path.getsize(emb_path)},
            },
            # explicit alignment info to avoid relying on file order
            "id_to_row": id_to_row,   # {chunk_id: row_index}
        }
        with open(os.path.join(db_path, self.MANIFEST_NAME), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def from_disk(self, db_path: str):
        logger.info(f"Loading DB from disk path {db_path}")
        manifest_path = os.path.join(db_path, self.MANIFEST_NAME)
        meta_path = os.path.join(db_path, self.META_NAME)
        emb_path  = os.path.join(db_path, self.EMB_NAME)

        # 1) Read manifest first (so we know what to expect)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing {self.MANIFEST_NAME} in {db_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    
        # 2) Verify files exist and checksums match
        for key, info in manifest.get("files", {}).items():
            p = os.path.join(db_path, info["name"])
            if not os.path.exists(p):
                raise FileNotFoundError(f"Manifest references missing file: {p}")
            actual_sha = _sha256_file(p)
            if actual_sha != info.get("sha256"):
                raise ValueError(f"Checksum mismatch for {info['name']}: expected {info.get('sha256')}, got {actual_sha}")

        # 2b) Validate embedder type consistency.
        saved_embedder_type = manifest.get("embedder_type")
        current = self.config.embedder_type.value
        if saved_embedder_type != current:
            raise ValueError(
                "Embedder type mismatch between manifest and current configuration: "
                f"saved='{saved_embedder_type}' current='{current}'. "
                "Refuse to load to prevent dimension/semantic inconsistencies."
            )
        else:
            logger.info(f"Embedder type validated: {current}")

        # 3) Load files
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        emb = np.load(emb_path)

        # 4) Basic schema checks
        expected_n = manifest.get("count")
        expected_d = manifest.get("dim")
        expected_dtype = manifest.get("dtype")

        if emb.ndim != 2:
            raise ValueError(f"Embeddings should be 2D (N, D), got shape {emb.shape}")
        n, d = emb.shape

        if n != expected_n or d != expected_d or str(emb.dtype) != expected_dtype:
            raise ValueError(
                f"Emb matrix schema mismatch. "
                f"Got (n={n}, d={d}, dtype={emb.dtype}); "
                f"expected (n={expected_n}, d={expected_d}, dtype={expected_dtype})."
            )
        if len(meta) != n:
            raise ValueError(f"Meta length ({len(meta)}) != embeddings rows ({n})")

        # 5) Use id_to_row mapping to align (no reliance on file order)
        id_to_row = manifest.get("id_to_row", None)
        if id_to_row is None:
            # Fallback: try to reconstruct by the saved order in meta (still safe, but warn)
            # (We include a conservative check that meta IDs are unique)
            print("No id_to_row in manifest, fallign back to sorted order")
            seen = set()
            ids = []
            for m in meta:
                cid = m["id"]
                if cid in seen:
                    raise ValueError(f"Duplicate id {cid} in meta without id_to_row mapping.")
                seen.add(cid)
                ids.append(cid)
            id_to_row = {cid: i for i, cid in enumerate(ids)}
        else:
            id_to_row = {int(k): int(v) for k, v in id_to_row.items()}

        # 6) Rebuild chunks using the verified mapping
        self.chunks = {}
        for m in meta:
            cid = m["id"]
            if cid not in id_to_row:
                raise ValueError(f"Chunk id {cid} not present in id_to_row mapping.")
            row = id_to_row[cid]
            if row < 0 or row >= n:
                raise ValueError(f"Row index {row} for id {cid} out of bounds for embeddings with n={n}.")
            vec = emb[row]
            chunk = Chunk(
                id=cid,
                doc_id=m["doc_id"],
                string=m["string"],
                embedding=Embedding(vec.astype(np.float32, copy=False)),
            )
            self.add(chunk)

        logger.info(f"Loading DB completed, total chunks - {len(self.chunks)}")

    def add(self, chunk: Chunk):
        assert chunk.id not in self.chunks, f"Same chunk id already exists - {self.chunks[chunk.id]}, {chunk}"
        self.chunks[chunk.id] = chunk

    def remove(self, chunk: Chunk):
        assert chunk.id in self.chunks, f"Cant find chunk with id - {chunk}"
        assert chunk.doc_id == self.chunks[chunk.id].doc_id
        del self.chunks[chunk.id]

    def get_chunk_by_id(self, chunk_id: int) -> Chunk:
        return self.chunks[chunk_id]

    def embeddings_matrix(self) -> np.ndarray:
        return np.vstack([c.embedding.get_vector() for _,c in self.chunks.items()]).astype('float32')
        

    def fixed_size_chunking(self, content, chunk_size, overlap_size):
        """Splits document content into fixed-size chunks with overlap based on character count."""
        chunks = []
        start = 0
        acceptable_starts = ['.', ' ', '\n', '-', ';']
        while start < len(content):
            while content[start] not in acceptable_starts:
                start -= 1
                if start < 0:
                    start = max(0, start)
                    break
            end = start + chunk_size
            chunks.append(content[start:end])
            start = end - overlap_size
            start = max(0, start)
            if start >= len(content):
                break
        return chunks

    def from_embedder(self, embedder: Embedder, data_path: str):
        """Process JSON documents into chunks and store with embeddings"""
        logger.info(f"Embedding started")
        
        logger.info(f"Loading raw documents from: {data_path}")
        with open(data_path, 'r') as f:
            data = json.load(f)

        # Handle different JSON formats
        if isinstance(data, list):
            # Format: [{"data": {"id": "...", "content": "..."}}, ...]
            documents = [item['data'] for item in data if 'data' in item]
        elif isinstance(data, dict) and 'documents' in data:
            # Format: {"documents": [{"id": "...", "content": "..."}, ...]}
            documents = data['documents']
        else:
            raise ValueError(f"Unknown document format in {data_path}. Expected list or dict with 'documents' key.")

        logger.info(f"Loaded {len(documents)} documents")
        
        chunk_id = 0
        failed_chunks = 0
        skipped_docs = []
        
        for doc_idx, doc in enumerate(documents):
            doc_id = doc['id']
            content = doc['content']
            
            # Log progress every 10 documents
            if doc_idx % 10 == 0:
                logger.info(f"Processing document {doc_idx + 1}/{len(documents)}: {doc_id} (chunks so far: {chunk_id})")
            
            # Skip empty content
            if not content or not content.strip():
                logger.warning(f"Skipping document {doc_id} with empty content")
                skipped_docs.append(doc_id)
                continue
            
            chunks = self.fixed_size_chunking(
                content,
                self.config.chunk_size,
                self.config.chunk_overlap
            )
            
            for chunk_idx, chunk_text in enumerate(chunks):
                # Skip empty chunks
                if not chunk_text or not chunk_text.strip():
                    continue
                
                # Truncate extremely long chunks (sentence-transformers has token limits)
                # Most models handle up to 512 tokens, so truncate at ~2000 chars to be safe
                original_length = len(chunk_text)
                if original_length > 2000:
                    chunk_text = chunk_text[:2000]
                    logger.debug(f"Truncated chunk {chunk_id} from doc {doc_id} (was {original_length} chars, now {len(chunk_text)})")
                
                try:
                    embedding = embedder.embed(chunk_text)
                    chunk = Chunk(id=chunk_id, doc_id=doc_id, string=chunk_text, embedding=embedding)
                    self.add(chunk)
                    chunk_id += 1
                except Exception as e:
                    failed_chunks += 1
                    logger.warning(f"Failed to embed chunk {chunk_idx} from doc {doc_id} (chunk_id would be {chunk_id}): {e}")
                    logger.debug(f"Problematic chunk preview: {chunk_text[:100]}...")
                    continue
        
        logger.info(f"Embedding completed, total chunks - {len(self.chunks)}")
        if failed_chunks > 0:
            logger.warning(f"Skipped {failed_chunks} chunks that failed to embed")
        if skipped_docs:
            logger.warning(f"Skipped {len(skipped_docs)} documents with empty content")


