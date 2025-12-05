from sentence_transformers import SentenceTransformer
from rag.embed.embedding import Embedding
from config import RAGConfig, EmbedderType
import logging
import os

logger = logging.getLogger(__name__)

try:
    import ollama
except ImportError:
    ollama = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# CMU Gateway configuration for API-based services
CMU_GATEWAY_BASE_URL = "https://ai-gateway.andrew.cmu.edu"
GATEWAY_API_KEY = "sk-gx6H5IC6DMWM311Ef2Gdyw"

class Embedder:
    def __init__(self, config):
        self.config = config
        self.model = None  
        self.max_token_size = 512 
        self.embedding_size = 1024  
    
    def embed(self, query: str) -> Embedding:
        return Embedding([1,2,3])

class MiniLM6Embedder(Embedder):
    def __init__(self, config):
        super().__init__(config)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.max_token_size = 256
        self.embedding_size = 384

    def embed(self, query: str) -> Embedding:
        # Ensure query is a string and not empty
        if not isinstance(query, str):
            query = str(query)
        if not query.strip():
            raise ValueError("Cannot embed empty string")
        
        # Truncate if too long (sentence-transformers models typically handle 512 tokens max)
        # Approximate: 1 token ≈ 4 characters, so 512 tokens ≈ 2000 chars
        max_length = 2000
        if len(query) > max_length:
            query = query[:max_length]
        
        try:
            embedding_vector = self.model.encode(query, show_progress_bar=False, convert_to_numpy=True)
            return Embedding(embedding_vector)
        except Exception as e:
            logger.error(f"Failed to embed text (length: {len(query)}): {e}")
            raise

class MiniLM12Embedder(Embedder):
    def __init__(self, config):
        super().__init__(config)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        self.max_token_size = 256
        self.embedding_size = 384
        
    def embed(self, query: str) -> Embedding:
        # Ensure query is a string and not empty
        if not isinstance(query, str):
            query = str(query)
        if not query.strip():
            raise ValueError("Cannot embed empty string")
        
        # Truncate if too long (sentence-transformers models typically handle 512 tokens max)
        # Approximate: 1 token ≈ 4 characters, so 512 tokens ≈ 2000 chars
        max_length = 2000
        if len(query) > max_length:
            query = query[:max_length]
        
        try:
            embedding_vector = self.model.encode(query, show_progress_bar=False, convert_to_numpy=True)
            return Embedding(embedding_vector)
        except Exception as e:
            logger.error(f"Failed to embed text (length: {len(query)}): {e}")
            raise

class Mxbai(Embedder):
    def __init__(self, config):
        super().__init__(config)
        self.model = "mxbai-embed-large"
        self.max_token_size = 512
        self.embedding_size = 1024 
        
    def embed(self, query: str) -> Embedding:
        if ollama is None:
            raise ImportError("ollama package is required for MXBAI embedder")
        embedding_vector = ollama.embed(model=self.model, input=query)["embeddings"][0]
        return Embedding(embedding_vector)

class AzureEmbedder(Embedder):
    """API-based embedder using Azure OpenAI embeddings via CMU Gateway."""
    
    def __init__(self, config):
        super().__init__(config)
        if OpenAI is None:
            raise ImportError("openai package required for AzureEmbedder. Install with: pip install openai")
        
        env_key = os.getenv("CMU_GATEWAY_API_KEY") or os.getenv("OPENAI_API_KEY")
        api_key = env_key if env_key and env_key.startswith('sk-') else GATEWAY_API_KEY
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=CMU_GATEWAY_BASE_URL
        )
        self.model_name = "azure/text-embedding-3-small" 
        self.embedding_size = 1536  
        self.max_token_size = 8191
    
    def embed(self, query: str) -> Embedding:
        if not isinstance(query, str):
            query = str(query)
        if not query.strip():
            raise ValueError("Cannot embed empty string")
        max_length = 32000  
        if len(query) > max_length:
            query = query[:max_length]
        
        response = self.client.embeddings.create(
            model=self.model_name,
            input=query
        )
        embedding_vector = response.data[0].embedding
        return Embedding(embedding_vector)
    
def embedder_factory(config: RAGConfig):
    if config.embedder_type is EmbedderType.BASE:
        return Embedder(config)
    elif config.embedder_type is EmbedderType.MINILM6:
        return MiniLM6Embedder(config)
    elif config.embedder_type is EmbedderType.MINILM12:
        return MiniLM12Embedder(config)
    elif config.embedder_type is EmbedderType.MXBAI:
        return Mxbai(config)
    elif config.embedder_type is EmbedderType.AZURE:
        return AzureEmbedder(config)
    else:
        raise TypeError(f"Weird embedder type - {config.embedder_type}")

