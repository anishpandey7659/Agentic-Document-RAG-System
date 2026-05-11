from .pinecone_embedder import PineconeEmbedder
from .mistral_embedder import MistralEmbedder
from .embedding_store import EmbeddingStore

class EmbeddingService:
    def __init__(
        self,
        mistral_embedder: MistralEmbedder,
        pinecone_embedder: PineconeEmbedder,
        store: EmbeddingStore
    ):
        self.mistral   = mistral_embedder
        self.pinecone  = pinecone_embedder
        self.store     = store
    
