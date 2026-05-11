from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict
from config import PINECONE_API_KEY, PINECONE_CLOUD, PINECONE_REGION


class PineconeClient:
    """Handles Pinecone connection and index management."""

    def __init__(self, api_key: str = PINECONE_API_KEY):
        self._pc = Pinecone(api_key=api_key)

    def get_or_create_index(self, index_name: str, dimension: int = 1024) -> None:
        """Creates index if it doesn't already exist."""
        existing = [idx["name"] for idx in self._pc.list_indexes()]
        if index_name not in existing:
            self._pc.create_index(
                name=index_name,
                vector_type="dense",
                dimension=dimension,
                metric="dotproduct",
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION
                )
            )

    def get_index(self, index_name: str):
        return self._pc.Index(index_name)

    @property
    def client(self):
        """Expose raw client for embedder or other services if needed."""
        return self._pc


class PineconeVectorStore:
    """Handles upsert and search operations against a Pinecone index."""

    def __init__(self, pinecone_client: PineconeClient):
        self._client = pinecone_client  # injected, not created here

    def store(
        self,
        doc_id: str,
        chunks: List[str],
        embeddings: List[Dict],   # each dict has 'dense' and 'sparse'
        index_name: str,
        file_name: str,
        domain: str
    ) -> str:
        self._client.get_or_create_index(index_name)
        index = self._client.get_index(index_name)

        vectors = [
            {
                "id": f"{doc_id}_chunk_{i}",
                "values": emb["dense"],
                "sparse_values": emb["sparse"],
                "metadata": {
                    "text": chunk,
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "source": file_name,
                    "domain": domain,
                }
            }
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]

        index.upsert(vectors=vectors)
        return index_name

    def search(
        self,
        index_name: str,
        dense: List[float],
        sparse: Dict,
        top_k: int = 5
    ) -> List[Dict]:
        index = self._client.get_index(index_name)

        results = index.query(
            namespace="",
            top_k=top_k,
            vector=dense,
            sparse_vector=sparse,
            include_metadata=True
        )

        return [
            {
                "score":    round(match["score"], 4),
                "text":     match["metadata"]["text"],
                "doc_id":   match["metadata"]["doc_id"],
                "chunk_id": match["metadata"]["chunk_id"],
                "domain":   match["metadata"]["domain"],
                "source":   match["metadata"]["source"],
            }
            for match in results["matches"]
        ]