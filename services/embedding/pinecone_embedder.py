from pinecone import Pinecone

class PineconeEmbedder:
    def __init__(self, client: Pinecone):
        self._pc = client  # injected, not a global

    def embed_dense(self, texts: list[str], input_type: str = "passage") -> list[list[float]]:
        response = self._pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=texts,
            parameters={"input_type": input_type, "truncate": "END"}
        )
        return [item["values"] for item in response]

    def embed_sparse(self, texts: list[str], input_type: str = "passage") -> list[dict]:
        response = self._pc.inference.embed(
            model="pinecone-sparse-english-v0",
            inputs=texts,
            parameters={"input_type": input_type, "truncate": "END"}
        )
        return [
            {"indices": item["sparse_indices"], "values": item["sparse_values"]}
            for item in response
        ]

    def embed_both(self, texts: list[str], input_type: str = "passage") -> list[dict]:
        dense  = self.embed_dense(texts, input_type)
        sparse = self.embed_sparse(texts, input_type)
        return [{"dense": d, "sparse": s} for d, s in zip(dense, sparse)]