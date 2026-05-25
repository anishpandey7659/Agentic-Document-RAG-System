from cohere import Document
from pinecone import Pinecone


class PineconeEmbedder:
    
    def __init__(self, client: Pinecone):
        self._pc = client  

    _MAX_BATCH_SIZE = 96

    def _batch(self, items: list[str]):
        for i in range(0, len(items), self._MAX_BATCH_SIZE):
            yield items[i:i + self._MAX_BATCH_SIZE]

    def embed_dense(
        self,
        texts: list[str],
        input_type: str = "passage"
    ) -> list[list[float]]:
        embeddings = []
        for batch in self._batch(texts):
            response = self._pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=batch,
                parameters={
                    "input_type": input_type,
                    "truncate": "END"
                }
            )

            embeddings.extend(
                item["values"]
                for item in response
            )

        return embeddings

    def embed_sparse(
        self,
        texts: list[str],
        input_type: str = "passage"
    ) -> list[dict]:

        sparse_embeddings = []

        for batch in self._batch(texts):

            response = self._pc.inference.embed(
                model="pinecone-sparse-english-v0",
                inputs=batch,
                parameters={
                    "input_type": input_type,
                    "truncate": "END"
                }
            )
            sparse_embeddings.extend([
                {
                    "indices": item["sparse_indices"],
                    "values": item["sparse_values"]
                }
                for item in response
            ])

        return sparse_embeddings

    def embed_both(
        self,
        texts: list[str],
        input_type: str = "passage"
    ) -> list[dict]:

        dense = self.embed_dense(texts, input_type)
        sparse = self.embed_sparse(texts, input_type)

        return [
            {"dense": d, "sparse": s}
            for d, s in zip(dense, sparse)
        ]