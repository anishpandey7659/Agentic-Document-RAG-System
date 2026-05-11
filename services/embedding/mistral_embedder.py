from mistralai.client import Mistral
from typing import List
from config import EMBED_MODEL

class MistralEmbedder:
    def __init__(self, api_key: str):
        self._client = Mistral(api_key=api_key)

    def embed(self, chunks: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(
            model=EMBED_MODEL,
            inputs=chunks
        )
        return [item.embedding for item in response.data]