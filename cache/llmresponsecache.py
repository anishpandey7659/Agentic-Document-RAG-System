import hashlib
import json
from cache.redis_client import r

class LLMResponseCache:
    def __init__(self, ttl: int = 3600):
        self._r   = r
        self._ttl = ttl

    def _make_key(self, query: str, chunks: list) -> str:
        chunk_text = " ||| ".join(c["text"] for c in chunks)
        combined   = f"{query} ||| {chunk_text}"
        return "llm:" + hashlib.sha256(combined.encode()).hexdigest()

    def find(self, query: str, chunks: list) -> dict | None:
        key  = self._make_key(query, chunks)
        data = self._r.get(key)
        return json.loads(data) if data else None 

    def store(self, query: str, chunks: list, answer: str, sources) -> None:
        key = self._make_key(query, chunks)
        self._r.set(key, json.dumps({"answer": answer, "sources": sources}), ex=self._ttl)  