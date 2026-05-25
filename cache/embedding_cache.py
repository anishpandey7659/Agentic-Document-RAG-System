# cache/embedding_cache.py

import hashlib
import json
from cache.redis_client import r

class EmbeddingCache:

    def __init__(self):
        self.r = r

    def __hash_key(self, text):
        return hashlib.sha256(text.encode()).hexdigest()

    def get_embedding_cache(self, text):
        key = "emb:" + self.__hash_key(text)

        val = self.r.get(key)
        if val:
            print("[CACHE HIT] embedding")
            return json.loads(val)

        return None


    def set_embedding_cache(self, text, embedding):
        key = "emb:" + self.__hash_key(text)
        self.r.set(key, json.dumps(embedding))