# cache/semantic_cache.py

import numpy as np
import hashlib
import json
from cache.redis_client import r


class SemanticCache:

    def __init__(self, threshold=0.90):
        self._r = r
        self.threshold = threshold

    def find(self, query_emb):
        keys = self._r.keys("sem:*")

        for k in keys:
            item = json.loads(self._r.get(k))
            score = self.cosine(query_emb, item["emb"])

            if score > self.threshold:
                print("[CACHE HIT] semantic")
                return item["answer"]

        return None
    
    def cosine(self, a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


    def store(self, query, emb, answer,chunks):
        key = "sem:" + hashlib.sha256(query.encode()).hexdigest()

        self._r.set(key, json.dumps({
            "emb": emb,
            "answer": answer,
            "chunks": chunks
        }))