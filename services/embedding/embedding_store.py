import os
import pickle
from core.config import EMBEDDING_FILE

class EmbeddingStore:
    def __init__(self, filepath: str = EMBEDDING_FILE):
        self._filepath = filepath

    def load(self) -> dict:
        if not os.path.exists(self._filepath) or os.path.getsize(self._filepath) == 0:
            return {}
        try:
            with open(self._filepath, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[WARNING] Failed to load embeddings: {e}")
            return {}

    def save(self, doc_id: str, embedding) -> None:
        data = self.load()
        data[doc_id] = embedding
        with open(self._filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"[INFO] Saved embedding for {doc_id}")
    
    def delete(self, doc_id: str) -> None:
        data = self.load()        
        
        if doc_id in data:
            del data[doc_id]      
            with open(self._filepath, "wb") as f:
                pickle.dump(data, f)   
            print(f"[INFO] Deleted embedding for {doc_id}")
        else:
            print(f"[WARNING] {doc_id} not found")