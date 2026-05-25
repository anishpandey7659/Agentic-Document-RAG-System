import json
from typing import Dict, Optional
from dataclasses import asdict
from Model_Memory_store.models import DocumentAgent
from core.config import EMBED_MODEL, MEMORY_FILE,EMBEDDING_FILE


class DocumentAgentFactory:
    """Creates DocumentAgent instances."""

    @staticmethod
    def create(
        doc_id: str,
        summary_data: Dict,
        vector_db: str,
        file_name: str,
    ) -> DocumentAgent:
        return DocumentAgent(
            doc_id=doc_id,
            summary=summary_data["summary"],
            keywords=summary_data["keywords"],
            vector_db=vector_db,
            metadata={
                "type":        "document_agent",
                "status":      "active",
                "embed_model": EMBED_MODEL,
                "file_name":   file_name,
            }
        )


class AgentMemoryStore:
    """Handles persistence of DocumentAgents to a JSON memory file."""

    def __init__(self, memory_file: str = MEMORY_FILE,embeded_file:str =EMBEDDING_FILE):
        self._path = memory_file
        self._emb_path = embeded_file


 
    def _load(self) -> Dict:
        try:
            with open(self._path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        

    def _save(self, memory: Dict) -> None:
        with open(self._path, "w") as f:
            json.dump(memory, f, indent=2)


    def register(self, agent: DocumentAgent) -> None:
        memory = self._load()
        memory[agent.doc_id] = asdict(agent)
        self._save(memory)
        print(f"[INFO] Agent '{agent.doc_id}' saved to {self._path}")

    def load_all(self) -> Dict:
        return self._load()

    def get(self, doc_id: str) -> Optional[Dict]:
        return self._load().get(doc_id)

    def delete(self, doc_id: str,source_name:str,emb_store,pinecone_client=None) -> bool:
        memory = self._load()

        if doc_id not in memory :
            print(f"[WARNING] Agent '{doc_id}' not found in memory.")
            return False

        if pinecone_client:
            index_name = memory[doc_id].get("vector_db")
            if index_name:
                try:
                    pinecone_client.delete_by_source(index_name,source_name)
                    print(f"[INFO] From Pinecone index '{index_name}':'{source_name}' is deleted.")
                except Exception as e:
                    print(f"[WARNING] Could not delete index '{index_name}': {e}")

        del memory[doc_id]
        emb_store.delete(doc_id)
        self._save(memory)

        print(f"[INFO] Agent '{doc_id}' removed from {self._path}")
        return True

    def list_all(self) -> None:
        memory = self._load()
        if not memory:
            print("[INFO] No documents registered yet.")
            return

        print(f"[INFO] {len(memory)} document(s) in memory:\n{'=' * 55}")
        for doc_id, data in memory.items():
            print(f"  doc_id  : {doc_id}")
            print(f"  summary : {data['summary']}")
            print(f"  keywords: {', '.join(data['keywords'])}")
            print(f"  index   : {data['vector_db']}")
            print(f"  {'-' * 53}")