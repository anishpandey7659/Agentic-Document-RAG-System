from typing import List, Dict
from Model_Memory_store.models import DocumentAgent
from config import EMBED_MODEL, MEMORY_FILE
import json
from dataclasses import asdict

def create_document_agent(
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
            "type":"document_agent",
            "status":"active",
            "embed_model":EMBED_MODEL,
            "file_name":file_name,  
        }
    )

def register_document_agent(agent: DocumentAgent):
    try:
        with open(MEMORY_FILE, "r") as f:
            memory = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        memory = {}

    memory[agent.doc_id] = asdict(agent)

    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

    print(f"[INFO] Agent '{agent.doc_id}' saved to {MEMORY_FILE}")


def load_system_memory() -> Dict:
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    
    
def list_available_documents() -> None:
    memory = load_system_memory()
    if not memory:
        print("[INFO] No documents registered yet.")
        return
    print(f"[INFO] {len(memory)} document(s) found in system memory:")
    print(f"\n{'='*55}")
    for doc_id, data in memory.items():
        print(f"  doc_id  : {doc_id}")
        print(f"  summary : {data['summary']}")
        print(f"  keywords: {', '.join(data['keywords'])}")
        print(f"  index   : {data['vector_db']}")
        print(f"  {'-'*53}")


def delete_document_agent(doc_id: str, pinecone_client=None) -> bool:
    """
    Deletes a document agent from system memory.
    Also deletes its Pinecone index too.
    """
    memory = load_system_memory()

    if doc_id not in memory:
        print(f"[WARNING] Agent '{doc_id}' not found in memory.")
        return False

    agent_data = memory[doc_id]

    # Optionally delete the Pinecone index
    if pinecone_client:
        index_name = agent_data.get("vector_db")
        if index_name:
            try:
                pinecone_client.delete_index(index_name)
                print(f"[INFO] Pinecone index '{index_name}' deleted.")
            except Exception as e:
                print(f"[WARNING] Could not delete Pinecone index '{index_name}': {e}")

    # Remove from memory
    del memory[doc_id]

    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

    print(f"[INFO] Agent '{doc_id}' removed from {MEMORY_FILE}")
    return True