from config import EMBED_MODEL, MISTRAL_API_KEY,EMBEDDING_FILE
from mistralai.client import Mistral
from typing import List, Dict
from .pinecone_client import pc
import pickle

mistral_client = Mistral(api_key=MISTRAL_API_KEY)

def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    response = mistral_client.embeddings.create(
        model=EMBED_MODEL,
        inputs=chunks        
    )
    return [item.embedding for item in response.data]


def embed_dense(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    """
    Generate dense embeddings using llama-text-embed-v2.

    Args:
        texts      : list of strings to embed
        input_type : "passage" for documents, "query" for search queries

    Returns:
        list of float vectors (1024-dimensional)
    """
    response = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=texts,
        parameters={
            "input_type": input_type,
            "truncate": "END"
        }
    )
    return [item["values"] for item in response]


def embed_sparse(texts: list[str], input_type: str = "passage") -> list[dict]:
    """
    Generate sparse embeddings using pinecone-sparse-english-v0.

    Args:
        texts      : list of strings to embed
        input_type : "passage" for documents, "query" for search queries

    Returns:
        list of dicts with 'indices' and 'values' keys
    """
    response = pc.inference.embed(
        model="pinecone-sparse-english-v0",
        inputs=texts,
        parameters={
            "input_type": input_type,
            "truncate": "END"
        }
    )
    return [
        {
            "indices": item["sparse_indices"],
            "values":  item["sparse_values"]
        }
        for item in response
    ]


def embed_both(texts: list[str], input_type: str = "passage") -> list[dict]:
    """
    Generate both dense and sparse embeddings together.

    Args:
        texts      : list of strings to embed
        input_type : "passage" for documents, "query" for search queries

    Returns:
        list of dicts with 'dense' and 'sparse' keys
    """
    dense_results  = embed_dense(texts, input_type)
    sparse_results = embed_sparse(texts, input_type)

    return [
        {"dense": d, "sparse": s}
        for d, s in zip(dense_results, sparse_results)
    ]

import os
import pickle




def load_document_embeddings():

    if not os.path.exists(EMBEDDING_FILE):
        return {}
    
    if os.path.getsize(EMBEDDING_FILE) == 0:
        return {}

    try:
        with open(EMBEDDING_FILE, "rb") as f:
            return pickle.load(f)

    except Exception as e:
        print(f"[WARNING] Failed to load embeddings: {e}")
        return {}


def save_document_embedding(doc_id, embedding):

    doc_embeddings = load_document_embeddings()

    doc_embeddings[doc_id] = embedding

    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump(doc_embeddings, f)

    print(f"[INFO] Saved embedding for {doc_id}")

