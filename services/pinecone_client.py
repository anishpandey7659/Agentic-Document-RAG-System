from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict
from config import PINECONE_API_KEY, PINECONE_CLOUD, PINECONE_REGION


pc= Pinecone(api_key=PINECONE_API_KEY)


def store_in_pinecone(doc_id: str, chunks: List[str], embeddings: List[List[float]]) -> str:
    index_name = f"{doc_id.lower()}-index"

    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            vector_type="dense",   # Required for hybrid
            dimension=1024,        
            metric="dotproduct",  # Required for hybrid — NOT cosine or euclidean
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )

    index = pc.Index(index_name)

    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": f"{doc_id}_chunk_{i}",
                "values": emb["dense"],           # ← dense vector
                "sparse_values": emb["sparse"],   # ← sparse vector (new)
                "metadata": {
                    "text": chunk,
                    "doc_id": doc_id,
                    "chunk_id": i
                }
            })

    index.upsert(vectors=vectors)
    return index_name



def search_index(index_name: str, hdense: List[float], hsparse: Dict, top_k: int = 5) -> List[Dict]:
    index = pc.Index(index_name)

    results = index.query(
        namespace="",
        top_k=top_k,
        vector=hdense,
        sparse_vector=hsparse,
        include_metadata=True
    )

    matches = []
    for match in results["matches"]:
        matches.append({
            "score":    round(match["score"], 4),
            "text":     match["metadata"]["text"],
            "doc_id":   match["metadata"]["doc_id"],
            "chunk_id": match["metadata"]["chunk_id"],
        })

    return matches