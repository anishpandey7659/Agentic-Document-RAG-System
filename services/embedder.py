from config import EMBED_MODEL, MISTRAL_API_KEY
from mistralai.client import Mistral
from typing import List, Dict


mistral_client = Mistral(api_key=MISTRAL_API_KEY)

def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    response = mistral_client.embeddings.create(
        model=EMBED_MODEL,
        inputs=chunks        
    )
    return [item.embedding for item in response.data]