from typing import List, Dict
from config import GROQ_API_KEY, Tool_MODEL
from groq import Groq
from models import RouteDecision


groq_client = Groq(api_key=GROQ_API_KEY)


def route_query_to_documents(query: str, memory: Dict) -> List[str]:
    if not memory:
        raise RuntimeError("No documents in memory.")

    doc_catalog = []
    for doc_id, data in memory.items():
        doc_catalog.append(
            f"doc_id: {doc_id}\n"
            f"file: {data['metadata'].get('file_name', 'unknown')}\n"
            f"summary: {data['summary']}\n"
            f"keywords: {', '.join(data['keywords'])}"
        )

    catalog_text = "\n\n".join(doc_catalog)

    prompt = f"""
You are a document routing agent. Given a user query, select the
doc_ids of documents that are likely to contain the answer.
Return an empty list if nothing is relevant.

--- AVAILABLE DOCUMENTS ---
{catalog_text}
--- END OF DOCUMENTS ---

User Query: {query}
"""

    response = groq_client.chat.completions.create(
        model=Tool_MODEL,
        messages=[
            {"role": "system", "content": "You are a document routing agent."},
            {"role": "user",   "content": prompt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name":   "RouteDecision",
                "schema": RouteDecision.model_json_schema()  # ← { doc_ids: [...] }
            }
        }
    )

    content = response.choices[0].message.content
    result  = RouteDecision.model_validate_json(content)  # ← validated, no regex

    # Filter to only doc_ids that actually exist in memory
    valid_doc_ids = [d for d in result.doc_ids if d in memory]

    print(f"[ROUTER] Query routed to {len(valid_doc_ids)} document(s): {valid_doc_ids}")
    return valid_doc_ids

