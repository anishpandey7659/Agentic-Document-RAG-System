from pydantic import BaseModel
from services import _call_groq_with_retry
from typing import List
from config import summarize_llm
from models import DocumentSummary



def _summarize_chunk(chunk: str, chunk_index: int, total_chunks: int) -> str:
    response = _call_groq_with_retry(
        model=summarize_llm,
        messages=[
            {"role": "system", "content": "You are a document analysis system. Summarize the given text concisely in max 100 words."},
            {"role": "user",   "content": f"Chunk {chunk_index+1} of {total_chunks}:\n\n{chunk}"}
        ]
    )
    return response.choices[0].message.content.strip()


def _merge_summaries(chunk_summaries: List[str]) -> DocumentSummary:
    combined = "\n\n".join(f"[Part {i+1}]: {s}" for i, s in enumerate(chunk_summaries))

    response = _call_groq_with_retry(
        model=summarize_llm,
        messages=[
            {"role": "system", "content": "You are a document analysis system."},
            {"role": "user",   "content": f"""
These are summaries of different parts of a single document.
Produce:
1. One final concise summary (max 50 words) of the whole document
2. 5 most important keywords across all parts

Parts:
{combined}
"""}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name":   "DocumentSummary",
                "schema": DocumentSummary.model_json_schema()
            }
        }
    )
    return DocumentSummary.model_validate_json(response.choices[0].message.content)



def summarize_and_extract_keywords(text: str) -> dict:
    CHUNK_SIZE = 12000

    if len(text) <= CHUNK_SIZE:
        chunks = [text]
    else:
        chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE - 500)]
        print(f"[INFO] Document split into {len(chunks)} chunks for summarization.")

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"[INFO] Summarizing chunk {i+1}/{len(chunks)}...")
        
        summary = _summarize_chunk(chunk, i, len(chunks))
        chunk_summaries.append(summary)

    print("[INFO] Merging chunk summaries...")
    result = _merge_summaries(chunk_summaries)
    return {"summary": result.summary, "keywords": result.keywords}