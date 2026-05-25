# pipeline/summarizer.py
from typing import List
from core.config import summarize_llm
from core.dependencies import groq_client
from Model_Memory_store.models import DocumentSummary


class Summarizer:
    """
    Summarizes long documents by:
    1. Splitting into chunks
    2. Summarizing each chunk
    3. Merging into a final summary + keywords
    """

    CHUNK_SIZE    = 12_000
    CHUNK_OVERLAP = 500

    SYSTEM_CHUNK  = "You are a document analysis system. Summarize the given text concisely in max 100 words."
    SYSTEM_MERGE  = "You are a document analysis system."

    def __init__(self, groq_client: groq_client, model: str = summarize_llm):
        self._groq  = groq_client
        self._model = model


    def _chunk_text(self, text: str) -> List[str]:
        if len(text) <= self.CHUNK_SIZE:
            return [text]

        chunks = [
            text[i: i + self.CHUNK_SIZE]
            for i in range(0, len(text), self.CHUNK_SIZE - self.CHUNK_OVERLAP)
        ]
        print(f"[INFO] Document split into {len(chunks)} chunks.")
        return chunks

    def _summarize_chunk(self, chunk: str, index: int, total: int) -> str:
        response = self._groq.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self.SYSTEM_CHUNK},
                {"role": "user",   "content": f"Chunk {index + 1} of {total}:\n\n{chunk}"}
            ]
        )
        return response.choices[0].message.content.strip()

    def _merge_summaries(self, chunk_summaries: List[str]) -> DocumentSummary:
        combined = "\n\n".join(
            f"[Part {i + 1}]: {s}" for i, s in enumerate(chunk_summaries)
        )

        response = self._groq.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self.SYSTEM_MERGE},
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
        return DocumentSummary.model_validate_json(
            response.choices[0].message.content
        )


    def summarize(self, text: str) -> dict:
        chunks = self._chunk_text(text)

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"[INFO] Summarizing chunk {i + 1}/{len(chunks)}...")
            chunk_summaries.append(self._summarize_chunk(chunk, i, len(chunks)))

        print("[INFO] Merging chunk summaries...")
        result = self._merge_summaries(chunk_summaries)
        return {"summary": result.summary, "keywords": result.keywords}