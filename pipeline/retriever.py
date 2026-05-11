from typing import List, Dict, Generator, Union
from services.GroqClient import groq_client
from pipeline.smart_search import SmartSearch
from pipeline.rerank import Reranker
from config import Tool_MODEL

NO_CONTENT_MSG = "No relevant content found for your query."

SYSTEM_PROMPT_RAG    = "You are a document Q&A assistant. Answer only from the provided context."
SYSTEM_PROMPT_NORMAL = "You are a helpful assistant. Give accurate answers."


class Retriever:
    """
    Orchestrates the full RAG pipeline:
    SmartSearch → Rerank → Answer
    """

    def __init__(
        self,
        groq_client: groq_client,
        smart_search: SmartSearch,
        reranker: Reranker,
        model: str = Tool_MODEL,
    ):
        self._client        = groq_client
        self._smart_search = smart_search
        self._reranker     = reranker
        self._model        = model

    def _answer_with_context(self, query: str, context: str, stream: bool = False):
        prompt = f"""
Answer the user's question using ONLY the document context below.
If the context does not contain enough information, say so clearly.

--- DOCUMENT CONTEXT ---
{context}
--- END CONTEXT ---

User Question: {query}
Answer:"""

        return  self._client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_RAG},
                {"role": "user",   "content": prompt}
            ],
            model=self._model,
            stream=stream
        )


    def retrieve_and_answer(
        self,
        query: str,
        alpha: float = 0.7,
        top_k: int = 6,
        top_n: int = 3,
        show_chunks: bool = False,
        stream: bool = False,
    ) -> Union[Dict, Generator]:

        print(f"\n[QUERY] {query}")
        matches = self._smart_search.search(query, alpha, top_k)

        if not matches:
            return self._empty_response(stream)

        reranked = self._reranker.rerank(query, matches, top_n)
        if not reranked:
            return self._empty_response(stream)

        if show_chunks:
            self._log_chunks(reranked)

        context = SmartSearch.build_context(reranked)

        if stream:
            return self._stream_response(query, context, reranked)

        answer = self._answer_with_context(query, context)
        return {
            "answer":  answer.choices[0].message.content.strip(),
            "sources": reranked
        }


    def answer_normal(self,query: str, stream: bool = False):
        response = groq_client.chat.completions.create(
            model=Tool_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_NORMAL},
                {"role": "user", "content": query}
            ],
            stream=stream,
        )
        full_response = ""
        if stream:
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    yield delta
        else:
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta

            return full_response.strip()

    # Private helpers
    def _empty_response(self, stream: bool):
        if stream:
            def _gen():
                yield {"type": "sources", "sources": []}
                yield {"type": "token",   "token": NO_CONTENT_MSG}
            return _gen()
        return {"answer": NO_CONTENT_MSG, "sources": []}

    def _stream_response(self, query: str, context: str, reranked: List[Dict]):
        yield {"type": "sources", "sources": reranked}
        for chunk in self._answer_with_context(query, context, stream=True):
            token = chunk.choices[0].delta.content or ""
            if token:
                yield {"type": "token", "token": token}

    @staticmethod
    def _log_chunks(chunks: List[Dict]) -> None:
        print("\n--- Retrieved Chunks ---")
        for m in chunks:
            print(f"Rank {m['rank']} (score={m['score']}) | {m['text']}")
        print("------------------------\n")