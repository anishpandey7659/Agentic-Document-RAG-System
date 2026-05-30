from typing import List, Dict, Generator, Union
from services.GroqClient import groq_client
from pipeline.smart_search import SmartSearch
from pipeline.rerank import Reranker
from core.config import Tool_MODEL
import time


NO_CONTENT_MSG = "No relevant content found for your query."


SYSTEM_PROMPT_RAG    = """You are a helpful assistant. Answer the user's question using ONLY the provided context. Follow these rules strictly:

1. If the answer is in the context, answer directly and cite the relevant source.
2. If the context partially answers the question, share what you know and explicitly state what is missing.
3. If the context does not contain the answer, say: "I don't have enough information in the provided context to answer this question."
4. Do NOT use prior knowledge or make assumptions beyond what the context states.
5. Keep answers concise and grounded in the context."""

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


    def retrieve_chunks(
        self,
        query: str,
        query_embedding: List[float],
        mode: str = "dense",
        alpha: float = 0.7,
        top_k: int = 6,
        top_n: int = 3,
        show_chunks: bool = False,
    ) -> List[Dict]:
        """Step 1 — retrieval + rerank only, no LLM call."""

        t_retrieve = time.perf_counter()
        if mode == "hybrid":
            matches = self._smart_search.search(query, query_embedding, mode, alpha, top_k)
        else:
            matches = self._smart_search.search(query, query_embedding, mode, top_k)
        # print(f"[Latency] Retrieval: {time.perf_counter()-t_retrieve:.4f}s")

        if not matches:
            return []

        t_rerank = time.perf_counter()
        reranked = self._reranker.rerank(query, matches, top_n)
        print(f"[Latency] Reranking: {time.perf_counter()-t_rerank:.4f}s")

        if show_chunks:
            self._log_chunks(reranked)

        return reranked or []


    def answer_from_chunks(
        self,
        query: str,
        chunks: List[Dict],
        stream: bool = False,
    ) -> Union[Dict, Generator]:
        """Step 2 — LLM call only, chunks already retrieved."""

        if not chunks:
            return self._empty_response(stream)

        t_context = time.perf_counter()
        context = SmartSearch.build_context(chunks)
        print(f"[Latency] Context Building: {time.perf_counter()-t_context:.4f}s")

        if stream:
            return self._stream_response(query, context, chunks)

        t_generation = time.perf_counter()
        answer = self._answer_with_context(query, context)
        print(f"[Latency] Answer Generation: {time.perf_counter()-t_generation:.4f}s")

        return {
            "answer":  answer.choices[0].message.content.strip(),
            "sources": chunks
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