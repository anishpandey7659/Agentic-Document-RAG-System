"""
pipeline/chunker.py

Sentence/paragraph-aware text chunker for RAG pipelines.
Outputs LangChain Document objects with rich, safe metadata.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_RESERVED_METADATA_KEYS = frozenset(
    {"chunk_id", "chunk_index", "chunk_total", "chunk_char_length", "source"}
)
_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

class Chunker:
    """
    Sentence/paragraph-aware chunker for RAG pipelines.

    Args:
        chunk_size:     Maximum size of each chunk, measured by
                        ``length_function`` (default: character count).
                        Must be >= 50.
        chunk_overlap:  Number of units overlapping between consecutive
                        chunks.  Must be > 0 and < ``chunk_size``.
        length_function: Function used to measure chunk size.
                        Defaults to ``len`` (character count).  Pass a
                        token-counting function to align with your model's
                        context window (e.g. ``lambda t: len(tokenizer.encode(t))``).
        separators:     Ordered list of separator strings passed to
                        ``RecursiveCharacterTextSplitter``.  Overriding
                        this is rarely necessary.

    Example::

        chunker = Chunker(chunk_size=512, chunk_overlap=64)
        docs = chunker.chunk(text, source="report.pdf")
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        length_function: Callable[[str], int] = len,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size < 50:
            raise ValueError(
                f"chunk_size must be at least 50, got {chunk_size}"
            )
        if chunk_overlap <= 0:
            raise ValueError(
                f"chunk_overlap must be > 0, got {chunk_overlap}"
            )
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller "
                f"than chunk_size ({chunk_size})"
            )

        self._length_function = length_function
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            keep_separator=True,
            separators=separators or _DEFAULT_SEPARATORS,
        )

    def chunk(
        self,
        text: str,
        source: str = "unknown",
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Split *text* into overlapping chunks and return them as a list
        of LangChain ``Document`` objects.

        Args:
            text:           The text to split.
            source:         Identifier for the originating document
                            (e.g. a file path or URL).  Used to namespace
                            ``chunk_id`` so IDs are unique across calls.
            extra_metadata: Additional key/value pairs merged into every
                            chunk's metadata.  Reserved keys
                            (``chunk_id``, ``chunk_index``, ``chunk_total``,
                            ``chunk_char_length``, ``source``) are ignored
                            to prevent accidental overwrites.

        Returns:
            A (possibly empty) list of ``Document`` objects.
        """
        if not text.strip():
            return []

        safe_extra: dict[str, Any] = {
            k: v
            for k, v in (extra_metadata or {}).items()
            if k not in _RESERVED_METADATA_KEYS
        }

        raw_chunks = self._splitter.split_text(text)
        total = len(raw_chunks)

        documents: list[Document] = []
        for idx, chunk in enumerate(raw_chunks):
            metadata: dict[str, Any] = {
                "chunk_id": f"{source}:{idx}",
                "chunk_index": idx,
                "chunk_total": total,
                "source": source,
                "chunk_char_length": self._length_function(chunk),
                **safe_extra,
            }
            documents.append(Document(page_content=chunk, metadata=metadata))

        return documents