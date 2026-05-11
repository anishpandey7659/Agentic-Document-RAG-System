# pipeline/chunker.py

from typing import List


class Chunker:
    """Splits cleaned text into overlapping word-based chunks."""

    def __init__(self, chunk_size: int = 100, overlap: int = 20):
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        self._chunk_size = chunk_size
        self._overlap    = overlap

    def chunk(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []

        chunks = []
        start  = 0

        while start < len(words):
            end = start + self._chunk_size
            chunks.append(" ".join(words[start:end]))
            start += self._chunk_size - self._overlap

        return chunks