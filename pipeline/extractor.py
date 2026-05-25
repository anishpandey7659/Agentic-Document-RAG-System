# pipeline/extractor.py

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Callable

import fitz  # PyMuPDF
import docx
from docx.table import Table


class ExtractionError(Exception):
    """Raised when text extraction fails for a supported file."""


class Extractor:
    """
    Extracts and cleans text from PDF, DOCX, and TXT files
    for use in RAG pipelines.

    Supported formats: .pdf, .docx, .txt

    Usage:
        extractor = Extractor()
        text = extractor.extract("document.pdf")
    """

    _EXTRACTORS: dict[str, str] = {
        ".pdf": "_from_pdf",
        ".docx": "_from_docx",
        ".txt": "_from_txt",
    }

    @classmethod
    def supported_formats(cls) -> frozenset[str]:
        return frozenset(cls._EXTRACTORS)

    def extract(self, file_path: str | Path) -> str:
        """
        Extract and clean text from a file.

        Args:
            file_path: Path to the file to extract.

        Returns:
            Cleaned plain text string.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
            ExtractionError: If extraction fails.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        method_name = self._EXTRACTORS.get(ext)

        if method_name is None:
            raise ValueError(
                f"Unsupported format '{ext}'. "
                f"Supported: {sorted(self.supported_formats())}"
            )

        extractor: Callable[[Path], str] = getattr(self, method_name)

        try:
            raw_text = extractor(path)
        except (ExtractionError, FileNotFoundError):
            raise
        except Exception as exc:
            raise ExtractionError(
                f"Failed to extract text from '{path}': {exc}"
            ) from exc

        return self.clean(raw_text)

    def clean(self, text: str) -> str:
        """
        Clean extracted text while preserving semantic document structure.

        Steps (order matters):
          1. Normalize unicode to NFKC form.
          2. Normalize line endings to \\n.
          3. Collapse runs of spaces/tabs to a single space per line.
          4. Trim trailing whitespace from each line.
          5. Collapse 3+ consecutive blank lines down to 2.
          6. Strip leading/trailing whitespace from the full string.
        """
        # 1. Unicode normalization (must happen before whitespace cleanup)
        text = unicodedata.normalize("NFKC", text)
        # 2. Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # 3. Collapse inline whitespace
        text = re.sub(r"[ \t]+", " ", text)
        # 4. Trim trailing spaces from each line
        text = "\n".join(line.rstrip() for line in text.splitlines())
        # 5. Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 6. Strip
        return text.strip()

    def _from_pdf(self, file_path: Path) -> str:
        """
        Extract text from a PDF using PyMuPDF.
        Each page is stripped before joining to avoid blank-line
        accumulation at page boundaries.
        """
        pages: list[str] = []
        with fitz.open(file_path) as pdf:
            for page in pdf:
                page_text = page.get_text().strip()
                if page_text:
                    pages.append(page_text)
        return "\n\n".join(pages)

    def _from_docx(self, file_path: Path) -> str:
        """
        Extract text from a DOCX file, preserving both paragraphs
        and table content (which python-docx's .paragraphs skips).

        Iterates over the document body in document order so that
        tables appear in their correct position relative to paragraphs.
        """
        doc = docx.Document(file_path)
        parts: list[str] = []

        for block in doc.element.body:
            tag = block.tag.split("}")[-1]  # strip XML namespace

            if tag == "p":
                text = block.text.strip()
                if text:
                    parts.append(text)

            elif tag == "tbl":
                table = Table(block, doc)
                for row in table.rows:
                    row_text = "\t".join(
                        cell.text.strip() for cell in row.cells
                    )
                    if row_text.strip():
                        parts.append(row_text)

        return "\n".join(parts)

    def _from_txt(self, file_path: Path) -> str:
        """
        Read a plain-text file.

        Attempts UTF-8 first; falls back to Latin-1 (which never
        raises on arbitrary bytes) if the file is not valid UTF-8.
        """
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return file_path.read_text(encoding="latin-1")