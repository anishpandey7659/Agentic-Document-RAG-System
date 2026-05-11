# pipeline/extractor.py

import os
import re
import PyPDF2
import docx

SUPPORTED = {".pdf", ".docx", ".txt"}


class Extractor:
    """Extracts and cleans raw text from PDF, DOCX, and TXT files."""

    def extract(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()

        if ext not in SUPPORTED:
            raise ValueError(f"Unsupported format '{ext}'. Use: {SUPPORTED}")

        extractor = {
            ".pdf":  self._from_pdf,
            ".docx": self._from_docx,
            ".txt":  self._from_txt,
        }
        return extractor[ext](file_path)

    def clean(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)        
        text = re.sub(r'[^\x00-\x7F]+', '', text)  
        return text.strip()

    def _from_pdf(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join(page.extract_text() or "" for page in reader.pages)

    def _from_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)

    def _from_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()