from typing import List, Dict
from pydantic import BaseModel
from dataclasses import dataclass, asdict



@dataclass
class DocumentAgent:
    doc_id: str
    summary: str
    keywords: List[str]
    vector_db: str
    metadata: Dict

class DocumentSummary(BaseModel):
    summary: str
    keywords: List[str]


class RouteDecision(BaseModel):
    doc_ids: List[str]