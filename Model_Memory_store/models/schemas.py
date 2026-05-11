from typing import List, Dict
from pydantic import BaseModel, Field
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


class RouteLLM(BaseModel):
    retrieve: bool = Field(
        default=False,
        description="Whether to retrieve documents from the vector database or not."
    )