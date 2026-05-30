from core.config import GROQ_API_KEY, Tool_MODEL
from services.mistralai_client import mistral_client
from Model_Memory_store.models.schemas import RouteLLM
from services.GroqClient import groq_client

class RetrievalRouter:
    """Decides whether a query needs document retrieval at all."""

    SYSTEM_PROMPT = """
    You are a routing agent that decides whether a user query needs document retrieval.
    Return only a JSON object: {"retrieve": true} or {"retrieve": false}

    Rules:
    - true  → query needs external documents, files, or stored knowledge
    - false → query can be answered with general knowledge or reasoning
    - If unsure, return true.

    Output only valid JSON. No explanation, no extra text.
    """

    def __init__(self, groq_client: groq_client, model: str = Tool_MODEL):
        self._client = groq_client
        self._model  = model

    def should_retrieve(self, query: str) -> bool:
        response = self._client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": query}
            ],
            model=self._model,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name":   "RouteLLM",
                    "schema": RouteLLM.model_json_schema()
                }
            }
        )
        content = response.choices[0].message.content
        return RouteLLM.model_validate_json(content).retrieve

