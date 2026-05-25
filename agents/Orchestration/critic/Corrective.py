from services.GroqClient import groq_client
from config import Tool_MODEL
from Model_Memory_store.models.schemas import GradeDocuments
from config import TAVILY_API_KEY
from agents.ExternalAccess.tavily import web_search

SYSTEM_PROMPT = """
You are a strict relevance grader for a RAG system.
Task:
Decide whether a retrieved document can directly help answer the question.
Rules:
- Return only a JSON object matching the provided schema.
- binary_score must be true only if the document directly answers or strongly supports the question.
- Keyword overlap alone is NOT sufficient.
- Be strict and conservative (prefer false positives).
- If unsure, return false.
- Do not explain.
"""

class CorrectiveCritic:
    def __init__(self, groq_client, model: str = Tool_MODEL):
        self._client = groq_client
        self._model: str = model

    def grade_relevance(self, question: str, document: str) -> bool:
        prompt = f"""
        Question:
        {question}

        Document:
        {document}
        """
        response = self._client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            model=self._model,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name":   "GradeDocuments",
                    "schema": GradeDocuments.model_json_schema()
                }
            }
        )
        content = response.choices[0].message.content
        return GradeDocuments.model_validate_json(content).binary_score

    def Grade_router(self, question: str, document: str):
        grade = self.grade_relevance(question, document)
        if grade:
            return {'question': question, 'document': document, 'grade': grade}
        else:
            web_search_results = web_search(question)
            return {'question': question, 'document': web_search_results, 'grade': grade}
    
corr=CorrectiveCritic(groq_client)
print(corr.Grade_router("What is AIRA 2","Newtons predecessor with improved performance and new features."))

# python -m agents.Orchestration.critic.Corrective