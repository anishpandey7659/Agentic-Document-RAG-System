from groq import Groq
from config import Tool_MODEL,GROQ_API_KEY
from Model_Memory_store.models.schemas import RouteLLM

groq_client = Groq(api_key=GROQ_API_KEY)
system_prompt="""
You are a routing agent that decides whether a user query needs document retrieval.
Return only a JSON object:
{"retrieve": true}
or
{"retrieve": false}
Rules:
Set true if the query needs external documents, files, database, or specific stored knowledge.
Set false if the query can be answered using general knowledge, reasoning, or conversation.
If unsure, return true.
Output only valid JSON. No explanation, no extra text.
"""

def should_retrieve(query:str)->bool:
    response = groq_client.chat.completions.create(
            model=Tool_MODEL,
            messages=[
                {"role": "system", "content":system_prompt},
                {"role": "user", "content": query}
            ],
            response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "RouteLLM",
                "schema": RouteLLM.model_json_schema()
            }
        }
        )
    content = response.choices[0].message.content
    result = RouteLLM.model_validate_json(content)
    return result.retrieve



# query="Tell me what is in this document"
# result=should_retrieve(query)
# print(result)

# python -m agents.Orchestration.router.router_agent
