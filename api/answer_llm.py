from test.retrive_test import rag
from fastapi.responses import JSONResponse
from fastapi import APIRouter

chat = APIRouter()


@chat.post("/chat",summary="LLM answing the question from documents")
async def answer_llm(message:str):
    
    answer = rag.run(
        query=message
    )
    return JSONResponse(status_code=201, content={"question":message,"results": answer})