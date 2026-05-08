from pipeline import retrieve_and_answer
from fastapi.responses import JSONResponse
from fastapi import APIRouter

chat = APIRouter()


@chat.post("/chat",summary="LLM answing the question from documents")
async def answer_llm(message:str):
    
    answer = retrieve_and_answer(
        query=message,
        top_k=5,
        show_chunks=True
    )
    return JSONResponse(status_code=201, content={"question":message,"results": answer})