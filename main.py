import uvicorn
from fastapi import FastAPI
from api import chat,router  
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="File Upload API")



origins = [
    "http://localhost:5173",  # React app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   # 👈 IMPORTANT
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(chat)

# Run using uvicorn main:app --reload