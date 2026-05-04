import uvicorn
from fastapi import FastAPI
from api.upload_file import router  

app = FastAPI(title="File Upload API")

app.include_router(router)

# Run using uvicorn api.routes:app --reload