import os
import shutil
import logging
from pathlib import Path
from fastapi.responses import JSONResponse
from test.upload_test import upload_pipeline,INDEX_NAME
from config import PINECONE_API_KEY,MEMORY_FILE
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from services import AgentMemoryStore,EmbeddingStore,PineconeClient

memory_agent=AgentMemoryStore(memory_file=MEMORY_FILE)
emb_store=EmbeddingStore()

pc = PineconeClient()

logging.basicConfig(level=logging.INFO,format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


router = APIRouter() 

BASE_DIR = "/home/anish/Documents/Agentic-RAG/data"
EXT_FOLDER_MAP = {
    "pdf": "pdf",
    "docx": "docx",
    "txt": "text_files"
}


def get_save_path(filename: str) -> Path:
    """Resolve the destination folder based on file extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    folder = EXT_FOLDER_MAP.get(ext)
    if folder is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Allowed: {list(EXT_FOLDER_MAP.keys())}"
        )
    dest_dir = Path(BASE_DIR) / folder
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / filename


def ingest_to_pinecone(file_path: str, filename: str, domain: str):
    """
        Background task: called after the HTTP response is already sent.
        Run
    """
    try:
        logger.info(f"[Pinecone] Starting ingestion for: {filename}")
        upload_pipeline.run(file_path, index_name=INDEX_NAME, domain=domain)
        logger.info(f"[Pinecone] Successfully ingested: {filename}")

    except Exception as e:
        logger.error(f"[Pinecone] Ingestion failed for {filename}: {e}", exc_info=True)
        try:
            Path(file_path).unlink(missing_ok=True)
            logger.info(f"[Pinecone] Cleaned up file: {filename}")
        except Exception as cleanup_error:
            logger.error(f"[Pinecone] Cleanup failed for {filename}: {cleanup_error}")


@router.post("/upload", summary="Upload a single file + ingest to Pinecone")
async def upload_file(file: UploadFile, background_tasks: BackgroundTasks,domain:str):
    save_path = get_save_path(file.filename)

    if save_path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"File '{file.filename}' already exists."
        )

    try:
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        await file.close()

    # ✅ Fires AFTER response is sent — user doesn't wait for Pinecone
    background_tasks.add_task(ingest_to_pinecone, str(save_path), file.filename,domain)

    return JSONResponse(
        status_code=201,
        content={
            "message": "File uploaded successfully. Pinecone ingestion started in background.",
            "filename": file.filename,
            "saved_to": str(save_path),
        },
    )


@router.post("/upload/multiple", summary="Upload multiple files + ingest to Pinecone")
async def upload_multiple_files(files: list[UploadFile], background_tasks: BackgroundTasks,domain:list[str]):
    results = []
    for file in files:
        try:
            save_path = get_save_path(file.filename)
            if save_path.exists():
                results.append({"filename": file.filename, "status": "skipped", "reason": "already exists"})
                continue
            with save_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # ✅ One background task queued per file
            background_tasks.add_task(ingest_to_pinecone, str(save_path), file.filename,domain)

            results.append({
                "filename": file.filename,
                "status": "uploaded",
                "saved_to": str(save_path),
                "pinecone": "ingestion queued"
            })
        except HTTPException as e:
            results.append({"filename": file.filename, "status": "error", "reason": e.detail})
        finally:
            await file.close()

    return JSONResponse(status_code=207, content={"results": results})


@router.get("/files", summary="List all uploaded files grouped by type")
def list_files():
    grouped = {}
    for ext, folder in EXT_FOLDER_MAP.items():
        folder_path = Path(BASE_DIR) / folder
        if folder_path.exists():
            grouped[ext] = [f.name for f in folder_path.iterdir() if f.is_file()]
        else:
            grouped[ext] = []
    return grouped


@router.delete("/files/{filename}", summary="Delete an uploaded file")
def delete_file(filename: str):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    folder = EXT_FOLDER_MAP.get(ext)

    if folder is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'."
        )
    
    file_path = Path(get_save_path(filename))

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File '{filename}' not found."
        )
    memory = memory_agent.load_all()
    doc_id_to_delete = None

    for doc_id, data in memory.items():
        if data.get("metadata", {}).get("file_name") == filename:
            doc_id_to_delete = doc_id
            break

    if doc_id_to_delete:
        memory_agent.delete(
            doc_id_to_delete,
            source_name=filename,
            pinecone_client=pc,
            emb_store=emb_store,
        )

    file_path.unlink()

    return {
        "message": f"File '{filename}' deleted successfully.",
        "deleted_agent": doc_id_to_delete
    }