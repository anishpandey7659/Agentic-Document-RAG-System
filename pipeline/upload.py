import os
import uuid
from services import (create_document_agent,register_document_agent,generate_embeddings,embed_both,embed_dense,store_in_pinecone,save_document_embedding)
from pipeline import (chunk_text,clean_text,extract_text,summarize_and_extract_keywords)
from Model_Memory_store.models import DocumentAgent
from config import INDEX_NAME


def upload_document_pipeline(file_path: str,index_name:str,domain:str) -> DocumentAgent:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        doc_id    = f"doc-{uuid.uuid4().hex[:8]}"
        file_name = os.path.basename(file_path)   # ← extract just filename

        print(f"[INFO] Starting pipeline for '{file_name}' → doc_id: {doc_id}")

        print("[INFO] Extracting text...")
        raw_text = extract_text(file_path)

        print("[INFO] Cleaning text...")
        cleaned_text = clean_text(raw_text)

        if not cleaned_text:
            raise ValueError("Document appears to be empty after cleaning.")

        print("[INFO] Chunking text...")
        chunks = chunk_text(cleaned_text)
        print(f"[INFO] Created {len(chunks)} chunks.")
        # print("\n",chunks[2])
        # print("\n\n\n",chunks[3])

        print("[INFO] Generating embeddings with Mistral...")
        embeddings = embed_both(chunks, input_type="passage")



        print("[INFO] Storing in Pinecone...")
        vector_db = store_in_pinecone(doc_id, chunks,embeddings,index_name,file_name,domain)

        print("[INFO] Summarizing document...")
        summary_data = summarize_and_extract_keywords(cleaned_text)
        
        document_text = f"""
        Title: {file_name}
        Domain: {domain}
        Summary: {summary_data["summary"]}
        Keywords: {", ".join(summary_data["keywords"])}
        """
        embed_document=embed_dense(document_text, input_type="passage")

        agent = create_document_agent(doc_id, summary_data, vector_db, file_name)
        save_document_embedding(doc_id,embed_document)
        register_document_agent(agent)

        print("[INFO] Pipeline complete.")
        return agent

    except Exception as e:
        print(f"[ERROR] Pipeline failed for '{file_path}': {e}")
        raise


# python -m pipeline.upload

if __name__ == "__main__":
    file_path = "/home/anish/Documents/Agentic-RAG/data/text_files/agentic_ai.txt"
    domain="Agentic Ai"
    agent = upload_document_pipeline(file_path,INDEX_NAME,domain)

    print("\nDocument Agent Created:")
    print(f"  doc_id     : {agent.doc_id}")
    print(f"  file_name  : {agent.metadata['file_name']}")
    print(f"  summary    : {agent.summary}")
    print(f"  keywords   : {agent.keywords}")
    print(f"  vector DB  : {agent.vector_db}")
    print(f"  embed model: {agent.metadata['embed_model']}")