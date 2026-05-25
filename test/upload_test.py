from core.dependencies import (
    extractor,chunker,summarizer,
    embedder,embedding_store,vector_store,
    agent_factory,memory_store,
)
from pipeline import UploadPipeline
from core.config import INDEX_NAME

upload_pipeline = UploadPipeline(
    extractor       = extractor,
    chunker         = chunker,
    summarizer      = summarizer,
    embedder        = embedder,
    embedding_store = embedding_store,
    vector_store    = vector_store,
    agent_factory   = agent_factory,
    memory_store    = memory_store,
)

if __name__ == "__main__":
    agent = upload_pipeline.run(
        file_path  = "/home/anish/Documents/Agentic-RAG/data/text_files/Deep Agent in LangChain.txt",
        index_name = INDEX_NAME,
        domain     = "Langchain Deep Agent"
    )
    print(f"\n  doc_id  : {agent.doc_id}")
    print(f"  file    : {agent.metadata['file_name']}")
    print(f"  summary : {agent.summary}")
    print(f"  keywords: {agent.keywords}")
    print(f"  vector  : {agent.vector_db}")



# python -m test.upload_test