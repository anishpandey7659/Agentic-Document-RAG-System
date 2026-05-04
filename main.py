from pipeline import retrieve_and_answer
from services import list_available_documents

if __name__ == "__main__":

    list_available_documents()

    answer = retrieve_and_answer(
        query="How is the recent Market of AI",
        top_k=5,
        show_chunks=True
    )
    print(f"\n[ANSWER]\n{answer}")
