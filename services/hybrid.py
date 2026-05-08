from services import generate_embeddings,search_index
from pinecone_text.sparse import BM25Encoder
from .embedder import embed_both


def hybrid_score_norm(dense: list, sparse: dict, alpha: float):
    """
    Blend dense and sparse vectors using a convex combination:
        alpha * dense + (1 - alpha) * sparse

    Args:
        dense  : list of floats — the dense embedding
        sparse : dict with 'indices' and 'values' keys
        alpha  : float in [0, 1]
                 1.0 = pure semantic (dense only)
                 0.0 = pure keyword  (sparse only)
                 0.75 = recommended starting point
    """
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")

    scaled_dense = [v * alpha for v in dense]
    scaled_sparse = {
        "indices": sparse["indices"],
        "values":  [v * (1 - alpha) for v in sparse["values"]]
    }
    return scaled_dense, scaled_sparse




# python -m services.hybrid