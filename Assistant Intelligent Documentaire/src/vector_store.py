# src/vector_store.py
"""
Création, sauvegarde et chargement d'un index FAISS.

Fonctions principales :
- create_faiss_index(embeddings, metadatas, save_dir)
- load_faiss_index(save_dir)
- search(index, query_embedding, k)

Structure :
data/vectorstore/
    ├── index.faiss
    └── metadatas.pkl
"""

from pathlib import Path
import faiss
import numpy as np
import pickle
from typing import List, Tuple, Dict


def create_faiss_index(
    embeddings: np.ndarray,
    metadatas: List[Dict],
    save_dir: str = "data/vectorstore"
):
    """
    embeddings: np.ndarray (n, dim)
    metadatas: list[dict] de longueur n
    save_dir: dossier de sauvegarde
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if embeddings.shape[0] != len(metadatas):
        raise ValueError("Le nombre d'embeddings et de metadatas doit être identique")

    n, dim = embeddings.shape

    # Normalisation → produit scalaire ≈ cosine similarity
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Sauvegarde
    faiss.write_index(index, str(save_dir / "index.faiss"))
    with open(save_dir / "metadatas.pkl", "wb") as f:
        pickle.dump(metadatas, f)

    print(f"Index FAISS sauvegardé dans : {save_dir}")
    return index


def load_faiss_index(save_dir: str = "data/vectorstore") -> Tuple[faiss.Index, List[Dict]]:
    save_dir = Path(save_dir)
    index_path = save_dir / "index.faiss"
    meta_path = save_dir / "metadatas.pkl"

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Index FAISS introuvable dans {save_dir}")

    index = faiss.read_index(str(index_path))

    with open(meta_path, "rb") as f:
        metadatas = pickle.load(f)

    return index, metadatas


def search(
    index: faiss.Index,
    query_embedding: np.ndarray,
    k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retourne les scores et indices des k documents les plus proches.
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    query_embedding = query_embedding.astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, k)
    return scores[0], indices[0]


# =========================
# TEST LOCAL
# =========================
if __name__ == "__main__":
    print("Test FAISS vector store...")

    X = np.random.randn(15, 384).astype("float32")
    metas = [{"chunk_id": i, "source": "test"} for i in range(15)]

    create_faiss_index(X, metas, "data/vectorstore")

    idx, metadatas = load_faiss_index("data/vectorstore")

    q = np.random.randn(1, 384).astype("float32")
    scores, ids = search(idx, q, k=3)

    print("Top results:")
    for i, s in zip(ids, scores):
        print(f"ID: {i}, Score: {s:.4f}, Meta: {metadatas[i]}")
