# src/embeddings.py
"""
Génération d'embeddings avec Sentence-Transformers.
Classe EmbeddingsModel:
- encode(list_texts) -> np.ndarray
"""
import numpy as np
from typing import List


class EmbeddingsModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        model_name: modèle sentence-transformers
        device: 'cpu' ou 'cuda'
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Le package 'sentence-transformers' est requis. "
                "Installez-le avec: pip install sentence-transformers"
            ) from exc

        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Retourne un tableau numpy (n_texts, dim_emb)
        """
        embs = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        embs = np.asarray(embs, dtype=np.float32)
        return embs


if __name__ == "__main__":
    m = EmbeddingsModel()
    v = m.encode(["Bonjour le monde", "Comment ça va ?"])
    print("shape:", v.shape)
