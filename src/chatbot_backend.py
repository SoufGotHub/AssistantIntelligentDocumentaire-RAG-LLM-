"""
Backend service for the web chatbot.
Keeps RAG orchestration isolated from the UI layer.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

from embeddings import EmbeddingsModel
from rag_pipeline import answer_question, llm_transformers_callable
from vector_store import load_faiss_index


@dataclass
class ChatbotConfig:
    vector_dir: str = "data/vectorstore"
    top_k: int = 4
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_max_new_tokens: int = 192
    device: int = -1


class RAGChatbot:
    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.index, self.metadatas = load_faiss_index(config.vector_dir)
        self.embedder = EmbeddingsModel()
        self.llm = llm_transformers_callable(
            model_name=config.llm_model,
            device=config.device,
            max_new_tokens=config.llm_max_new_tokens,
        )

    def ask(self, question: str, top_k: int | None = None) -> Dict[str, Any]:
        k = top_k if top_k is not None else self.config.top_k
        answer, retrieved = answer_question(
            question=question,
            index=self.index,
            metadatas=self.metadatas,
            embedder=self.embedder,
            llm_callable=self.llm,
            top_k=k,
        )
        return {"answer": answer, "retrieved": retrieved}

    @staticmethod
    def format_sources(retrieved: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        sources: List[Dict[str, str]] = []
        for item in retrieved:
            meta = item.get("meta", {})
            text = item.get("text", "")
            preview = " ".join(text.split())[:260]
            sources.append(
                {
                    "source": str(meta.get("source", "unknown")),
                    "chunk_id": str(meta.get("chunk_id", "?")),
                    "preview": preview,
                    "score": f"{item.get('score', 0.0):.4f}",
                }
            )
        return sources
