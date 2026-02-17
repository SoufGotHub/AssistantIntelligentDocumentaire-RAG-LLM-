# src/rag_pipeline.py
"""
Pipeline RAG complet:
- encodage question
- recherche FAISS
- construction du contexte
- appel LLM

Fonctions:
- build_prompt
- answer_question
"""

from typing import List, Dict, Tuple, Callable, Optional
import numpy as np
from .embeddings import EmbeddingsModel
from .vector_store import search, load_faiss_index


# ============================
# PROMPT BUILDER
# ============================
def build_prompt(question: str, retrieved: List[Dict], instr: Optional[str] = None) -> str:
    header = instr or (
        "Tu es un assistant qui répond uniquement à partir du contexte fourni. "
        "Si l'information n'est pas dans le contexte, répond 'Je ne sais pas'."
    )

    parts = [header, "\n---\nCONTEXTE RELEVANT:\n"]

    for i, r in enumerate(retrieved):
        src = r.get("meta", {})
        source = src.get("source", "unknown")
        page = src.get("page", "?")
        parts.append(f"Passage {i+1} (source: {source}, page: {page}):\n{r['text']}\n")

    parts.append("\nQUESTION:\n" + question + "\n")
    parts.append("\nRÉPONSE:")
    return "\n".join(parts)


# ============================
# LLM CALLABLE
# ============================
def llm_transformers_callable(
    model_name: str = "gpt2",
    device: int = -1,
    max_new_tokens: int = 256
) -> Callable[[str], str]:

    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    gen = pipeline("text-generation", model=model, tokenizer=tok, device=device)

    def generate(prompt: str) -> str:
        out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return out[0]["generated_text"]

    return generate


# ============================
# MAIN RAG FUNCTION
# ============================
def answer_question(
    question: str,
    index,
    metadatas: List[Dict],
    embedder: EmbeddingsModel,
    llm_callable: Callable[[str], str],
    top_k: int = 4,
) -> Tuple[str, List[Dict]]:

    # 1. Encode question
    q_emb = embedder.encode([question])[0]

    # 2. Recherche FAISS
    scores, indices = search(index, q_emb, k=top_k)

    # 3. Reconstruction du contexte
    retrieved = []
    for rank, idx in enumerate(indices):
        if idx < 0 or idx >= len(metadatas):
            continue

        meta = metadatas[idx]

        text = meta.get("text", "").strip()
        if not text:
            continue   # skip chunks vides

        retrieved.append({
            "text": text,
            "meta": meta,
            "score": float(scores[rank])
        })

    if not retrieved:
        return "Je ne sais pas.", []

    # 4. Prompt + génération
    prompt = build_prompt(question, retrieved)
    answer = llm_callable(prompt)

    return answer, retrieved


# ============================
# TEST LOCAL
# ============================
if __name__ == "__main__":
    try:
        index, metadatas = load_faiss_index("data/vectorstore")
    except FileNotFoundError:
        print("Index FAISS introuvable. Lance d'abord la phase d'indexation.")
        raise SystemExit(1)

    embedder = EmbeddingsModel()
    llm = llm_transformers_callable("gpt2", device=-1, max_new_tokens=128)

    question = "De quoi parle le document ?"

    ans, retrieved = answer_question(
        question, index, metadatas, embedder, llm, top_k=3
    )

    print("\n========================")
    print("RÉPONSE:\n", ans)
    print("\nCHUNKS RETROUVÉS:\n")
    for r in retrieved:
        print("Score:", r["score"])
        print(r["text"][:300], "...")
        print("-" * 80)
