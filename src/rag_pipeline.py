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

try:
    from .embeddings import EmbeddingsModel
    from .vector_store import search, load_faiss_index
except ImportError:
    from embeddings import EmbeddingsModel
    from vector_store import search, load_faiss_index


# ============================
# PROMPT BUILDER
# ============================
def build_prompt(question: str, retrieved: List[Dict], instr: Optional[str] = None) -> str:
    header = instr or (
        "Tu es un assistant QA RAG en français. "
        "Réponds uniquement à partir du contexte fourni. "
        "Si l'information n'est pas dans le contexte, répond exactement: Je ne sais pas. "
        "Réponse courte (3 phrases max), sans inventer."
    )

    parts = [header, "\n---\nCONTEXTE RELEVANT:\n"]

    for i, r in enumerate(retrieved):
        src = r.get("meta", {})
        source = src.get("source", "unknown")
        page = src.get("page", "?")
        parts.append(f"Passage {i+1} (source: {source}, page: {page}):\n{r['text']}\n")

    parts.append("\nQUESTION:\n" + question + "\n")
    parts.append("\nREPONSE:")
    return "\n".join(parts)


def clean_generated_answer(text: str) -> str:
    cleaned = text.strip()
    for marker in [
        "\nHuman:",
        "\nAssistant:",
        "\nUser:",
        "\n---",
        "\nQUESTION:",
        "\nCONTEXTE",
    ]:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()
    return cleaned


# ============================
# LLM CALLABLE
# ============================
def llm_transformers_callable(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: int = -1,
    max_new_tokens: int = 192,
) -> Callable[[str], str]:

    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    gen = pipeline("text-generation", model=model, tokenizer=tok, device=device)

    def generate(prompt: str) -> str:
        out = gen(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
        return clean_generated_answer(out[0]["generated_text"])

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

    if index.ntotal == 0:
        print("ERREUR: L'index FAISS est vide (0 documents). Verifiez que vos PDF contiennent du texte selectionnable (pas des scans).")
        return "Je ne sais pas.", []

    # 2. Recherche FAISS
    scores, indices = search(index, q_emb, k=top_k)

    print(f"DEBUG: Index ntotal={index.ntotal}, Metadatas len={len(metadatas)}")
    print(f"DEBUG: FAISS indices={indices}")

    # 3. Reconstruction du contexte
    retrieved = []
    for rank, idx in enumerate(indices):
        if idx < 0 or idx >= len(metadatas):
            print(f"DEBUG: Ignored index {idx} (out of bounds or -1)")
            continue

        meta = metadatas[idx]

        text = meta.get("text", "").strip()
        if not text:
            print(f"DEBUG: Ignored index {idx} (empty text)")
            continue  # skip chunks vides

        retrieved.append({
            "text": text,
            "meta": meta,
            "score": float(scores[rank]),
        })

    if not retrieved:
        print("Aucun document pertinent trouve dans l'index (retrieved est vide).")
        return "Je ne sais pas.", []

    # 4. Prompt + generation
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
    llm = llm_transformers_callable(
        "Qwen/Qwen2.5-1.5B-Instruct", device=-1, max_new_tokens=192
    )

    question = "De quoi parle le document ?"

    ans, retrieved = answer_question(
        question, index, metadatas, embedder, llm, top_k=3
    )

    print("\n========================")
    print("REPONSE:\n", ans)
    print("\nCHUNKS RETROUVES:\n")
    for r in retrieved:
        print("Score:", r["score"])
        print(r["text"][:300], "...")
        print("-" * 80)
