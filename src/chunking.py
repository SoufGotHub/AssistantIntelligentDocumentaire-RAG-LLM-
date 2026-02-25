# src/chunking.py
"""
Découpage du texte en chunks avec overlap.

Fonction principale:
- split_text_to_chunks(text, chunk_size=1000, overlap=200)

Retour:
List[Dict] avec la structure :
{
    "id": int,
    "text": str,
    "start": int,
    "end": int
}
"""

from typing import List, Dict


def split_text_to_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[Dict]:
    """
    Découpe un texte en morceaux (chunks) avec chevauchement (overlap).

    Args:
        text (str): texte complet
        chunk_size (int): taille maximale d’un chunk
        overlap (int): nombre de caractères de chevauchement

    Returns:
        List[Dict]: liste de chunks
    """
    if not text.strip():
        return []

    if chunk_size <= overlap:
        raise ValueError("chunk_size doit être strictement supérieur à overlap")

    chunks = []
    start = 0
    text_length = len(text)
    chunk_id = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "start": start,
                "end": end
            })
            chunk_id += 1

        if end >= text_length:
            break

        start = end - overlap
        if start < 0:
            start = 0

    return chunks


# ============================
# TEST LOCAL
# ============================
if __name__ == "__main__":
    sample = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 200
    chunks = split_text_to_chunks(sample, chunk_size=300, overlap=50)

    print(f"Nombre de chunks: {len(chunks)}")
    print("Exemple chunk:\n", chunks[0])
