# src/build_index.py
from pathlib import Path

try:
    from .ingestion import extract_text_from_pdf
    from .chunking import split_text_to_chunks
    from .embeddings import EmbeddingsModel
    from .vector_store import create_faiss_index
except ImportError:
    from ingestion import extract_text_from_pdf
    from chunking import split_text_to_chunks
    from embeddings import EmbeddingsModel
    from vector_store import create_faiss_index

DATA_DIR = Path("data")
RAW_PDFS = DATA_DIR / "raw_pdfs"
VECTOR_DIR = DATA_DIR / "vectorstore"

embedder = EmbeddingsModel()

all_texts = []
metadatas = []

print("Indexation des PDF...")

for pdf_path in RAW_PDFS.glob("*.pdf"):
    print(f"Lecture : {pdf_path.name}")
    text = extract_text_from_pdf(pdf_path)
    
    if not text.strip():
        print(f"AVERTISSEMENT: Aucun texte trouvé dans {pdf_path.name} (PDF scanné ou vide ?).")
        continue

    chunks = split_text_to_chunks(text)

    for c in chunks:
        if not c["text"].strip():
            continue  # On ignore les chunks vides

        all_texts.append(c["text"])
        metadatas.append({
            "chunk_id": c["id"],
            "source": pdf_path.name,
            "text": c["text"]
        })

print(f"Total chunks: {len(all_texts)}")

if len(all_texts) == 0:
    print("ERREUR: Aucun chunk valide n'a été généré. Vérifiez vos PDF.")
    exit(1)

print("Génération embeddings...")
embeddings = embedder.encode(all_texts)

print("Création index FAISS...")
create_faiss_index(embeddings, metadatas, VECTOR_DIR)

print("Index FAISS créé avec succès !")
