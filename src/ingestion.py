# src/ingestion.py
"""
Extraction automatique du texte depuis tous les PDF du dossier raw_pdfs.
Sauvegarde les fichiers .txt correspondants dans processed.
"""

from pathlib import Path

try:
    import pymupdf as fitz  # package: pymupdf
except ImportError:
    import fitz  # fallback for environments exposing PyMuPDF as fitz

try:
    from .utils import RAW_PDF_DIR, PROCESSED_DIR
except ImportError:
    from utils import RAW_PDF_DIR, PROCESSED_DIR


def extract_pages_from_pdf(pdf_path: Path):
    """
    Retourne une liste de pages (texte brut) du PDF.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"{pdf_path} introuvable")

    doc = fitz.open(str(pdf_path))
    pages = [page.get_text("text") for page in doc]
    doc.close()

    return pages


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Concatène toutes les pages en un seul texte.
    """
    pages = extract_pages_from_pdf(pdf_path)
    return "\n\n".join(pages)


def save_text(text: str, out_path: Path):
    """
    Sauvegarde le texte extrait dans un fichier .txt.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def process_all_pdfs():
    """
    Traite automatiquement tous les PDF présents dans raw_pdfs/.
    """
    pdf_files = list(RAW_PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("Aucun fichier PDF trouvé dans raw_pdfs/.")
        return

    print(f"{len(pdf_files)} PDF(s) trouvé(s). Démarrage extraction...\n")

    for pdf_path in pdf_files:
        out_txt = PROCESSED_DIR / f"{pdf_path.stem}.txt"

        print(f"Extraction : {pdf_path.name}")
        text = extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            print(f"ATTENTION : Le texte extrait de {pdf_path.name} est vide. Ce fichier sera ignoré.")
            
        save_text(text, out_txt)
        print(f"   Sauvegardé : {out_txt.name}\n")

    print("Extraction terminée pour tous les fichiers.")


if __name__ == "__main__":
    process_all_pdfs()
