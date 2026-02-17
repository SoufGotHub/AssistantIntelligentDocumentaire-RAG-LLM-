# src/utils.py
"""
Gestion centralisée des chemins + fonctions utilitaires communes.
"""

from pathlib import Path
import json
from typing import Any

# ========================
# PATHS DU PROJET
# ========================

# Racine du projet (dossier principal)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Dossier data
DATA_DIR = ROOT_DIR / "data"

# Sous-dossiers data
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

# Création automatique des dossiers
RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


# ========================
# FONCTIONS UTILITAIRES
# ========================

def save_json(obj: Any, path: Path):
    """Sauvegarde un objet Python en JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    """Charge un fichier JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    """Nettoyage simple du texte."""
    return " ".join(text.split())


# ========================
# TEST RAPIDE
# ========================

if __name__ == "__main__":
    print("ROOT_DIR:", ROOT_DIR)
    print("RAW_PDF_DIR:", RAW_PDF_DIR)
    print("PROCESSED_DIR:", PROCESSED_DIR)
    print("VECTORSTORE_DIR:", VECTORSTORE_DIR)
