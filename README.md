# AssistantIntelligentDocumentaire-RAG-LLM-

Simple RAG pipeline for PDF documents:
- extract text from PDF files
- split into chunks
- generate embeddings
- index with FAISS
- answer questions from retrieved context

## Project Structure

```text
src/
  ingestion.py
  chunking.py
  embeddings.py
  vector_store.py
  build_index.py
  rag_pipeline.py
data/
  raw_pdfs/
  processed/
  vectorstore/
requirements.txt
Dockerfile
docker-compose.yml
```

## Prerequisites

- Python 3.11+ (3.13 also works in this project)
- pip
- Optional: Docker + Docker Compose

## Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Run Locally

1. Put your PDF files in `data/raw_pdfs/`.
2. Extract text:

```bash
python src/ingestion.py
```

3. Build FAISS index:

```bash
python src/build_index.py
```

4. Run RAG pipeline:

```bash
python src/rag_pipeline.py
```

## Run With Docker

Build image:

```bash
docker compose build
```

Build index:

```bash
docker compose run --rm app python src/build_index.py
```

Run pipeline:

```bash
docker compose run --rm app python src/rag_pipeline.py
```

## Notes

- The default generation model is `gpt2` in `src/rag_pipeline.py`.
- You may see Hugging Face warnings about unauthenticated requests; setting `HF_TOKEN` is optional but can improve download limits.
