"""
Embed the chunks and index them in a persistent Chroma collection.

Design choices:
  - We compute embeddings ourselves with a local multilingual model
    (intfloat/multilingual-e5-base) and hand them to Chroma, rather than letting
    Chroma embed internally. This keeps full control of the model and runs on
    CPU, leaving GPU VRAM for the generation LLM.
  - We embed `embedding_text` (context-prefixed) but store the raw `text` as the
    Chroma "document", so retrieved passages display cleanly and can be cited.
  - Chroma metadata must be scalar (str/int/float/bool) and non-null, so we
    sanitise each chunk's metadata before insertion.

Run locally:  python embed_and_index.py chunks.jsonl
Requires:     pip install chromadb sentence-transformers
"""

from __future__ import annotations

from pathlib import Path
import json

import chromadb
from sentence_transformers import SentenceTransformer

# --- Configuration ---------------------------------------------------------
MODEL_NAME = "intfloat/multilingual-e5-base"   # strong FR support, CPU-friendly
DB_PATH = "./chroma_db"
COLLECTION = "harcelement_scolaire"
BATCH_SIZE = 64

# e5 models expect a task prefix: "passage:" for indexed docs, "query:" for
# search queries. Forgetting this measurably hurts retrieval quality.
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

# Metadata fields we keep for filtering at retrieval time.
META_FIELDS = ("speaker", "role", "agenda_item", "is_procedural",
               "session_uid", "date_iso", "syceron_id", "part", "n_parts")


def load_chunks(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).open(encoding="utf-8")]


def context_prefix(c: dict) -> str:
    who = c["speaker"] or "—"
    if c["role"]:
        who += f", {c['role']}"
    iso = c["date_iso"]
    date = f"{iso[:4]}-{iso[4:6]}-{iso[6:8]}" if len(iso) == 8 else iso
    bits = [f"Séance du {date}"]
    if c["agenda_item"]:
        bits.append(c["agenda_item"])
    return f"[{' — '.join(bits)}] {who} :"


def embedding_text(c: dict) -> str:
    """Context-prefixed text that actually gets embedded (with e5 prefix)."""
    return PASSAGE_PREFIX + f"{context_prefix(c)} {c['text']}"


def clean_metadata(c: dict) -> dict:
    """Keep only scalar, non-null metadata fields (Chroma requirement)."""
    out = {}
    for k in META_FIELDS:
        v = c.get(k)
        if v is None:
            v = ""              # Chroma rejects None; use empty string
        out[k] = v
    return out


def main(chunks_path: str) -> None:
    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading embedding model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")   # downloads on first run

    print("Embedding (this runs on CPU; a few minutes is normal)...")
    texts = [embedding_text(c) for c in chunks]
    embeddings = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True,
        normalize_embeddings=True,            # cosine similarity via dot product
    ).tolist()

    client = chromadb.PersistentClient(path=DB_PATH)
    # Fresh start each run, so re-indexing is reproducible.
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},    # match normalized embeddings
    )

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]   # raw text, for display/citation
    metadatas = [clean_metadata(c) for c in chunks]

    for i in range(0, len(chunks), BATCH_SIZE):
        sl = slice(i, i + BATCH_SIZE)
        collection.add(
            ids=ids[sl], documents=documents[sl],
            embeddings=embeddings[sl], metadatas=metadatas[sl],
        )
    print(f"\nIndexed {collection.count()} chunks into "
          f"'{COLLECTION}' at {DB_PATH}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "chunks.jsonl")