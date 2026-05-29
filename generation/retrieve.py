"""
Retrieve the most relevant chunks from the Chroma collection for a question.

Mirror of the indexing step (embed_and_index.py) — these MUST match or the
vectors won't be comparable:
  - same model (multilingual-e5-base), on CPU
  - e5 task prefix: "query:" for questions (passages used "passage:")
  - normalized embeddings + cosine space (set on the collection)

Usage:
  python retrieve.py "Quels sont les chiffres du harcèlement scolaire ?"
  python retrieve.py "avis du Gouvernement" --k 5 --no-procedural
  python retrieve.py "amendements" --session CRSANR5L15S2022O1N165

Importable:
  from retrieve import Retriever
  r = Retriever()
  hits = r.search("…", k=5, where={"is_procedural": False})
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"
DB_PATH = "./chroma_db"
COLLECTION = "harcelement_scolaire"
QUERY_PREFIX = "query: "


@dataclass
class Hit:
    """One retrieved chunk with its provenance and similarity score."""
    rank: int
    score: float           # cosine similarity (higher = closer)
    text: str
    speaker: str
    role: str
    agenda_item: str
    date_iso: str
    session_uid: str
    is_procedural: bool
    chunk_id: str

    def header(self) -> str:
        who = self.speaker or "—"
        if self.role:
            who += f", {self.role}"
        tag = "  [procédural]" if self.is_procedural else ""
        return (f"#{self.rank}  (id: {self.chunk_id}) (score {self.score:.3f})  "
                f"{self.date_iso} — {self.agenda_item[:40]}{tag}\n    {who}")


class Retriever:
    def __init__(self, db_path: str = DB_PATH, collection: str = COLLECTION):
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        client = chromadb.PersistentClient(path=db_path)
        self.collection = client.get_collection(collection)

    def search(self, question: str, k: int = 5, where: dict | None = None) -> list[Hit]:
        # Embed the query with the e5 "query:" prefix, normalized to match the index.
        vec = self.model.encode(
            [QUERY_PREFIX + question], normalize_embeddings=True
        ).tolist()
        res = self.collection.query(
            query_embeddings=vec,
            n_results=k,
            where=where or None,           # metadata filter, e.g. {"is_procedural": False}
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Hit] = []
        ids = res["ids"][0]                # the full chunk_id used at indexing
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for i, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
            hits.append(Hit(
                rank=i + 1,
                score=1.0 - dist,          # Chroma returns cosine DISTANCE; similarity = 1 - d
                text=doc,
                speaker=meta.get("speaker", ""),
                role=meta.get("role", ""),
                agenda_item=meta.get("agenda_item", ""),
                date_iso=meta.get("date_iso", ""),
                session_uid=meta.get("session_uid", ""),
                is_procedural=bool(meta.get("is_procedural", False)),
                chunk_id=cid,              # full id, e.g. CRSANR5L15S2022O1N080_2703323_0
            ))
        return hits


def main() -> None:
    ap = argparse.ArgumentParser(description="Search the debate corpus.")
    ap.add_argument("question", help="the question to search for")
    ap.add_argument("--k", type=int, default=5, help="number of results")
    ap.add_argument("--no-procedural", action="store_true",
                    help="exclude procedural notes (votes, applause, session open/close)")
    ap.add_argument("--session", default=None,
                    help="restrict to one session uid (e.g. CRSANR5L15S2022O1N080)")
    args = ap.parse_args()

    where: dict = {}
    if args.no_procedural:
        where["is_procedural"] = False
    if args.session:
        where["session_uid"] = args.session

    r = Retriever()
    hits = r.search(args.question, k=args.k, where=where or None)

    print(f'\nQuestion: {args.question}\n' + "=" * 70)
    for h in hits:
        print(h.header())
        print(f"    {h.text[:280]}{'…' if len(h.text) > 280 else ''}\n")


if __name__ == "__main__":
    main()