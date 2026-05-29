"""
Retrieval with cross-encoder reranking.

Pipeline:
  1. Use the dense retriever (retrieve.py) to fetch a wider pool of candidates
     (default n_candidates=20) — the bi-encoder e5 is fast but its scores tend
     to cluster, so it's an imperfect *ranker* even when it's a good *recaller*.
  2. Re-score every (question, chunk) pair with a cross-encoder, which encodes
     them JOINTLY rather than separately, capturing fine-grained matches the
     bi-encoder cannot see.
  3. Sort by the new score, return the top k.

Interface is intentionally identical to Retriever in retrieve.py, so
generate.py and run_eval.py can switch backends by changing one import.

Usage:
  python retrieve_reranked.py "Combien de décès attribués au harcèlement ?"
  python retrieve_reranked.py "..." --k 5 --candidates 20

Importable:
  from retrieve_reranked import RerankedRetriever
  r = RerankedRetriever()
  hits = r.search("...", k=5)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import argparse

from sentence_transformers import CrossEncoder

from .retrieve import Retriever, Hit  # reuse the dense retriever and Hit type

# Multilingual cross-encoder. bge-reranker-base is widely used, supports French,
# and is small enough to run on CPU (~280M params).
RERANKER_MODEL = "BAAI/bge-reranker-base"

# How many candidates to ask the dense retriever for, before reranking.
# 20 is a common default: wide enough to recover good chunks the bi-encoder
# ranked too low, narrow enough that 20 cross-encoder passes stay cheap.
DEFAULT_N_CANDIDATES = 20


class RerankedRetriever:
    """Dense retrieve + cross-encoder rerank, with the same interface as Retriever."""

    def __init__(self,
                 n_candidates: int = DEFAULT_N_CANDIDATES,
                 reranker_model: str = RERANKER_MODEL) -> None:
        self.n_candidates = n_candidates
        # Reuse the existing dense retriever — no duplication of indexing logic.
        self.dense = Retriever()
        # CrossEncoder loads on CPU by default; explicit for clarity.
        self.reranker = CrossEncoder(reranker_model, device="cpu")

    def search(self, question: str, k: int = 5,
               where: dict | None = None) -> list[Hit]:
        # 1. Wide dense recall.
        candidates = self.dense.search(question, k=self.n_candidates, where=where)
        if not candidates:
            return []

        # 2. Cross-encoder reranking: score every (question, chunk_text) pair.
        # NOTE: we feed the RAW chunk text, not the e5 "passage:" prefixed form.
        # Cross-encoders have their own tokenization and don't need that hint.
        pairs = [(question, h.text) for h in candidates]
        rerank_scores = self.reranker.predict(pairs)

        # 3. Re-sort by rerank score (higher = more relevant), keep top k.
        scored = list(zip(candidates, rerank_scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]

        # Rebuild Hit objects with NEW rank and the rerank score.
        # We keep the cross-encoder score in `score` so downstream code reads
        # the reranker's judgment, not the bi-encoder's; the original dense
        # score is intentionally dropped to avoid confusion.
        out: list[Hit] = []
        for new_rank, (hit, rscore) in enumerate(top, start=1):
            out.append(replace(hit, rank=new_rank, score=float(rscore)))
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Dense retrieval with reranking.")
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--candidates", type=int, default=DEFAULT_N_CANDIDATES,
                    help="how many candidates to fetch before reranking")
    ap.add_argument("--no-procedural", action="store_true")
    args = ap.parse_args()

    r = RerankedRetriever(n_candidates=args.candidates)
    where = {"is_procedural": False} if args.no_procedural else None
    hits = r.search(args.question, k=args.k, where=where)

    print(f"\nQuestion: {args.question}\n" + "=" * 70)
    for h in hits:
        iso = h.date_iso
        date = f"{iso[:4]}{iso[4:6]}{iso[6:8]}" if len(iso) == 8 else iso
        print(f"\n#{h.rank}  (id: {h.chunk_id}) (rerank score {h.score:+.3f})")
        print(f"    {date} — {h.agenda_item}")
        print(f"    {h.speaker or '—'}")
        body = h.text[:280] + ("…" if len(h.text) > 280 else "")
        print(f"    {body}")


if __name__ == "__main__":
    main()