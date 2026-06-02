"""
Hybrid retrieval: dense (e5) + lexical (BM25), fused with Reciprocal Rank Fusion.

Why both? Dense captures meaning (paraphrase, synonymy) but blurs exact tokens;
BM25 captures exact-term matches (proper nouns, numbers, rare terms) but misses
reformulations. Together they cover each other's blind spots.

Why RRF for fusion? Dense scores (cosine, ~0.8-0.9) and BM25 scores (unbounded,
corpus-dependent) live on incomparable scales. RRF combines RANKS instead of
scores, which is scale-free and has one well-known default (k=60) that works
universally — no per-corpus tuning needed.

Interface mirrors Retriever from retrieve.py, so generate.py and run_eval.py
can switch backends by changing one import.

Usage:
  python retrieve_hybrid.py "Qu'a dit le ministre Blanquer ?"
  python retrieve_hybrid.py "..." --k 5 --candidates 20

Importable:
  from retrieve_hybrid import HybridRetriever
  r = HybridRetriever()
  hits = r.search("...", k=5)
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import replace
from pathlib import Path

from rank_bm25 import BM25Okapi

from generation.retrieve import Hit, Retriever  # reuse dense retriever and Hit type

CHUNKS_PATH = "chunks.jsonl"
DEFAULT_N_CANDIDATES = 20
RRF_K = 60  # Cormack et al.'s well-known default


def normalize_text(s: str) -> str:
    """Lowercase + accent-strip — same normalisation as search_corpus.py."""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# Token = run of letters/digits, accent-insensitive. Keeps numbers
# (700 000, article 4) and acronyms (PHARE, KiVa, CLEMI) as searchable tokens.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Simple French-friendly tokenizer. Drops 1-letter tokens (mostly noise)."""
    return [t for t in _TOKEN_RE.findall(normalize_text(text)) if len(t) > 1]


class HybridRetriever:
    """Dense + BM25 with RRF fusion. Same interface as Retriever."""

    def __init__(
        self,
        chunks_path: str = CHUNKS_PATH,
        n_candidates: int = DEFAULT_N_CANDIDATES,
        rrf_k: int = RRF_K,
    ) -> None:
        self.n_candidates = n_candidates
        self.rrf_k = rrf_k
        self.dense = Retriever()

        # Load chunks and build the BM25 index over their text.
        # We keep them in the same order Chroma sees them so we can map back.
        rows = [json.loads(l) for l in Path(chunks_path).open(encoding="utf-8")]
        self._chunks_by_id: dict[str, dict] = {r["chunk_id"]: r for r in rows}
        self._ordered_ids: list[str] = [r["chunk_id"] for r in rows]
        tokenized = [tokenize(r["text"]) for r in rows]
        self.bm25 = BM25Okapi(tokenized)

    # ------------------------------------------------------------------ BM25
    def _bm25_topk(self, question: str, n: int) -> list[str]:
        """Return the chunk_ids of the top-n BM25 matches, best first."""
        q_tokens = tokenize(question)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        # Argsort descending, take top n. Uses raw indices into _ordered_ids.
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        # Strip tail of zero-score matches (irrelevant chunks the index pads with).
        return [self._ordered_ids[i] for i in top_idx if scores[i] > 0]

    # ------------------------------------------------------------------ RRF
    @staticmethod
    def _rrf_fuse(rank_lists: list[list[str]], k: int) -> list[tuple[str, float]]:
        """Fuse multiple ranked lists by Reciprocal Rank Fusion.
        For each id d: RRF(d) = sum over lists of 1 / (k + rank_in_list(d)).
        Returns ids sorted by RRF score, best first.
        """
        scores: dict[str, float] = {}
        for ranked in rank_lists:
            for rank, doc_id in enumerate(ranked, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: -x[1])

    # --------------------------------------------------------------- search
    def search(self, question: str, k: int = 5, where: dict | None = None) -> list[Hit]:
        # 1. Dense candidates (with optional metadata filter, e.g. is_procedural).
        dense_hits = self.dense.search(question, k=self.n_candidates, where=where)
        dense_ids = [h.chunk_id for h in dense_hits]
        # Keep dense hits accessible for metadata reconstruction below.
        dense_by_id = {h.chunk_id: h for h in dense_hits}

        # 2. BM25 candidates. Apply the same procedural filter manually, since
        #    BM25 doesn't know about Chroma metadata.
        bm25_ids = self._bm25_topk(question, self.n_candidates)
        if where and "is_procedural" in where:
            wanted_proc = where["is_procedural"]
            bm25_ids = [
                cid
                for cid in bm25_ids
                if bool(self._chunks_by_id[cid].get("is_procedural", False))
                == wanted_proc
            ]

        # 3. RRF fusion of the two ranked lists.
        fused = self._rrf_fuse([dense_ids, bm25_ids], k=self.rrf_k)[:k]

        # 4. Rebuild Hit objects. Prefer dense hit when available (it carries
        #    a real cosine similarity); otherwise build from the raw chunk row.
        out: list[Hit] = []
        for new_rank, (cid, rrf_score) in enumerate(fused, start=1):
            if cid in dense_by_id:
                out.append(replace(dense_by_id[cid], rank=new_rank, score=rrf_score))
            else:
                # BM25-only hit: no cosine score; expose the RRF score in `score`.
                row = self._chunks_by_id[cid]
                out.append(
                    Hit(
                        rank=new_rank,
                        score=rrf_score,
                        text=row["text"],
                        speaker=row.get("speaker", "") or "",
                        role=row.get("role", "") or "",
                        agenda_item=row.get("agenda_item", "") or "",
                        date_iso=row.get("date_iso", "") or "",
                        session_uid=row.get("session_uid", "") or "",
                        is_procedural=bool(row.get("is_procedural", False)),
                        chunk_id=cid,
                    )
                )
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid dense+BM25 retrieval.")
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--candidates", type=int, default=DEFAULT_N_CANDIDATES)
    ap.add_argument("--no-procedural", action="store_true")
    ap.add_argument("--chunks", default=CHUNKS_PATH)
    args = ap.parse_args()

    r = HybridRetriever(chunks_path=args.chunks, n_candidates=args.candidates)
    where = {"is_procedural": False} if args.no_procedural else None
    hits = r.search(args.question, k=args.k, where=where)

    print(f"\nQuestion: {args.question}\n" + "=" * 70)
    for h in hits:
        iso = h.date_iso
        date = f"{iso[:4]}{iso[4:6]}{iso[6:8]}" if len(iso) == 8 else iso
        print(f"\n#{h.rank}  (id: {h.chunk_id})  (RRF score {h.score:.4f})")
        print(f"    {date} — {h.agenda_item}")
        print(f"    {h.speaker or '—'}")
        body = h.text[:280] + ("…" if len(h.text) > 280 else "")
        print(f"    {body}")


if __name__ == "__main__":
    main()
