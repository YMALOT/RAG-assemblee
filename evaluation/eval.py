"""
Evaluation harness for the retrieval step.

Loads eval/questions.jsonl, runs the retriever on each question, and computes
standard retrieval metrics against the hand-annotated relevant_chunk_ids:
  - recall@k : fraction of a question's relevant chunks found in the top k
  - hit@k    : did we find AT LEAST ONE relevant chunk in the top k? (per question)
  - MRR      : mean reciprocal rank of the FIRST relevant chunk (rewards ranking
               the right passage near the top, not just retrieving it somewhere)

Out-of-corpus questions (in_corpus=false) are reported separately: for those,
there is no relevant chunk, so we instead look at the top similarity score to
see whether it is distinguishable from in-corpus questions.

Usage:
  python run_eval.py                       # uses eval/questions.jsonl, k=5
  python run_eval.py --k 10 --no-procedural
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


QUESTIONS_PATH = "evaluation/questions.jsonl"


def load_questions(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).open(encoding="utf-8")]


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    if not relevant_ids:
        return float("nan")          # undefined when there are no relevant chunks
    found = sum(1 for r in relevant_ids if r in retrieved_ids)
    return found / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """1/rank of the first relevant chunk; 0 if none retrieved."""
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / i
    return 0.0


def evaluate(retriever, questions: list[dict], k: int, exclude_procedural: bool) -> None:
    where = {"is_procedural": False} if exclude_procedural else None

    in_corpus_rows, out_corpus_rows = [], []

    for q in questions:
        hits = retriever.search(q["question"], k=k, where=where)
        # retrieve.py exposes chunk_id as the syceron id; align with annotation.
        retrieved_ids = [h.chunk_id for h in hits]
        top_score = hits[0].score if hits else float("nan")

        if q.get("in_corpus", True):
            rel = q.get("relevant_chunk_ids", [])
            rec = recall_at_k(retrieved_ids, rel)
            rr = reciprocal_rank(retrieved_ids, rel)
            hit = 1.0 if rr > 0 else 0.0
            in_corpus_rows.append((q, rec, rr, hit, top_score))
            print(f'[{q["id"]}] {q.get("type","?"):12} '
                  f'recall@{k}={rec:.2f}  RR={rr:.2f}  top={top_score:.3f}  '
                  f'{q["question"][:50]}')
        else:
            out_corpus_rows.append((q, top_score))
            print(f'[{q["id"]}] {"out_of_corpus":12} '
                  f'top={top_score:.3f}  {q["question"][:50]}')

    print("\n" + "=" * 70)
    if in_corpus_rows:
        recalls = [r for _, r, _, _, _ in in_corpus_rows]
        rrs = [rr for _, _, rr, _, _ in in_corpus_rows]
        hits_ = [h for _, _, _, h, _ in in_corpus_rows]
        in_scores = [s for _, _, _, _, s in in_corpus_rows]
        print(f"In-corpus questions: {len(in_corpus_rows)}")
        print(f"  mean recall@{k} : {mean(recalls):.3f}")
        print(f"  hit@{k}         : {mean(hits_):.3f}  "
              f"(fraction with >=1 relevant chunk in top {k})")
        print(f"  MRR            : {mean(rrs):.3f}")
        print(f"  mean top score : {mean(in_scores):.3f}")
    if out_corpus_rows:
        out_scores = [s for _, s in out_corpus_rows]
        print(f"\nOut-of-corpus questions: {len(out_corpus_rows)}")
        print(f"  mean top score : {mean(out_scores):.3f}")
        if in_corpus_rows:
            gap = mean(in_scores) - mean(out_scores)
            print(f"  score gap (in - out): {gap:.3f}  "
                  f"(larger = easier to detect out-of-corpus by score)")

    # Per-type breakdown (in-corpus only)
    by_type: dict[str, list[float]] = {}
    for q, rec, _, _, _ in in_corpus_rows:
        by_type.setdefault(q.get("type", "?"), []).append(rec)
    if by_type:
        print("\nMean recall@{} by question type:".format(k))
        for t, vals in sorted(by_type.items()):
            print(f"  {t:14} {mean(vals):.3f}  (n={len(vals)})")
