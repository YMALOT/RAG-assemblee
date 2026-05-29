"""
Interactive annotation tool to build eval/questions.jsonl efficiently.

For each question you type, it:
  1. shows the top-k retrieved chunks (full text + chunk_id + metadata),
  2. lets you mark which ones are relevant (by number),
  3. optionally runs a keyword search to surface chunks retrieval may have
     missed (so you don't only annotate what the system already finds),
  4. asks for the question type and in/out-corpus flag,
  5. appends a correctly-formatted line to eval/questions.jsonl
     (with the FULL chunk_id, so it matches run_eval.py).

YOU make the relevance judgments — the tool just removes the friction and
guarantees the format. Run it, annotate your questions in one session, then
have the resulting file reviewed.

Usage:
  python annotate.py
  python annotate.py --k 10 --out eval/questions.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.retrieve import Retriever
from eval.search_corpus import load_chunks, search, CHUNKS_PATH

TYPES = ["factual", "speaker", "disagreement", "conceptual",
         "synthesis", "out_of_corpus"]


def show_chunk(n: int, cid: str, text: str, meta_line: str) -> None:
    print(f"\n  ({n}) {cid}")
    print(f"      {meta_line}")
    # full text, wrapped for readability
    for i in range(0, len(text), 100):
        print(f"      {text[i:i+100]}")


def parse_selection(raw: str, max_n: int) -> list[int]:
    """Parse '1 3 5' or '1,3,5' into [1,3,5], ignoring out-of-range."""
    raw = raw.replace(",", " ")
    out = []
    for tok in raw.split():
        if tok.isdigit() and 1 <= int(tok) <= max_n:
            out.append(int(tok))
    return out


def annotate_one(question: str, retriever: Retriever, all_chunks: list[dict],
                 k: int) -> dict:
    print("\n" + "=" * 72)
    print(f"QUESTION: {question}")
    print("=" * 72)

    # 1. Retrieved candidates
    hits = retriever.search(question, k=k)
    candidates: list[tuple[str, str]] = []   # (chunk_id, short text) for selection
    print(f"\n--- Top {k} retrieved chunks ---")
    for i, h in enumerate(hits, start=1):
        who = h.speaker or "—"
        if h.role:
            who += f", {h.role}"
        meta = f"{h.date_iso} | {who} | {h.agenda_item[:40]} | score {h.score:.3f}"
        show_chunk(i, h.chunk_id, h.text, meta)
        candidates.append((h.chunk_id, h.text))

    sel = parse_selection(
        input("\nRelevant chunk numbers (e.g. '1 3'), or empty if none: "),
        len(candidates))
    relevant = {candidates[i - 1][0] for i in sel}

    # 2. Optional keyword search to catch missed chunks
    kw = input("Optional keyword(s) to double-check the corpus "
               "(space-separated, empty to skip): ").strip()
    if kw:
        kw_hits = search(all_chunks, kw.split(), require_all=False)
        print(f"\n--- {len(kw_hits)} keyword matches (showing up to 15) ---")
        kw_candidates = []
        for i, c in enumerate(kw_hits[:15], start=1):
            who = c["speaker"] or "—"
            meta = f'{c["date_iso"]} | {who} | {c["agenda_item"][:40]}'
            show_chunk(i, c["chunk_id"], c["text"], meta)
            kw_candidates.append(c["chunk_id"])
        sel2 = parse_selection(
            input("\nAdditional relevant numbers from keyword search "
                  "(empty if none): "), len(kw_candidates))
        relevant |= {kw_candidates[i - 1] for i in sel2}

    # 3. Type and corpus flag
    print(f"\nTypes: {', '.join(f'{i+1}={t}' for i, t in enumerate(TYPES))}")
    t_raw = input("Type number: ").strip()
    qtype = TYPES[int(t_raw) - 1] if t_raw.isdigit() and 1 <= int(t_raw) <= len(TYPES) else "?"
    in_corpus = qtype != "out_of_corpus" and len(relevant) > 0
    if qtype != "out_of_corpus" and not relevant:
        # No relevant chunk found but not flagged out_of_corpus — confirm.
        c = input("No relevant chunk selected. Mark as out_of_corpus? [y/N]: ")
        if c.strip().lower() == "y":
            qtype, in_corpus = "out_of_corpus", False

    expected = input("Expected answer (short, optional): ").strip() or None

    return {
        "question": question,
        "type": qtype,
        "in_corpus": in_corpus,
        "relevant_chunk_ids": sorted(relevant),
        "expected_answer": expected,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive eval-set annotation.")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="eval/questions.jsonl")
    ap.add_argument("--chunks", default=CHUNKS_PATH)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume-friendly: count existing questions to continue ids.
    existing = 0
    if out_path.exists():
        existing = sum(1 for _ in out_path.open(encoding="utf-8"))
        print(f"{out_path} already has {existing} question(s); appending.")

    retriever = Retriever()
    all_chunks = load_chunks(args.chunks)

    print("\nType a question and press Enter. Empty line to finish.\n")
    idx = existing
    with out_path.open("a", encoding="utf-8") as f:
        while True:
            q = input("\n>>> New question (empty to stop): ").strip()
            if not q:
                break
            idx += 1
            record = annotate_one(q, retriever, all_chunks, k=args.k)
            record = {"id": f"q{idx:02d}", **record}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"\n✓ Saved {record['id']}  "
                  f"({len(record['relevant_chunk_ids'])} relevant chunk(s), "
                  f"type={record['type']})")

    print(f"\nDone. {idx - existing} question(s) added to {out_path}.")


if __name__ == "__main__":
    main()