"""
Fetch and display chunks by their chunk_id — useful for reviewing annotations,
inspecting a chunk cited in an answer, or re-calibrating relevance judgments.

Usage:
  python get_chunk.py CRSANR5L15S2022O1N080_2703323_0
  python get_chunk.py id1 id2 id3                       # several at once
  python get_chunk.py --from-question q01               # all relevant chunks of a question
  python get_chunk.py --file ids.txt                    # ids listed in a file (one per line)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CHUNKS_PATH = "chunks.jsonl"
QUESTIONS_PATH = "evaluation/questions.jsonl"


def load_chunks_index(path: str | Path) -> dict[str, dict]:
    """Map chunk_id -> chunk record for O(1) lookup."""
    index = {}
    for line in Path(path).open(encoding="utf-8"):
        c = json.loads(line)
        index[c["chunk_id"]] = c
    return index


def ids_from_question(questions_path: str | Path, qid: str) -> list[str]:
    for line in Path(questions_path).open(encoding="utf-8"):
        if not line.strip():
            continue
        q = json.loads(line)
        if q.get("id") == qid:
            return q.get("relevant_chunk_ids", [])
    raise SystemExit(f"Question '{qid}' not found in {questions_path}")


def pretty_date(iso: str) -> str:
    return f"{iso[:4]}-{iso[4:6]}-{iso[6:8]}" if len(iso) == 8 else iso


def show(c: dict) -> None:
    who = c["speaker"] or "—"
    if c["role"]:
        who += f", {c['role']}"
    proc = "  [procédural]" if c["is_procedural"] else ""
    part = f"  part {c['part']+1}/{c['n_parts']}" if c["n_parts"] > 1 else ""
    print("\n" + "=" * 72)
    print(f"{c['chunk_id']}{part}{proc}")
    print(f"  {pretty_date(c['date_iso'])} | {who} | {c['agenda_item']}")
    print(f"  {len(c['text'])} chars")
    print("-" * 72)
    for i in range(0, len(c["text"]), 100):
        print(f"  {c['text'][i:i+100]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch chunks by id.")
    ap.add_argument("ids", nargs="*", help="one or more chunk_id")
    ap.add_argument("--from-question", help="show all relevant chunks of a question id")
    ap.add_argument("--file", help="read ids from a file, one per line")
    ap.add_argument("--chunks", default=CHUNKS_PATH)
    ap.add_argument("--questions", default=QUESTIONS_PATH)
    args = ap.parse_args()

    wanted: list[str] = list(args.ids)
    if args.from_question:
        wanted += ids_from_question(args.questions, args.from_question)
    if args.file:
        wanted += [
            l.strip() for l in Path(args.file).open(encoding="utf-8") if l.strip()
        ]

    if not wanted:
        raise SystemExit("Provide chunk ids, --from-question, or --file.")

    index = load_chunks_index(args.chunks)
    print(f"Looking up {len(wanted)} chunk(s)...")
    for cid in wanted:
        c = index.get(cid)
        if c is None:
            print(f"\n  /!\\ NOT FOUND: {cid}")
        else:
            show(c)


if __name__ == "__main__":
    main()
