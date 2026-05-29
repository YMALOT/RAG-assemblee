"""
Keyword search over the chunks — a NON-vector way to find relevant passages.

Purpose: when annotating the evaluation set, vector retrieval (retrieve.py)
only shows what the system already finds. To label relevant_chunk_ids without
that blind spot, also search the corpus literally by keyword and check whether
a relevant passage exists that retrieval may have missed. Those gaps are the
most informative cases for measuring recall honestly.

Usage:
  python search_corpus.py "harcèlement"               # simple substring
  python search_corpus.py "700 000" "million"          # ANY of the terms
  python search_corpus.py "sanction" --all              # ALL terms required
  python search_corpus.py "cyberharcèlement" --full     # print full chunk text
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

CHUNKS_PATH = "chunks.jsonl"


def normalize(s: str) -> str:
    """Lowercase and strip accents, for accent-insensitive matching."""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def load_chunks(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).open(encoding="utf-8")]


def search(chunks: list[dict], terms: list[str], require_all: bool) -> list[dict]:
    norm_terms = [normalize(t) for t in terms]
    hits = []
    for c in chunks:
        hay = normalize(c["text"])
        matches = [t in hay for t in norm_terms]
        if (all(matches) if require_all else any(matches)):
            hits.append(c)
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description="Keyword search over chunks.")
    ap.add_argument("terms", nargs="+", help="one or more search terms")
    ap.add_argument("--all", action="store_true",
                    help="require ALL terms (default: ANY term)")
    ap.add_argument("--full", action="store_true",
                    help="print the full chunk text (default: 160-char preview)")
    ap.add_argument("--chunks", default=CHUNKS_PATH)
    args = ap.parse_args()

    chunks = load_chunks(args.chunks)
    hits = search(chunks, args.terms, require_all=args.all)

    mode = "ALL" if args.all else "ANY"
    print(f'{len(hits)} chunk(s) matching [{mode}] {args.terms}\n' + "=" * 70)
    for c in hits:
        who = c["speaker"] or "—"
        if c["role"]:
            who += f", {c['role']}"
        date = c["date_iso"]
        date = f"{date[:4]}-{date[4:6]}-{date[6:8]}" if len(date) == 8 else date
        body = c["text"] if args.full else c["text"][:160] + (
            "…" if len(c["text"]) > 160 else "")
        print(f'\n[{c["chunk_id"]}]  {date} — {who}')
        print(f'  {body}')


if __name__ == "__main__":
    main()