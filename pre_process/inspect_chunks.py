"""
Visually inspect the chunking: sample k random chunks and print them with
useful diagnostics, plus a short summary of the whole chunk set.

This is a quality-control tool — the kind that surfaces issues like glued
sentences, chunks that start mid-sentence, or oddly long/short chunks. Run it
after any change to ingest.py or chunk.py to eyeball the result.

Usage:
  python inspect.py                 # 5 random chunks + global stats
  python inspect.py --k 10          # 10 random chunks
  python inspect.py --seed 42       # reproducible sample
  python inspect.py --long          # bias the sample toward the longest chunks
  python inspect.py --flags-only    # skip samples, only show the health report
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path

try:
    from .chunk import CHUNKS_PATH
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from chunk import CHUNKS_PATH

# Heuristic quality flags (not errors — just things worth a human glance).
_GLUED = re.compile(r'[a-zà-öø-ÿ]{2}[.!?…][«"A-ZÀ-Þ]')  # sentence glued
_GLUED_QUOTE = re.compile(r'[.!?…]\s*»[«"A-ZÀ-Þ]')  # glued after »
_STARTS_LOWER = re.compile(r"^[a-zà-öø-ÿ]")  # starts mid-sentence


def load_chunks(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).open(encoding="utf-8")]


def flags_for(text: str) -> list[str]:
    f = []
    if _GLUED.search(text):
        f.append("glued-sentence")
    if _GLUED_QUOTE.search(text):
        f.append("glued-after-quote")
    if _STARTS_LOWER.match(text.strip()):
        f.append("starts-lowercase")
    if text.count("«") != text.count("»"):
        f.append("unbalanced-quotes")  # often normal: citation spans chunks
    return f


def pretty_date(iso: str) -> str:
    return f"{iso[:4]}-{iso[4:6]}-{iso[6:8]}" if len(iso) == 8 else iso


def print_chunk(c: dict) -> None:
    who = c["speaker"] or "—"
    if c["role"]:
        who += f", {c['role']}"
    flags = flags_for(c["text"])
    flag_str = ("  ⚑ " + ", ".join(flags)) if flags else ""
    proc = "  [procédural]" if c["is_procedural"] else ""
    part = f"  part {c['part']+1}/{c['n_parts']}" if c["n_parts"] > 1 else ""

    print("\n" + "-" * 72)
    print(f"{c['chunk_id']}{part}{proc}{flag_str}")
    print(f"  {pretty_date(c['date_iso'])} | {who} | {c['agenda_item'][:50]}")
    print(f"  {len(c['text'])} chars")
    print()
    # wrap text at ~100 chars for readability
    for i in range(0, len(c["text"]), 100):
        print(f"  {c['text'][i:i+100]}")


def summary(chunks: list[dict], k_target: int) -> None:
    lengths = sorted(len(c["text"]) for c in chunks)
    n = len(chunks)
    print("=" * 72)
    print(f"CHUNK SET HEALTH REPORT  ({n} chunks)")
    print("=" * 72)
    print(
        f"length (chars): min={lengths[0]}  "
        f"p50={statistics.median(lengths):.0f}  "
        f"mean={statistics.mean(lengths):.0f}  "
        f"p95={lengths[int(n*0.95)]}  max={lengths[-1]}"
    )

    procedural = sum(1 for c in chunks if c["is_procedural"])
    print(
        f"procedural: {procedural} ({100*procedural/n:.1f}%)  |  "
        f"debate: {n-procedural}"
    )

    subchunks = sum(1 for c in chunks if c["n_parts"] > 1)
    print(f"sub-chunks (from re-split interventions): {subchunks}")

    # very short chunks are worth knowing about (micro-replies)
    tiny = sum(1 for L in lengths if L < 40)
    print(f"very short chunks (<40 chars): {tiny}")

    # aggregate quality flags across the whole set
    counts: dict[str, int] = {}
    for c in chunks:
        for f in flags_for(c["text"]):
            counts[f] = counts.get(f, 0) + 1
    if counts:
        print("\nquality flags across all chunks:")
        for f, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            note = ""
            if f == "unbalanced-quotes":
                note = "  (often legitimate: a citation split across chunks)"
            print(f"  {f:20} {cnt}{note}")
    else:
        print("\nno quality flags raised — clean set.")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect chunking quality.")
    ap.add_argument("--k", type=int, default=5, help="number of chunks to sample")
    ap.add_argument("--seed", type=int, default=None, help="seed for reproducibility")
    ap.add_argument(
        "--long",
        action="store_true",
        help="sample from the longest chunks instead of randomly",
    )
    ap.add_argument(
        "--flags-only",
        action="store_true",
        help="only show the health report, no sampled chunks",
    )
    ap.add_argument("--chunks", default=CHUNKS_PATH)
    args = ap.parse_args()

    chunks = load_chunks(args.chunks)
    summary(chunks, args.k)

    if args.flags_only:
        return

    if args.long:
        sample = sorted(chunks, key=lambda c: -len(c["text"]))[: args.k]
        print(f"\n{args.k} LONGEST chunks:")
    else:
        if args.seed is not None:
            random.seed(args.seed)
        sample = random.sample(chunks, min(args.k, len(chunks)))
        print(
            f"\n{len(sample)} RANDOM chunks"
            + (f" (seed={args.seed})" if args.seed is not None else "")
            + ":"
        )

    for c in sample:
        print_chunk(c)


if __name__ == "__main__":
    main()
