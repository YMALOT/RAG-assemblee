"""
Chunk the parsed corpus into embedding-ready units.

Strategy (baseline):
  - Each intervention <= MAX_CHARS becomes a single chunk (metadata preserved).
  - Longer interventions are split on sentence boundaries, sentences accumulated
    up to a target size, with a one-sentence overlap between consecutive chunks.
  - Nothing is discarded: short replies and procedural notes are all kept,
    flagged via metadata so retrieval can filter later.

French sentence segmentation uses a regex tuned to parliamentary abbreviations
(M., Mme, no, art., etc.). If it proves too noisy on inspection, swap in pysbd.

Code/identifiers in English; French kept only for corpus text and abbreviations.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --- Tunable parameters (later varied in evaluation) -----------------------
MAX_CHARS = 1000  # interventions longer than this get re-split
TARGET_CHARS = 900  # target size when accumulating sentences into a chunk
OVERLAP_SENTENCES = 1  # sentence overlap between consecutive sub-chunks

# Abbreviations whose trailing "." must NOT be treated as a sentence end.
# Python's `re` forbids variable-width look-behind, so instead of encoding these
# in the split regex we split first, then merge back the false breaks caused by
# an abbreviation appearing right before the boundary.
_ABBREVIATIONS = {
    "M",
    "MM",
    "Mme",
    "Mmes",
    "Mlle",
    "Dr",
    "art",
    "al",
    "no",
    "nos",
    "cf",
    "p",
    "etc",
    "vol",
    "ch",
    "param",
}
# Candidate sentence boundary: punctuation + space + capital/quote/digit.
_SENTENCE_END = re.compile(r'(?<=[.!?…])\s+(?=[«"A-ZÀ-ÖØ-Þ0-9])')
# Last "word" before a boundary, to test whether it is an abbreviation.
_LAST_TOKEN = re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿ]+)\.?\s*$")


@dataclass
class Chunk:
    """One embedding-ready unit of text plus its provenance metadata."""

    chunk_id: str  # stable id: "<session_uid>_<syceron_id>_<part>"
    text: str  # the chunk's own text (without context prefix)
    speaker: str | None
    role: str | None
    agenda_item: str
    is_procedural: bool
    session_uid: str
    date_iso: str
    syceron_id: str | None
    part: int  # sub-chunk index within the parent intervention
    n_parts: int  # total sub-chunks for that intervention

    def context_prefix(self) -> str:
        """A short header prepended at EMBEDDING time to enrich the vector.

        Stored separately from `text` so the raw quote stays clean for display.
        """
        who = self.speaker or "—"
        if self.role:
            who += f", {self.role}"
        bits = [f"Séance du {_pretty_date(self.date_iso)}"]
        if self.agenda_item:
            bits.append(self.agenda_item)
        return f"[{' — '.join(bits)}] {who} :"

    def embedding_text(self) -> str:
        return f"{self.context_prefix()} {self.text}"


def _pretty_date(iso: str) -> str:
    """20211201 -> 2021-12-01 (cheap, locale-free)."""
    return f"{iso[:4]}-{iso[4:6]}-{iso[6:8]}" if len(iso) == 8 else iso


def split_sentences(text: str) -> list[str]:
    """Split French text into sentences, respecting common abbreviations.

    We split on every candidate boundary, then merge a fragment back onto the
    previous one when the previous fragment ends with a known abbreviation
    (e.g. "M.", "art.", "no") — those dots are not real sentence ends.
    """
    candidates = _SENTENCE_END.split(text)
    sentences: list[str] = []
    for frag in candidates:
        frag = frag.strip()
        if not frag:
            continue
        if sentences:
            m = _LAST_TOKEN.search(sentences[-1])
            if m and m.group(1) in _ABBREVIATIONS:
                sentences[-1] = sentences[-1] + " " + frag
                continue
        sentences.append(frag)
    return sentences


def _pack_sentences(sentences: list[str]) -> list[str]:
    """Group sentences into ~TARGET_CHARS chunks with a one-sentence overlap.

    A sentence longer than TARGET_CHARS becomes its own chunk (we never split
    mid-sentence). The index always advances, so no infinite loop is possible.
    """
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sent in sentences:
        # If adding this sentence would overflow a non-empty chunk, close it.
        if current and size + len(sent) + 1 > TARGET_CHARS:
            chunks.append(" ".join(current))
            current = current[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES else []
            size = sum(len(s) + 1 for s in current)
        current.append(sent)
        size += len(sent) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_intervention(row: dict) -> list[Chunk]:
    """Turn one corpus row (parsed intervention) into one or more Chunks."""
    text = row["text"]
    base = dict(
        speaker=row["speaker"],
        role=row["role"],
        agenda_item=row["agenda_item"],
        is_procedural=row["is_procedural"],
        session_uid=row["session_uid"],
        date_iso=row["date_iso"],
        syceron_id=row["syceron_id"],
    )
    sid = row["syceron_id"] or "na"

    if len(text) <= MAX_CHARS:
        pieces = [text]
    else:
        sentences = split_sentences(text)
        pieces = _pack_sentences(sentences) if sentences else [text]

    n = len(pieces)
    return [
        Chunk(
            chunk_id=f"{row['session_uid']}_{sid}_{k}",
            text=piece,
            part=k,
            n_parts=n,
            **base,
        )
        for k, piece in enumerate(pieces)
    ]


def build_chunks(corpus_path: str | Path) -> list[Chunk]:
    rows = [json.loads(l) for l in Path(corpus_path).open(encoding="utf-8")]
    chunks: list[Chunk] = []
    for row in rows:
        chunks.extend(chunk_intervention(row))
    return chunks


def save_chunks(chunks: list[Chunk], out_path: str | Path) -> None:
    with Path(out_path).open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import statistics
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "corpus.jsonl"
    print(src)
    chunks = build_chunks(src)
    save_chunks(chunks, "chunks.jsonl")

    lengths = sorted(len(c.text) for c in chunks)
    split_parents = sum(1 for c in chunks if c.n_parts > 1 and c.part == 0)
    print(f"{len(chunks)} chunks from corpus (was fewer interventions).")
    print(f"  re-split interventions: {split_parents}")
    print(
        f"  chunk length: min={lengths[0]} "
        f"median={statistics.median(lengths):.0f} max={lengths[-1]}"
    )
    print("\nExample embedding text (a long, re-split intervention):")
    for c in chunks:
        if c.n_parts > 1:
            print(f"  {c.chunk_id} (part {c.part+1}/{c.n_parts})")
            print(f"  {c.embedding_text()[:240]}...")
            break
