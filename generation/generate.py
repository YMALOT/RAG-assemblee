"""
Generation step: take a question, retrieve relevant chunks, build a grounded
prompt, and ask an LLM to answer using ONLY those chunks (with citations).

Uses the OpenAI-compatible API, so the SAME code targets:
  - Ollama locally:  base_url="http://localhost:11434/v1", model="qwen2.5:3b"
  - a remote API:    base_url=<provider>, model=<model>, real api_key
Switch backend by changing base_url / model / api_key only.

The "I don't know" guardrail is enforced by the system prompt, not by a score
threshold: we measured that e5 similarity barely separates in-corpus (~0.88)
from out-of-corpus (~0.85) questions, so the LLM reading the passages is the
more reliable judge of whether the answer is actually present.

Run locally (Ollama must be running, model pulled):
  python generate.py "Combien d'enfants sont victimes de harcèlement scolaire ?"
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from openai import OpenAI

from .retrieve import Retriever, Hit

# --- Backend configuration (override via env vars to switch to a remote API) -
load_dotenv()
BASE_URL = os.environ.get("RAG_LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.environ.get("RAG_LLM_API_KEY", "ollama")   # unused by Ollama
MODEL = os.environ.get("RAG_LLM_MODEL", "qwen2.5:0.5b")

SYSTEM_PROMPT = (
    "Tu es un assistant qui répond à des questions sur les débats de "
    "l'Assemblée nationale concernant la loi visant à combattre le harcèlement "
    "scolaire. Réponds UNIQUEMENT à partir des extraits fournis ci-dessous. "
    "Pour chaque affirmation, cite l'orateur et la date entre crochets, par "
    "exemple : [M. Balanant, 2021-12-01]. "
    "Si les extraits ne contiennent pas de quoi répondre, dis explicitement : "
    "« Les débats fournis ne permettent pas de répondre à cette question. » "
    "N'invente jamais d'information absente des extraits. "
    "Quand plusieurs extraits donnent des chiffres ou avis différents, "
    "rapporte cette pluralité plutôt que d'en choisir un seul."
)


def format_context(hits: list[Hit]) -> str:
    """Render retrieved chunks as a numbered, citable context block."""
    blocks = []
    for h in hits:
        iso = h.date_iso
        date = f"{iso[:4]}-{iso[4:6]}-{iso[6:8]}" if len(iso) == 8 else iso
        who = h.speaker or "—"
        if h.role:
            who += f", {h.role}"
        blocks.append(
            f"[Extrait {h.rank} | {who} | {date} | {h.agenda_item}]\n{h.text}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    context = format_context(hits)
    user = (
        f"Extraits des débats :\n\n{context}\n\n"
        f"Question : {question}\n\n"
        f"Réponds en t'appuyant uniquement sur les extraits ci-dessus, "
        f"avec citations."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def answer(question: str, k: int = 5, exclude_procedural: bool = True,
           retriever: Retriever | None = None) -> tuple[str, list[Hit]]:
    """Full RAG turn: retrieve, build prompt, generate. Returns (answer, hits)."""
    retriever = retriever or Retriever()
    where = {"is_procedural": False} if exclude_procedural else None
    hits = retriever.search(question, k=k, where=where)

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=build_messages(question, hits),
        temperature=0.2,        # low: we want faithful, not creative, answers
    )
    return resp.choices[0].message.content, hits


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask the debate-corpus RAG.")
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--include-procedural", action="store_true",
                    help="also retrieve procedural notes (votes, applause)")
    args = ap.parse_args()

    text, hits = answer(
        args.question, k=args.k,
        exclude_procedural=not args.include_procedural,
    )
    print(f"\nQuestion : {args.question}\n" + "=" * 70)
    print(text)
    print("\n" + "-" * 70)
    print("Sources utilisées :")
    for h in hits:
        iso = h.date_iso
        date = f"{iso[:4]}-{iso[4:6]}-{iso[6:8]}" if len(iso) == 8 else iso
        print(f"  [{h.rank}] {h.speaker or '—'} — {date} "
              f"(score {h.score:.3f})")


if __name__ == "__main__":
    main()