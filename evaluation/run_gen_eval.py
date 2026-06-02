"""
End-to-end generation evaluation using an LLM-as-judge.

For each question in eval/questions.jsonl:
  1. Retrieve chunks with the reranked retriever (our best retrieval config).
  2. Generate an answer with the production generate.py pipeline.
  3. Ask Claude Sonnet 4.6 to judge the answer along two/three axes,
     returning a structured JSON verdict.

Metrics (in-corpus):
  - faithfulness : faithful / partial / unfaithful — is the answer grounded
                   in the retrieved chunks, or does it invent?
  - correctness  : correct / partial / incorrect  — does it match the
                   hand-written expected_answer?

Metrics (out-of-corpus):
  - refusal      : refused / answered — did the system correctly decline,
                   or did it produce an answer despite the absence of
                   relevant information?

Outputs:
  - eval/generation_results.jsonl  one JSON object per question, including
                                   the generated answer, the judge's verdict,
                                   and the justifications (transparency).
  - aggregate summary to stdout.

Single-run methodology by design (low temperature, no repeats) for speed and
cost; see methodology.md for the trade-off.

Usage:
  python run_gen_eval.py
  python run_gen_eval.py --limit 3              # try on a subset
  python run_gen_eval.py --out results.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

# Reuse the production generation pipeline and its prompts, just swap in
# the reranked retriever as the retriever component.
from generation import API_KEY, BASE_URL, RerankedRetriever
from generation import answer as rag_answer

QUESTIONS_PATH = "evaluation/questions.jsonl"
RESULTS_PATH = "evaluation/generation_results.jsonl"

# A different (more capable) model than the generator, to limit
# self-evaluation bias. Versioned id for reproducibility.
JUDGE_MODEL = "claude-sonnet-4-6"
JUDGE_TEMPERATURE = 0.0


# ---------------------------------------------------------------- judging
JUDGE_SYSTEM = (
    "Tu es un évaluateur rigoureux d'un système de question-réponse. "
    "Tu juges des réponses produites à partir d'extraits de débats "
    "parlementaires. Tu rends UNIQUEMENT un objet JSON, sans commentaire "
    "additionnel, suivant exactement le schéma demandé."
)

JUDGE_PROMPT_IN_CORPUS = """Évalue la réponse selon deux axes.

**Fidélité (faithfulness)** : la réponse est-elle ancrée dans les extraits \
fournis ? Une réponse fidèle ne s'appuie QUE sur ce qui est dans les extraits. \
Toute affirmation absente des extraits, même vraie par ailleurs, compte comme \
non fidèle.
  - "faithful"   : tout ce qui est affirmé peut être tracé aux extraits.
  - "partial"    : la majorité est tracée, mais au moins une affirmation \
n'est pas dans les extraits.
  - "unfaithful" : la réponse invente plusieurs éléments absents des extraits.

**Exactitude (correctness)** : la réponse couvre-t-elle l'information \
attendue (expected_answer) ?
  - "correct" : couvre l'essentiel de la réponse attendue.
  - "partial" : couvre une partie significative, mais en omet ou en déforme.
  - "incorrect" : ne répond pas à la question, ou affirme l'inverse.

Question : {question}

Extraits fournis au modèle générateur :
{context}

Réponse attendue (référence d'annotation) :
{expected}

Réponse produite par le système :
{answer}

Rends UNIQUEMENT ce JSON :
{{
  "faithfulness": "faithful" | "partial" | "unfaithful",
  "faithfulness_justification": "une phrase courte",
  "correctness": "correct" | "partial" | "incorrect",
  "correctness_justification": "une phrase courte"
}}"""

JUDGE_PROMPT_OUT_CORPUS = """La question suivante n'a PAS de réponse dans le \
corpus. Le système doit donc refuser de répondre et le signaler explicitement.

**Comportement (refusal)** :
  - "refused"  : le système dit explicitement qu'il ne peut pas répondre \
ou que l'information n'est pas dans les extraits.
  - "answered" : le système produit une réponse de fond, sans signaler \
l'absence d'information.

Question : {question}

Extraits fournis au modèle générateur (qui ne contiennent pas la réponse) :
{context}

Réponse produite par le système :
{answer}

Rends UNIQUEMENT ce JSON :
{{
  "refusal": "refused" | "answered",
  "refusal_justification": "une phrase courte"
}}"""


def format_context(hits) -> str:
    """Render the retrieved chunks as they were given to the generator."""
    blocks = []
    for h in hits:
        iso = h.date_iso
        date = f"{iso[:4]}-{iso[4:6]}-{iso[6:8]}" if len(iso) == 8 else iso
        who = h.speaker or "—"
        if h.role:
            who += f", {h.role}"
        blocks.append(f"[Extrait {h.rank} | {who} | {date}]\n{h.text}")
    return "\n\n".join(blocks)


def parse_judge_json(raw: str) -> dict:
    """Extract the JSON object from the judge's response, tolerant to fences."""
    # Strip ```json … ``` fences if the judge added them.
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: find the first {...} block.
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def judge(client: OpenAI, question: dict, generated: str, context: str) -> dict:
    """Call Sonnet 4.6 as judge, return the parsed verdict."""
    if question["in_corpus"]:
        prompt = JUDGE_PROMPT_IN_CORPUS.format(
            question=question["question"],
            context=context,
            expected=question.get("expected_answer") or "(non fournie)",
            answer=generated,
        )
    else:
        prompt = JUDGE_PROMPT_OUT_CORPUS.format(
            question=question["question"],
            context=context,
            answer=generated,
        )

    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=JUDGE_TEMPERATURE,
    )
    raw = resp.choices[0].message.content
    return parse_judge_json(raw)


# ------------------------------------------------------------- aggregation
def summarise(results: list[dict]) -> None:
    in_c = [r for r in results if r["in_corpus"]]
    out_c = [r for r in results if not r["in_corpus"]]

    print("\n" + "=" * 70)
    print(f"Generation evaluation — {len(results)} questions")
    print("=" * 70)

    if in_c:
        faith = [r["verdict"]["faithfulness"] for r in in_c]
        corr = [r["verdict"]["correctness"] for r in in_c]
        n = len(in_c)
        print(f"\nIn-corpus questions ({n})")
        print("  Faithfulness:")
        for label in ("faithful", "partial", "unfaithful"):
            c = faith.count(label)
            print(f"    {label:11} {c:2}  ({100*c/n:4.1f}%)")
        print("  Correctness:")
        for label in ("correct", "partial", "incorrect"):
            c = corr.count(label)
            print(f"    {label:11} {c:2}  ({100*c/n:4.1f}%)")

    if out_c:
        refusal = [r["verdict"]["refusal"] for r in out_c]
        n = len(out_c)
        print(f"\nOut-of-corpus questions ({n})")
        print("  Refusal behaviour:")
        for label in ("refused", "answered"):
            c = refusal.count(label)
            print(f"    {label:11} {c:2}  ({100*c/n:4.1f}%)")


# -------------------------------------------------------------------- main
def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="LLM-as-judge generation eval.")
    ap.add_argument("--questions", default=QUESTIONS_PATH)
    ap.add_argument("--out", default=RESULTS_PATH)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N questions (debug)",
    )
    args = ap.parse_args()

    questions = [
        json.loads(l) for l in Path(args.questions).open(encoding="utf-8") if l.strip()
    ]
    if args.limit:
        questions = questions[: args.limit]
    print(
        f"Evaluating generation on {len(questions)} questions "
        f"(retriever: reranked, judge: {JUDGE_MODEL})\n"
    )

    # Single retriever instance — loading the cross-encoder costs a few seconds.
    retriever = RerankedRetriever()
    judge_client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    with Path(args.out).open("w", encoding="utf-8") as f:
        for q in questions:
            print(f"[{q['id']}] {q['question'][:60]}...", flush=True)
            # 1+2. Retrieve + generate (uses the production pipeline).
            generated, hits = rag_answer(
                q["question"],
                k=5,
                exclude_procedural=True,
                retriever=retriever,
            )
            context = format_context(hits)

            # 3. Judge.
            try:
                verdict = judge(judge_client, q, generated, context)
            except Exception as e:
                verdict = {"error": str(e)}
                print(f"  /!\\ judge error: {e}")

            row = {
                "id": q["id"],
                "question": q["question"],
                "type": q["type"],
                "in_corpus": q["in_corpus"],
                "expected_answer": q.get("expected_answer"),
                "generated_answer": generated,
                "retrieved_chunk_ids": [h.chunk_id for h in hits],
                "verdict": verdict,
            }
            results.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

            # Brief per-question stdout feedback
            v = verdict
            if "error" in v:
                pass
            elif q["in_corpus"]:
                print(
                    f"    faithfulness={v.get('faithfulness')}  "
                    f"correctness={v.get('correctness')}"
                )
            else:
                print(f"    refusal={v.get('refusal')}")

    summarise(results)
    print(f"\nDetailed results written to {args.out}")


if __name__ == "__main__":
    main()
