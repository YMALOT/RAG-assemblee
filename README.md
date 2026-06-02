# RAG on the French National Assembly debates: anti-school-bullying law
 
A retrieval-augmented generation system over the verbatim debates of the French
National Assembly on the law against school bullying (PPL n°4658, loi n°2022-299,
3 sittings). Built end-to-end as a portfolio project: hand-built pipeline (no
LangChain), hand-annotated evaluation set, measured improvements.
 
🇫🇷 *Version française : [README.fr.md](README.fr.md)*
 
## Interactive demo

[HF link](https://huggingface.co/spaces/YMalot/rag-assemblee-demo)

## Pipeline
 
`XML syceron → ingest.py → chunk.py → embed_and_index.py → Chroma`
` → retrieve.py → generate.py (Claude Haiku 4.5 via API)`
 
- **Source.** Open-data XML (syceron) from data.assemblee-nationale.fr, Etalab
  licence. 3 sittings, 1209 speeches → 1838 chunks.
- **Chunking.** One chunk per speech if short, otherwise sentence-aware split
  with one-sentence overlap; ~900 chars target.
- **Embeddings.** `intfloat/multilingual-e5-base`, CPU, e5 prefixes
  (`passage:` / `query:`), cosine space.
- **Generation.** OpenAI-compatible backend (defaults to Anthropic API for
  quality; pipeline also runs against a local Ollama for an open-weights demo).
  System prompt enforces source-grounded answers with citations and explicit
  refusal when the corpus has no answer.
## Evaluation
 
17 hand-annotated questions covering 5 types (`factual`, `conceptual`,
`disagreement`, `synthesis`, `out_of_corpus`), with relevant chunk IDs and
expected answers. Annotation rule kept strict: *a chunk is relevant only if I
would cite it to answer the question* — vaguely-related chunks excluded.
 
Three retrieval configurations compared on the same evaluation set:
 
| Configuration            | Recall@5 | MRR   | Hit@5 | Score gap in/out |
|--------------------------|---------:|------:|------:|-----------------:|
| Dense (e5) baseline      |   0.451  | 0.750 | 0.929 |           0.015  |
| + Reranker (BGE-v2-m3)   | **0.467**| **0.764** | 0.929 |       **0.340**  |
| + Hybrid (BM25 + RRF)    |   0.339  | 0.750 | 0.929 |           0.003  |
 
### Reading the numbers
 
**Baseline.** High hit@5 (0.93) and MRR (0.75) with moderate recall (0.45):
the system reliably finds *one* good chunk near the top, but recovers only
part of the relevant set when answers are dispersed — exactly the expected
profile of a dense bi-encoder on a topically narrow corpus. Per-type recall
confirms it: factual/conceptual/disagreement at 0.47-0.63, synthesis at 0.33.
 
**Reranker.** The cross-encoder produces a modest gain on ranking (+1.6 pt
recall, +1.4 pt MRR), driven by improvements on the harder types
(synthesis 0.33 → 0.39, factual 0.47 → 0.53). The decisive gain is elsewhere:
the in/out score gap widens from 0.015 to 0.340 — a ×23 improvement that
turns score thresholding into a usable out-of-corpus detector, where the
dense score alone is essentially flat. Kept in production as a *guardrail
signal*, complementing the prompt-level "I don't know" instruction.
 
**Hybrid (dense + BM25).** Degrades recall by 11 points. This is a
*diagnostic* result, not just a failure: BM25's discriminative power depends
on IDF, and IDF collapses on a mono-thematic corpus where words like
*harcèlement*, *scolaire*, *élèves* recur everywhere. BM25 ends up boosting
chunks that *repeat the question's vocabulary* rather than chunks that
*answer it*, and RRF fusion propagates that noise. Hybrid retrieval would
likely become valuable again with thematic diversification — e.g., adding
the law text itself, which is the next planned extension.
 
## Generation evaluation
 
Beyond retrieval, the generated answers themselves were evaluated end-to-end
on the same 17 questions, using **Claude Sonnet 4.6 as an LLM-as-judge** —
deliberately a different (more capable) model than the generator
(Claude Haiku 4.5) to limit self-evaluation bias. Retrieval was set to the
reranked configuration; the judge received the question, the retrieved
chunks, the expected answer, and the generated answer, and returned a
structured JSON verdict with justification.
 
| Metric                  | Result                                |
|-------------------------|---------------------------------------|
| Faithfulness (in-corpus, n=14) | **100% faithful**, 0% partial, 0% unfaithful |
| Correctness (in-corpus, n=14)  | 28.6% correct, 71.4% partial, **0% incorrect** |
| Refusal (out-of-corpus, n=3)   | **100% refused**                       |
 
### Reading the numbers
 
**Faithfulness at 100% validates the guardrail.** Every in-corpus answer is
fully grounded in the retrieved chunks — no hallucination, no leakage of
general knowledge. The system prompt that constrains generation to the
provided extracts is doing its job.
 
**Refusal at 100% validates the out-of-corpus behaviour.** All three
out-of-corpus questions, including the thematically close one (causes of
school bullying — adjacent to the topic but not addressed in the debates),
were correctly refused. The prompt-level "I don't know" instruction holds
under varying difficulty.
 
**The 28.6% correctness with 0% incorrect tells the most interesting story.**
Combined with 100% faithfulness, this means the system never invents to
fill gaps: when retrieval is incomplete, the answer is honestly partial
rather than confidently wrong. The 71% partial rate therefore measures
*retrieval coverage*, not generation quality — consistent with the dense
retrieval recall@5 of 0.47 on synthesis questions. The system fails
gracefully, which is the correct behaviour for a RAG.
 
This identifies the **retrieval as the next lever**, not the generator. The
planned addition of the law text (see "What's next") should directly lift
the correctness rate on synthesis questions by improving coverage rather
than by changing the generation prompt.
 
### Methodology notes
 
Single judgement per answer with temperature 0 for stability — quick and
inexpensive, but does not measure intra-judge variance. A more rigorous
protocol would re-judge ambiguous cases (partial verdicts) 2-3 times and
report agreement; left as a future refinement.
 
See [eval/methodology.md](eval/methodology.md) for the annotation and
evaluation principles.
 
## Method notes
 
- **Annotation is iterative.** Initial runs revealed gaps (questions where the
  retriever surfaced relevant chunks I had missed). Each diagnostic cycle —
  inspect underperforming questions, search by keyword, fix annotation —
  raised recall and made the metric more honest. The annotation is the
  experiment, not a one-shot label.
- **Strict binary relevance is a deliberate choice.** Graded relevance (nDCG)
  would be a refinement; binary keeps interpretation clean for a first
  evaluation.
- **Out-of-corpus detection by score alone is unreliable** with e5
  (gap 0.015) — confirmed empirically and worked around via the LLM-level
  guardrail. The reranker score makes thresholding feasible if needed.
## Stack
 
`Python · Chroma · sentence-transformers (e5-base, BGE-reranker-v2-m3) ·`
`rank-bm25 · OpenAI-compatible client (Anthropic API / Ollama)`
 
## What's next
 
- Add the law text (initial PPL n°4658 + final loi n°2022-299) as a distinct
  document type with article-level chunking. Expected to lift synthesis
  questions and revive hybrid retrieval.
- More robust generation evaluation: multiple judgements per answer to
  measure intra-judge agreement.
- Chunking ablation: target size and overlap as measured hyperparameters.

## Repository

```
.
├── pre_process/
│   ├── ingest.py
│   ├── sort.py
│   ├── chunk.py
│   ├── embed_and_index.py
│   └── inspect_chunks.py
├── generation/
│   ├── retrieve.py
│   ├── retrieve_reranked.py
│   ├── retrieve_hybrid.py
│   └── generate.py
├── evaluation/
│   ├── eval.py
│   ├── search_corpus.py
│   ├── get_chunk.py
│   ├── questions.jsonl
│   └── results/
│       ├── eval_plain.txt
│       ├── eval_reranked.txt
│       └── eval_hybrid.txt
├── annotate.py
└── run_eval.py
```
