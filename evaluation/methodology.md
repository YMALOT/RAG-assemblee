# Evaluation methodology

How the evaluation set was built and what its numbers can and cannot tell us.

🇫🇷 *Version française : [methodology.fr.md](methodology.fr.md)*

## Question set

17 questions, written from the perspective of a user who has not read the
debates, *before* looking at the corpus. Question wording was kept natural
(not copied from chunk vocabulary), to avoid the trivial-recall trap where
the system "wins" by matching its own words.

The set was designed to cover five categories with distinct retrieval
profiles:

- **factual** — single localised answer (counts, names, articles)
- **conceptual** — definition or characterisation
- **disagreement** — positions to be reconstructed across speakers
- **synthesis** — answer dispersed across many speeches
- **out_of_corpus** — no answer in the debates, by design

The `out_of_corpus` set deliberately mixes *thematic far* (e.g. 2023 baccalauréat
results) and *thematic close* questions (e.g. school-bullying-related but
unaddressed in the debates), to test the guardrail under varying difficulty.

## Relevance rule

A chunk is annotated as relevant **only if it would be cited to answer the
question** — not if it merely mentions the topic. The rule is binary and strict.

A few ambiguous cases were resolved consistently across the set:

- *Actor vs. target.* For "who implements the measures?", chunks naming
  agents of the dispositif (educators, school nurses, CLEMI…) are relevant.
  Chunks naming entities *subject to* obligations (platforms, parents)
  are not — they would answer a different question. This distinction was
  applied uniformly to keep retrieval metrics interpretable.
- *Numbers as words.* "Two to three children per class" was treated as a
  valid quantitative answer to "how many children?", on par with explicit
  counts. The criterion is whether the chunk *answers*, not whether it
  contains a numeral.
- *Overlapping sub-chunks.* When two consecutive sub-chunks (`part_N`,
  `part_N+1`) share an overlap sentence and convey essentially the same
  content, only the more complete one is annotated relevant. Both annotated
  only when they contribute distinct information.
- *Multi-text sittings.* The source sittings cover several unrelated bills
  (sports, family name law…). Each annotated chunk was checked to belong
  to the bullying debate proper, not to neighbouring topics in the same
  sitting.

## Iteration

Annotation was revised as evaluation runs surfaced gaps:

- Initial runs missed a few relevant chunks (e.g. on suicide statistics,
  on the list of measures). The retrieval itself surfaced them; they were
  added after manual verification by keyword search.
- One question initially marked `out_of_corpus` (deaths attributed to
  bullying) turned out to be in-corpus — multiple speakers cite figures
  (18-19 suicides over a year). Reclassified as factual.

This iterative loop is part of the methodology, not a flaw to hide:
honest annotation is built by cycles, and metrics improve as the annotation
becomes more faithful to the corpus.

## Choices and limits

- **Binary relevance.** Graded relevance (e.g. nDCG) would better capture
  partial answers, especially for synthesis questions. Binary was kept for a
  first evaluation: simpler to apply consistently, simpler to interpret.
- **Manual annotation, single annotator.** A second annotator (inter-rater
  agreement) would strengthen the labels' validity. With 17 questions and a
  single domain, the iterative loop above was the substitute.
- **Sample size.** 17 questions is small. Reported numbers should be read
  as *indicative*, not as production benchmarks; the per-type breakdowns
  (n=2 to n=6) are especially noisy. Confidence intervals would be wider
  than the differences between configurations on some types.
- **Question-corpus mismatch known up front.** Synthesis questions are
  expected to score low on this corpus (dispersed answers across speakers).
  Including them is deliberate: they document a real limit, and form the
  baseline against which the planned addition of the law text will be
  measured.

## Files

- `questions.jsonl` — one JSON object per line: `id`, `question`, `type`,
  `in_corpus`, `relevant_chunk_ids`, `expected_answer`.
- `run_eval.py` — loads the set, runs retrieval, computes recall@k, hit@k,
  MRR, and the in/out score gap; supports `--rerank` and `--hybrid` to
  evaluate alternative configurations on the same questions.
