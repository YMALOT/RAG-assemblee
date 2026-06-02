# RAG sur les débats de l'Assemblée nationale : loi contre le harcèlement scolaire
 
Système de retrieval-augmented generation sur les comptes rendus intégraux
de l'Assemblée nationale concernant la loi contre le harcèlement scolaire
(PPL n°4658, loi n°2022-299, 3 séances). Construit de bout en bout comme
projet de portfolio : pipeline écrit à la main (pas de LangChain), jeu
d'évaluation annoté manuellement, améliorations chiffrées.
 
🇬🇧 *English version: [README.md](README.md)*
 
## Démo interactive

An interactive demo is available on [Huggingface Spaces](https://huggingface.co/spaces/YMalot/rag-assemblee-demo)

## Pipeline
 
`XML syceron → ingest.py → chunk.py → embed_and_index.py → Chroma`
` → retrieve.py → generate.py (Claude Haiku 4.5 via API)`
 
- **Source.** XML open-data (syceron) depuis data.assemblee-nationale.fr,
  licence Etalab. 3 séances, 1209 interventions → 1838 chunks.
- **Chunking.** Un chunk par intervention si elle est courte, sinon
  découpage par phrases entières avec chevauchement d'une phrase ; cible
  ~900 caractères.
- **Embeddings.** `intfloat/multilingual-e5-base`, CPU, préfixes e5
  (`passage:` / `query:`), espace cosinus.
- **Génération.** Backend compatible OpenAI (par défaut API Anthropic pour
  la qualité ; le pipeline tourne aussi sur Ollama en local pour une démo
  open-weights). Le system prompt impose des réponses ancrées dans les
  sources avec citations, et un refus explicite si la réponse n'est pas
  dans le corpus.
## Évaluation
 
17 questions annotées à la main, couvrant 5 types (`factual`, `conceptual`,
`disagreement`, `synthesis`, `out_of_corpus`), avec les chunk_ids pertinents
et les réponses attendues. Règle d'annotation stricte : *un chunk est
pertinent uniquement si je le citerais pour répondre à la question* — les
chunks vaguement liés sont exclus.
 
Trois configurations de retrieval comparées sur le même jeu d'évaluation :
 
| Configuration              | Recall@5 | MRR   | Hit@5 | Score gap in/out |
|----------------------------|---------:|------:|------:|-----------------:|
| Dense (e5) baseline        |   0.451  | 0.750 | 0.929 |           0.015  |
| + Reranker (BGE-v2-m3)     | **0.467**| **0.764** | 0.929 |       **0.340**  |
| + Hybride (BM25 + RRF)     |   0.339  | 0.750 | 0.929 |           0.003  |
 
### Lecture des résultats
 
**Baseline.** Hit@5 élevé (0,93) et MRR (0,75) avec un recall modéré (0,45) :
le système trouve fiablement *un* bon chunk en tête, mais ne récupère
qu'une partie de l'ensemble pertinent quand les réponses sont dispersées —
exactement le profil attendu d'un bi-encodeur dense sur un corpus
mono-thématique. Le détail par type le confirme : factual/conceptual/
disagreement entre 0,47 et 0,63, synthesis à 0,33.
 
**Reranker.** Le cross-encoder apporte un gain modeste sur le classement
(+1,6 pt recall, +1,4 pt MRR), porté par les types les plus difficiles
(synthesis 0,33 → 0,39, factual 0,47 → 0,53). Le vrai gain est ailleurs :
le score gap in/out passe de 0,015 à 0,340 — une multiplication par 23 qui
rend exploitable un seuil de score comme détecteur de hors-corpus, alors
que le score dense seul est quasiment plat. Conservé en production comme
*signal de garde-fou*, en complément de l'instruction « je ne sais pas »
au niveau du prompt.
 
**Hybride (dense + BM25).** Dégrade le recall de 11 points. C'est un
résultat *diagnostique*, pas un échec : le pouvoir discriminatif de BM25
dépend de l'IDF, qui s'effondre sur un corpus mono-thématique où des mots
comme *harcèlement*, *scolaire*, *élèves* sont partout. BM25 finit par
favoriser les chunks qui *répètent le vocabulaire de la question* plutôt
que ceux qui *y répondent*, et la fusion RRF propage ce bruit. L'hybride
redeviendrait probablement intéressant avec une diversification thématique
— par exemple l'ajout du texte de loi lui-même, prochaine extension prévue.
 
## Évaluation de la génération
 
Au-delà du retrieval, les réponses générées ont été évaluées de bout en
bout sur les 17 mêmes questions, avec **Claude Sonnet 4.6 comme juge
LLM-as-judge** — délibérément un modèle différent (et plus capable) que
le générateur (Claude Haiku 4.5) pour limiter le biais d'auto-évaluation.
Le retrieval était en configuration reranked ; le juge recevait la
question, les chunks récupérés, la réponse attendue et la réponse
produite, et rendait un verdict JSON structuré avec justification.
 
| Métrique                        | Résultat                              |
|---------------------------------|---------------------------------------|
| Fidélité (in-corpus, n=14)      | **100% fidèle**, 0% partielle, 0% non fidèle |
| Exactitude (in-corpus, n=14)    | 28,6% correcte, 71,4% partielle, **0% incorrecte** |
| Refus (out-of-corpus, n=3)      | **100% refusé**                        |
 
### Lecture des résultats
 
**La fidélité à 100% valide le garde-fou.** Chaque réponse in-corpus est
intégralement ancrée dans les chunks récupérés — aucune hallucination,
aucune fuite de connaissances générales. Le system prompt qui contraint
la génération aux extraits fournis fait son travail.
 
**Le refus à 100% valide le comportement hors-corpus.** Les trois
questions hors-corpus, y compris la plus subtile (causes du harcèlement
scolaire — thématiquement proche mais non traitée dans les débats), ont
été correctement refusées. L'instruction « je ne sais pas » au niveau du
prompt tient à difficulté variable.
 
**Les 28,6% d'exactitude avec 0% d'incorrect racontent l'histoire la plus
intéressante.** Combiné aux 100% de fidélité, ce résultat signifie que
le système n'invente jamais pour combler : quand le retrieval est
incomplet, la réponse est honnêtement partielle plutôt que confiante et
fausse. Les 71% de réponses partielles mesurent donc la *couverture du
retrieval*, pas la qualité de la génération — cohérent avec le recall@5
dense de 0,47 sur les questions de synthèse. Le système échoue
gracieusement, ce qui est le bon comportement pour un RAG.
 
Cela identifie le **retrieval comme le prochain levier**, pas le
générateur. L'ajout prévu du texte de loi (voir « Suite ») devrait
remonter directement le taux d'exactitude sur les questions de synthèse
en améliorant la couverture, sans modification du prompt de génération.
 
### Notes méthodologiques
 
Un seul jugement par réponse, à température 0 pour la stabilité — rapide
et économique, mais ne mesure pas la variance intra-juge. Un protocole
plus rigoureux re-jugerait les cas ambigus (verdicts partiels) 2-3 fois
et reporterait l'accord ; laissé comme raffinement futur.
 
Voir [eval/methodology.fr.md](eval/methodology.fr.md) pour les principes
d'annotation et d'évaluation.
 
## Notes méthodologiques
 
- **L'annotation est itérative.** Les premières évaluations ont révélé des
  trous (questions où le retrieval remontait des chunks pertinents que
  j'avais manqués). Chaque cycle de diagnostic — inspecter les questions
  qui sous-performent, chercher par mot-clé, corriger l'annotation — a
  amélioré le recall et rendu la métrique plus honnête. L'annotation est
  l'expérience, pas une étiquette ponctuelle.
- **La pertinence binaire stricte est un choix assumé.** Une pertinence
  graduée (nDCG) serait un raffinement ; le binaire garde l'interprétation
  claire pour une première évaluation.
- **La détection de hors-corpus par le score seul est peu fiable** avec e5
  (gap 0,015) — confirmé empiriquement et contourné par le garde-fou au
  niveau du LLM. Le score du reranker rend le seuillage envisageable.
## Stack
 
`Python · Chroma · sentence-transformers (e5-base, BGE-reranker-v2-m3) ·`
`rank-bm25 · client compatible OpenAI (API Anthropic / Ollama)`
 
## Suite
 
- Ajouter le texte de loi (PPL initiale n°4658 + loi finale n°2022-299)
  comme type de document distinct avec un chunking par article. Effet
  attendu sur les questions de synthèse et possible réactivation de
  l'hybride.
- Évaluation de la génération plus robuste : plusieurs jugements par
  réponse pour mesurer l'accord intra-juge.
- Étude d'ablation du chunking : taille cible et chevauchement comme
  hyperparamètres mesurés.

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
