# Méthodologie d'évaluation

Comment le jeu d'évaluation a été construit, et ce que ses chiffres peuvent
ou ne peuvent pas dire.

🇬🇧 *English version: [methodology.md](methodology.md)*

## Jeu de questions

17 questions, formulées du point de vue d'un utilisateur n'ayant pas lu les
débats, *avant* de consulter le corpus. Les formulations sont restées
naturelles (pas de reprise du vocabulaire des chunks) pour éviter le piège
du recall trivial où le système « gagne » en matchant ses propres mots.

Le jeu couvre cinq catégories aux profils de retrieval distincts :

- **factual** — réponse unique et localisée (chiffres, noms, articles)
- **conceptual** — définition ou caractérisation
- **disagreement** — positions à reconstruire entre orateurs
- **synthesis** — réponse dispersée à travers plusieurs interventions
- **out_of_corpus** — pas de réponse dans les débats, par construction

Le sous-ensemble `out_of_corpus` mélange volontairement des questions
*thématiquement lointaines* (par ex. taux de réussite au bac 2023) et
*thématiquement proches* (liées au harcèlement scolaire mais non abordées
dans les débats), pour tester le garde-fou à difficulté variable.

## Règle de pertinence

Un chunk est annoté pertinent **uniquement si je le citerais pour répondre
à la question** — pas s'il évoque simplement le sujet. La règle est
binaire et stricte.

Quelques cas ambigus ont été tranchés de façon cohérente sur l'ensemble du
jeu :

- *Acteur vs. cible.* Pour « qui met en œuvre les mesures ? », sont
  pertinents les chunks nommant des agents du dispositif (personnels
  éducatifs, infirmières scolaires, CLEMI…). Ne le sont pas les chunks
  nommant des entités *soumises à* des obligations (plateformes, parents)
  — ces dernières répondraient à une autre question. La distinction est
  appliquée uniformément pour préserver l'interprétabilité des métriques.
- *Les chiffres comme mots.* « Deux à trois enfants par classe » a été
  traité comme une réponse quantitative valide à « combien d'enfants ? »,
  au même titre qu'un chiffre brut. Le critère est de savoir si le chunk
  *répond*, pas s'il contient un nombre.
- *Sous-chunks chevauchants.* Quand deux sous-chunks consécutifs (`part_N`,
  `part_N+1`) partagent une phrase d'overlap et véhiculent essentiellement
  le même contenu, seul le plus complet est annoté pertinent. Les deux ne
  sont gardés que s'ils apportent des éléments distincts.
- *Séances multi-textes.* Les séances sources couvrent plusieurs textes
  sans rapport (loi sport, loi nom de famille…). Chaque chunk annoté a été
  vérifié comme relevant bien du débat sur le harcèlement et non d'un sujet
  voisin de la même séance.

## Itération

L'annotation a été révisée à mesure que les évaluations révélaient des
trous :

- Les premières évaluations manquaient quelques chunks pertinents (par ex.
  sur les chiffres de suicides, sur la liste des mesures). C'est le
  retrieval lui-même qui les a fait remonter ; ils ont été ajoutés après
  vérification manuelle par recherche mot-clé.
- Une question initialement marquée `out_of_corpus` (décès attribués au
  harcèlement) s'est révélée in-corpus — plusieurs orateurs citent des
  chiffres (18-19 suicides sur l'année). Reclassée en factual.

Cette boucle itérative fait partie de la méthodologie, ce n'est pas un
défaut à dissimuler : une annotation honnête se construit par cycles, et
les métriques s'améliorent à mesure que l'annotation devient plus fidèle
au corpus.

## Choix et limites

- **Pertinence binaire.** Une pertinence graduée (par ex. nDCG) capturerait
  mieux les réponses partielles, notamment pour les questions de synthèse.
  Le binaire a été conservé pour une première évaluation : plus simple à
  appliquer de façon cohérente, plus simple à interpréter.
- **Annotation manuelle, annotateur unique.** Un second annotateur (accord
  inter-juges) renforcerait la validité des étiquettes. Avec 17 questions
  et un seul domaine, la boucle itérative ci-dessus en a tenu lieu.
- **Taille de l'échantillon.** 17 questions, c'est peu. Les chiffres
  rapportés doivent se lire comme *indicatifs*, pas comme des benchmarks
  de production ; les ventilations par type (n=2 à n=6) sont particulièrement
  bruyantes. Des intervalles de confiance seraient plus larges que les
  écarts entre configurations sur certains types.
- **Décalage question-corpus connu d'avance.** Les questions de synthèse
  sont attendues comme plus difficiles sur ce corpus (réponses dispersées
  entre orateurs). Leur inclusion est volontaire : elles documentent une
  vraie limite, et constituent la baseline contre laquelle l'ajout prévu
  du texte de loi sera mesuré.

## Fichiers

- `questions.jsonl` — un objet JSON par ligne : `id`, `question`, `type`,
  `in_corpus`, `relevant_chunk_ids`, `expected_answer`.
- `run_eval.py` — charge le jeu, lance le retrieval, calcule recall@k,
  hit@k, MRR et le score gap in/out ; supporte `--rerank` et `--hybrid`
  pour évaluer des configurations alternatives sur les mêmes questions.
