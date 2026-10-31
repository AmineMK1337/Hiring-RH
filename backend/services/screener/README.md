# Screener Service

Screens and ranks candidate resumes against job offers. It ranks, flags and
explains; it never shortlists or rejects anyone. Only a recruiter does that,
through the approval queue.

## Status

| Part | File | Status |
|---|---|---|
| Data models | `models.py` | Done (semantic + combined ranking) |
| Config | `config.py` | Done (semantic + ranking weights) |
| Semantic score | `scoring/semantic.py`, `scoring/chunking.py` | Done, tested |
| Embeddings (shared) | `core/llm/embeddings.py` | Done, tested |
| Relevant experience | `scoring/experience.py` | Done, tested |
| Skill taxonomy | `graph_store/skill_taxonomy.json`, `graph_store/taxonomy.py` | Seeded, `related` links to curate |
| Combined ranking | `scoring/ranking_engine.py` | Done, tested (graph criterion N/A until Neo4j) |
| Graph score (Neo4j) | `graph_store/neo4j_writer.py`, a `GraphScorer` | Next |
| Rasch scoring | `scoring/rasch_scoring.py` | Not planned (see below) |
| Profile extraction | `extract.py` | Todo, needs `security/` output |
| Justification | `justify.py` | Todo |
| Orchestrator node | `node.py` | Todo |

## Inputs / Outputs

**Input:** `CandidateDocument` (from `security/` after sanitization):
pseudonymous `candidate_id`, visible text sections only (no name, contact
details, photo, age, address or hidden text), and `injection_flag`.
Plus a `JobOffer` with typed requirements (`required`, `preferred`,
`responsibility`).

**Output:** `CombinedRanking` (shape in `docs/api-contract.md`): the
reference date `as_of`, the parsed required years, the configured weights,
and per candidate the total score, per-criterion scores (`null` = N/A),
applied criteria and weights used, matched / missing / additional skills,
experience details with a short explanation, the injection flag and the
semantic evidence. `SemanticRanking` (semantic part only) is unchanged.

## Combined ranking

`rank_combined(job, candidates, semantic_index, as_of=...)` combines:

| Criterion | Source | N/A when |
|---|---|---|
| `semantic` | `SemanticIndex` | never (candidate indexed) |
| `experience` | `scoring/experience.py` | the job states no years requirement |
| `graph` | a `GraphScorer` (Neo4j, later) | the scorer is unavailable (`NullGraphScorer` today) |

**N/A rule:** a criterion scored `None` is left out of the total and the other
weights are renormalized, so nobody is penalized for something the job does
not ask for. Default weights: semantic 0.5, graph 0.3, experience 0.2
(`SCREENER_WEIGHT_*`). Without a graph scorer that is about 0.71 / 0.29.

**Experience:** only dated positions in `experience` sections count; a
position is relevant when its own lines mention a job skill (taxonomy name
or alias) or role term. Overlapping jobs count once. Dates in English,
French and Arabic (Middle-East and Tunisian month names, Eastern Arabic
digits), numeric dates (`03/2021`, `2021-03`), "present" in all three
languages. `as_of` replaces the system clock, so results are reproducible.

## Fairness choices

- No bonus beyond the required years: the experience score is
  `min(1, relevant / required)`. Extra years would otherwise favor older
  candidates.
- Unrelated experience counts for nothing, whatever its length or start date.
- Education, certifications and university names are not scored (excluded
  until explicitly added). Projects and skills lists do not count as
  experience.
- `additional_skills` is for display only and never changes a score.
- No criterion uses age, gender, nationality, photo, address or marital
  status; the input does not even contain them.
- Known limit: a resume without dated positions gets 0 relevant years when
  the job has a requirement (most of the older sample candidates). The
  explanation says so; a recruiter should check those resumes.

## How it works

1. `SemanticIndex.upsert()` splits each resume into sentence/bullet chunks,
   deduplicates them, caps them at 60, and embeds them once.
2. `SemanticIndex.rank()` embeds each requirement as a query. For every
   candidate, each requirement takes its best-matching chunk.
3. Similarities are calibrated to 0..1 between `floor` and `ceiling`, then
   averaged with weights (required 2, preferred 1, responsibility 1).

Default model: `intfloat/multilingual-e5-small`, run locally, so candidate
text never leaves the machine. Resumes can be in English, French or Arabic.

## How to test alone

```bash
cd backend
pip install -r requirements.txt
pytest services/screener -q                                       # offline
cd .. && python scripts/demo_screener_semantic.py --job job-backend-python
python scripts/demo_screener_semantic.py --job job-data-analyst --fake   # offline
python scripts/demo_screener_ranking.py --job job-backend-python-3y --fake --as-of 2026-11-01
```

Tests: `test_screener_semantic.py` (semantic), `test_screener_experience.py`
(dates and requirements in three languages, overlaps, fairness),
`test_screener_ranking.py` (N/A renormalization, determinism, ties, no PII),
`test_screener_taxonomy.py`.

To regenerate the taxonomy (overwrites curated `related` links):

```bash
git clone --depth 1 https://github.com/arpitpandey454507/NLP-Powered-Resume-Job-Matching-Candidate-Ranking-System /tmp/ref-ranker
python scripts/seed_taxonomy.py --source /tmp/ref-ranker
```

## Calibrate before trusting scores

The default `floor=0.78, ceiling=0.90` is a starting guess for E5. Run the
demo with the real model and read `raw_similarity` in the evidence. Set
`floor` just above typical unrelated pairs and `ceiling` near typical good
matches. Override with `SCREENER_SEMANTIC_FLOOR` / `SCREENER_SEMANTIC_CEILING`.

## Security and fairness notes

- Input must be sanitized visible text. Hidden keyword stuffing inflates
  semantic scores, so hidden text must be removed upstream.
- Repeating a line adds nothing, but a visible skills list with no supporting
  experience can still score well semantically (`cand-008`, `cand-011` in the
  sample data). The experience criterion, the graph score and human review
  exist for that.
- `injection_flag` is carried through, never used to filter.
- Evidence text is for the recruiter UI. Never pass it to an LLM without
  fencing it as untrusted data.

## Why no Rasch scoring

`rasch_scoring.py` comes from HireMatch AI, where it needs a large real
dataset to estimate skill difficulty. With synthetic data the estimates are
meaningless and hard to explain. The graph score covers skill coverage with
related-skill credit instead.

## Owner

TBD
