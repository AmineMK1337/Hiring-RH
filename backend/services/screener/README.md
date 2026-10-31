# Screener Service

Screens and ranks candidate resumes against job offers. It ranks, flags and
explains; it never shortlists or rejects anyone. Only a recruiter does that,
through the approval queue.

## Status

| Part | File | Status |
|---|---|---|
| Data models | `models.py` | Done (semantic part) |
| Config | `config.py` | Done (semantic part) |
| Semantic score | `scoring/semantic.py`, `scoring/chunking.py` | Done, tested |
| Embeddings (shared) | `core/llm/embeddings.py` | Done, tested |
| Graph score (Neo4j) | `graph_store/`, `scoring/ranking_engine.py` | Next |
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

**Output:** `SemanticRanking`: model id, config used (store it for audit),
and per candidate a score in 0..1, requirements met, the injection flag, and
for every requirement the best-matching evidence chunk.

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
  experience can still score well (`cand-008` in the sample data). The graph
  score and human review exist for that.
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
