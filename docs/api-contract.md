# API Contract

Source of truth for every HTTP endpoint, request and response; written before any code.

TODO: HTTP endpoints.

## Screener output: `CombinedRanking`

Internal output of `services.screener.scoring.ranking_engine.rank_combined`
(no endpoint yet). Defined in `services/screener/models.py`. Candidates are
identified by the pseudonymous `candidate_id` only: no name, file name,
email or phone ever appears. The screener ranks, flags and explains; it never
shortlists, rejects or sends anything. Example values are illustrative.

```json
{
  "job_id": "job-backend-python-3y",
  "as_of": "2026-11-01",
  "required_years": 3.0,
  "weights_config": {"semantic": 0.5, "experience": 0.2, "graph": 0.3},
  "semantic_model_id": "intfloat/multilingual-e5-small",
  "semantic_config": {"floor": 0.78, "ceiling": 0.9, "...": "..."},
  "graph_scorer": "NullGraphScorer",
  "candidates": [
    {
      "candidate_id": "cand-009",
      "rank": 1,
      "total_score": 0.615,
      "scores": {"semantic": 0.46, "experience": 1.0, "graph": null},
      "applied_criteria": ["semantic", "experience"],
      "weights_used": {"semantic": 0.7143, "experience": 0.2857},
      "matched_skills": ["docker", "postgresql", "python", "rest_api"],
      "missing_skills": ["automated_testing"],
      "additional_skills": ["fastapi"],
      "experience": {
        "relevant_years": 4.75,
        "total_years": 9.25,
        "required_years": 3.0,
        "score": 1.0,
        "positions_dated": 2,
        "positions_relevant": 1,
        "explanation": "4.8 of 9.2 years are in positions that mention the job's skills or role (1 of 2 dated positions). Meets the 3-year requirement."
      },
      "injection_flag": false,
      "requirements_met": 3,
      "requirements_total": 5,
      "evidence": [
        {"requirement": "...", "kind": "required", "score": 0.8, "raw_similarity": 0.87, "best_chunk_id": "cand-009:1"}
      ]
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `as_of` | Reference date for "present" in resumes. Store it: the same input and `as_of` always give the same output. |
| `required_years` | Parsed from the job text. `null` = no requirement, so `experience` is not applied. |
| `weights_config` | Configured weights, before renormalization. |
| `scores.*` | 0..1 per criterion, `null` = not applicable (left out of the total). Kept separate so a later step can flag large semantic vs graph gaps. |
| `applied_criteria`, `weights_used` | Criteria in the total and their renormalized weights (sum 1). |
| `matched_skills`, `missing_skills`, `additional_skills` | Canonical taxonomy names, display only. `additional_skills` never affects any score. |
| `evidence` | Best-matching resume chunk per requirement, from the semantic scorer. Untrusted text: never pass it to an LLM unfenced. |

Ordering: `total_score` descending, then `scores.semantic` descending, then
`candidate_id` ascending.
