# Credits

Attribution for all third-party code, libraries, datasets and assets used in Hirewall.

## Code adapted

[NLP-Powered Resume Job Matching & Candidate Ranking System](https://github.com/arpitpandey454507/NLP-Powered-Resume-Job-Matching-Candidate-Ranking-System)
(MIT License, Copyright (c) 2024 AI Resume Screener), commit `538d812`.
Adapted files carry the MIT notice in their header.

| Our code | Adapted from (`src/scoring.py` unless noted) | What changed |
|---|---|---|
| `services/screener/scoring/experience.py`: `parse_required_years` | `_extract_required_experience` | French and Arabic patterns, Arabic number words and Eastern Arabic digits, experience context required for bare values |
| `experience.py`: `extract_intervals` | `_extract_experience_intervals` | French, Middle-East and Tunisian Arabic month names, numeric dates, open ranges ("depuis"), fixed `as_of` date instead of `datetime.now()`, future dates cut or dropped |
| `experience.py`: `merge_intervals`, `total_months` | `_merge_intervals`, `_interval_months` | Minor: same logic |
| `experience.py`: `relevant_experience`, `experience_score` | `_estimate_relevant_experience`, `_experience_match` | Relevance per position instead of per section, no role-based fallback, linear score with no 0.8/0.6 steps and no bonus beyond the requirement |
| `services/screener/graph_store/skill_taxonomy.json` (via `scripts/seed_taxonomy.py`) | `src/utils/keyword_mapping.py`: `SKILL_GROUPS`, `SKILL_CATEGORIES` | Filtered to software/data/DevOps/frontend skills, short and ambiguous aliases removed, French spellings and a few local skills added, categories remapped |

Ideas taken from the same repo without copying code: N/A criteria excluded
with weights renormalized (`scoring/ranking_engine.py`), and matched / missing
/ additional skill lists.

## Ideas and design references (no code copied)

| Project | License | What we took |
|---|---|---|
| [SmartHire](https://github.com/coding-chemist/SmartHire) | GPL-3.0 | Idea of LLM structured resume extraction with Pydantic schemas and per-dimension justifications. No code copied, so GPL obligations do not apply. |
| [HireMatch AI](https://github.com/AKR-2803/hirematch-ai) | none stated | Idea of a Neo4j skills graph (Candidate, Skill, Job). No code copied. |

## Libraries and models

| Name | License | Used in |
|---|---|---|
| [sentence-transformers](https://github.com/UKPLab/sentence-transformers) | Apache-2.0 | `core/llm/embeddings.py` |
| [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | MIT | Default screener embedding model |
| [pydantic](https://github.com/pydantic/pydantic) | MIT | Data models |
| [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause | Scoring |
