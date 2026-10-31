# Credits

Attribution for all third-party code, libraries, datasets and assets used in Hirewall.

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
