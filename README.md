# Hiring-RH

AI-assisted resume screening for recruiters. Upload resumes, describe the job,
and get a ranked list of candidates with a written justification and three
sub-scores for each. The model ranks and explains; the recruiter decides.

## How it works

1. **Upload** PDF or DOCX resumes in the Streamlit app.
2. **Extract**: an LLM turns each resume into a structured profile (titles,
   companies, dates, skills, education, certifications), stored in MongoDB.
3. **Describe the job**: title, required skills, minimum years, extra
   criteria, and optionally the full job description (recommended).
4. **Score**: for every candidate the LLM writes its reasoning first, then
   scores three things from 0 to 10 against an explicit rubric:
   - **skills**: share of the job's key skills the candidate shows
   - **experience**: years and level in the same kind of work
   - **field**: how close the candidate's field and past roles are to the job

   The fit score (0-100) is computed from these three, not guessed directly.
5. **Rank**: candidates are sorted by fit score and shown with their
   justification, sub-scores and SWOT analysis.

The candidate's name is not sent to the scoring prompt, and experience in an
unrelated field does not count toward the experience score.

## LLM providers

| Provider | Setting | Notes |
|---|---|---|
| Ollama (local) | `LLM_PROVIDER=ollama` | Default model `qwen2.5:3b`; resumes never leave the machine. Output is constrained to the JSON schema. |
| Groq (cloud) | `LLM_PROVIDER=groq` | Default model `openai/gpt-oss-120b`; needs `GROQ_API_KEY`. Free tier: 8K tokens/min, 200K tokens/day. |

## Run locally (Windows)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env          # then edit .env (never commit it)
ollama pull qwen2.5:3b          # if you use Ollama
# MongoDB must be running on localhost:27017

python -m streamlit run app.py  # http://localhost:8501
```

## Talent Orbit web app

A second interface on the same pipeline: a FastAPI backend (`api/`) and a 3D
web page (`frontend/`) where the job offer is the sun and every candidate is a
planet whose distance from it is `100 − fit score`. Closer means a better fit.

**Start it (Windows):** double-click `run.bat`, or:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn api.main:app --port 8000   # then open http://localhost:8000
```

macOS / Linux: `./run.sh`.

It uses the same `.env` as the Streamlit app (`LLM_PROVIDER`, `OLLAMA_MODEL`,
`GROQ_API_KEY`). MongoDB is used when it is running on `localhost:27017`;
otherwise data is kept in `data/orbit_store.json`, so the app works without it.
`LLM_PROVIDER=demo` runs without any LLM (rule-based scores, clearly labeled):
use it to rehearse, never to hire.

**Demo flow**

1. **Load demo candidates** (13 fictional profiles, two contain injected
   instructions), or drop PDF/DOCX resumes. `demo_data/cvs/` has sample
   PDFs, including `attack_hidden_text.pdf` with an instruction written in
   invisible white text.
2. **Describe the job** (or start from a preset) and press **Score
   candidates**. All candidates wait as grey moons on the outer edge, then
   glide inward one by one as the LLM scores them.
3. **Click a planet** for the LLM's justification, the three sub-scores and
   the extracted profile. Shortlist or reject: only your click records a
   decision, with an audit entry.

**How it maps to the pipeline**

| Step | Code |
|---|---|
| Resume → structured profile | `api/pipeline.py: extract_profile` (same `ResumeData` schema, `with_schema` and retries as `app.py`; the prompt marks the resume as untrusted data) |
| Profile → justification and sub-scores | `utils.justify_candidate` (unchanged) |
| Ranking | fit score, then number of skills (same as `utils.rank_candidates`) |
| Orbit distance | `100 − fit_score`; orbital plane = profile family (display only) |

**Security**

- Uploads: PDF/DOCX only, 5 MB max, real file signature checked.
- Prompt injection: lines that instruct the AI (English, French, Arabic) and
  invisible characters are detected *before* the LLM reads the resume. The
  lines are removed, the candidate is flagged (red crystal in the orbit) and
  the recruiter sees the exact text. Structured fields are checked again
  before scoring.
- Blind review is on by default: names are hidden until the recruiter reveals them.
- The LLM never decides. Decisions are stored only on a recruiter's click.

**Tests:** `pip install pytest httpx` then `python -m pytest -q` (demo mode,
no LLM or MongoDB needed).

## Evaluation

`eval_dataset.py` runs the full pipeline on the labeled dataset
[layan009/RESUMES-JOBS-FIT-LABELS](https://huggingface.co/datasets/layan009/RESUMES-JOBS-FIT-LABELS)
(5 jobs, 20 resumes each: 5 Good Fit, 8 Potential Fit, 7 No Fit) and compares
the ranking with the labels. It downloads the dataset into `data/` and caches
every LLM result, so an interrupted run resumes where it stopped.

```powershell
python eval_dataset.py --provider ollama --model qwen2.5:3b            # each job picks its top 5
python eval_dataset.py --provider ollama --model qwen2.5:3b --reverse  # also: each CV vs every job
```

Results with local `qwen2.5:3b` on a 4 GB GPU (about 50 minutes for both tests):

| Metric | Result | Random |
|---|---|---|
| Good Fit resumes in each job's top 5 | 0.40 | 0.25 |
| Average fit score: Good / Potential / No Fit | 46 / 33 / 24 | equal |
| Good Fit CV ranks its own job first (of 5) | 0.40 | 0.20 |

## Limitations

- Small local models are noisy: the same resume can score differently between
  runs, even at temperature 0. A larger model (`qwen2.5:14b`, Groq) is more
  reliable but slower or rate-limited.
- Closely related jobs (for example software developer vs software
  engineering manager) are hard to tell apart.
- The duplicate check in MongoDB uses name and education; anonymized resumes
  with identical placeholder names can collide.
- The final decision always stays with a human recruiter.

## License and credits

Licensed under the GNU GPL v3.0 (see `LICENSE`).

This project is a modified version of an open-source resume screener by
Sindhuja Sivaraman ([original repository](https://github.com/coding-chemist/SmartHire), GPL-3.0).
Changes by AmineMK1337: local LLMs through Ollama, schema-constrained JSON
output, rubric-based sub-scores and computed fit score, full job description
in the prompt, candidate names removed from scoring, ranking by fit score,
retries on malformed output, and the evaluation script.

Evaluation dataset: [layan009/RESUMES-JOBS-FIT-LABELS](https://huggingface.co/datasets/layan009/RESUMES-JOBS-FIT-LABELS).
