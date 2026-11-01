"""Hiring-RH API + Talent Orbit web app.

    uvicorn api.main:app --reload        (from the repository root)
    open http://localhost:8000

Same LLM settings as the Streamlit app (.env: LLM_PROVIDER, OLLAMA_MODEL, ...).
LLM_PROVIDER=demo runs without an LLM for rehearsals.

The agent extracts, scores and explains. It never decides: shortlist and
rejection are recorded only when the recruiter clicks, with an audit entry.
"""
import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api import pipeline, tasks
from api.security import MAX_UPLOAD_BYTES
from api.store import now, open_store
from schema import HRFormData, ResumeData

ROOT = Path(__file__).resolve().parents[1]
store = open_store()
app = FastAPI(title="Hiring-RH API", version="1.0.0")

MAX_FILES = 30


class MatchRequest(BaseModel):
    job_title: str = Field(..., min_length=2, max_length=200)
    required_skills: list[str] = Field(default_factory=list, max_length=50)
    min_experience: int = Field(0, ge=0, le=50)
    num_candidates: int = Field(5, ge=1, le=100)
    additional_criteria: str = Field("", max_length=2000)
    job_description: str | None = Field(None, max_length=20000)


class Decision(BaseModel):
    job_key: str = Field(..., max_length=200)
    candidate_id: str = Field(..., max_length=20)
    status: Literal["pending", "shortlisted", "rejected"]


def _public(record: dict) -> dict:
    r = record["resume"]
    return {"id": record["id"], "profile": record["profile"], "source": record["source"],
            "filename": record["filename"], "injection": record["injection"],
            "name": f"{r['name']['first_name']} {r['name']['last_name']}".strip(),
            "current_title": r["job_title"], "skills": r["skills"]}


@app.get("/api/health")
def health():
    return {"status": "ok", "llm": pipeline.llm_info(), "store": store.kind}


@app.get("/api/candidates")
def list_candidates():
    return [_public(c) for c in store.list_candidates()]


@app.post("/api/candidates/upload")
async def upload(files: list[UploadFile] = File(...)):
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"at most {MAX_FILES} files per upload")
    payload = []
    for f in files:
        data = await f.read(MAX_UPLOAD_BYTES + 1)
        payload.append((Path(f.filename or "file").name, data))
    return tasks.start_upload(store, payload)


@app.post("/api/candidates/demo")
def load_demo():
    """Fictional, already-structured profiles (no extraction needed). Scoring still uses the LLM."""
    raw = json.loads((ROOT / "demo_data" / "candidates.json").read_text(encoding="utf-8"))
    messages = []
    for item in raw:
        _, msg = tasks.add_structured(store, ResumeData(**item), "demo", f"demo: {item['name']['first_name']}")
        messages.append(msg)
    return {"messages": messages, "count": len(store.list_candidates())}


@app.delete("/api/candidates")
def clear_candidates():
    store.clear_candidates()
    store.add_audit({"at": now(), "action": "clear_candidates"})
    return {"count": 0}


@app.post("/api/match")
def match(req: MatchRequest):
    if not store.list_candidates():
        raise HTTPException(400, "no candidates yet: upload resumes or load the demo candidates")
    hr = HRFormData(
        job_title=req.job_title.strip(),
        required_skills=[s.strip() for s in req.required_skills if s.strip()],
        min_experience=req.min_experience,
        num_candidates=req.num_candidates,
        additional_criteria=req.additional_criteria.strip(),
        job_description=(req.job_description or "").strip() or None,
    )
    return tasks.start_match(store, hr)


@app.get("/api/tasks/{task_id}")
def task_status(task_id: str):
    task = tasks.TASKS.get(task_id)
    if not task:
        raise HTTPException(404, "unknown task")
    return task


@app.get("/api/decisions")
def get_decisions(job_key: str):
    return store.decisions(job_key)


@app.post("/api/decisions")
def set_decision(d: Decision):
    """Called only from a recruiter's click. The agent never records a decision."""
    store.set_decision(d.job_key, d.candidate_id, d.status)
    entry = {"at": now(), "action": "decision", **d.model_dump()}
    store.add_audit(entry)
    return entry


@app.get("/api/audit")
def audit():
    return store.audit()


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
