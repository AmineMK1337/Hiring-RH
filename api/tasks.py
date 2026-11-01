"""Background tasks: resume ingestion and LLM scoring, with progress for the UI.

LLM calls are slow (seconds to a minute each on a small local model), so the
API starts a task and the page polls it. Scored candidates appear in the orbit
one by one. One worker thread: a local model serves one request at a time anyway.
"""
import hashlib
import json
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from api import pipeline
from api.security import UploadRejected, inspect_text, validate_upload
from api.store import now
from schema import HRFormData, ResumeData

EXECUTOR = ThreadPoolExecutor(max_workers=1)
TASKS: dict[str, dict] = {}
_SCORE_CACHE: dict[str, dict] = {}
_ID_LOCK = threading.Lock()


def _new_task(kind: str, total: int, **extra) -> dict:
    task = {"id": uuid.uuid4().hex[:12], "kind": kind, "status": "queued", "total": total, "done": 0,
            "messages": [], "results": [], "created": time.time(), **extra}
    TASKS[task["id"]] = task
    # keep memory bounded
    for old in sorted(TASKS.values(), key=lambda t: t["created"])[:-50]:
        TASKS.pop(old["id"], None)
    return task


def _say(task: dict, level: str, text: str) -> None:
    task["messages"].append({"level": level, "text": text})


def _llm_hint(exc: Exception) -> str:
    msg = str(exc)
    if pipeline.PROVIDER == "ollama" and ("Connection" in type(exc).__name__ or "connect" in msg.lower()):
        return "cannot reach Ollama. Start it with `ollama serve` and pull the model (see README)."
    if pipeline.PROVIDER == "groq" and ("401" in msg or "api_key" in msg.lower()):
        return "Groq rejected the API key. Set GROQ_API_KEY in .env."
    return msg[:300]


def _next_id(store) -> str:
    with _ID_LOCK:
        taken = {c["id"] for c in store.list_candidates()}
        n = len(taken) + 1
        while f"C-{n:02d}" in taken:
            n += 1
        return f"C-{n:02d}"


def add_structured(store, resume: ResumeData, source: str, filename: str, raw_report: dict | None = None) -> tuple[dict | None, str]:
    """Sanitize a structured profile and store it. Returns (record or None if duplicate, message)."""
    clean, removed = pipeline.sanitize_resume(resume)
    report = raw_report or {"flagged": False, "hidden_chars": False, "excerpts": []}
    for e in removed:
        if e not in report["excerpts"]:
            report["excerpts"].append(e)
    report["flagged"] = bool(report["excerpts"])
    content_hash = hashlib.sha256(json.dumps(clean.model_dump(), sort_keys=True).encode()).hexdigest()
    if store.find_by_hash(content_hash):
        return None, f"{filename}: already uploaded, skipped"
    record = {"id": _next_id(store), "hash": content_hash, "filename": filename, "source": source,
              "created_at": now(), "resume": clean.model_dump(), "injection": report,
              "profile": pipeline.profile_of(clean)}
    store.add_candidate(record)
    note = " (instruction aimed at the AI found and removed)" if report["flagged"] else ""
    return record, f"{filename}: added as {record['id']}{note}"


# ------------------------------------------------------------------ upload

def start_upload(store, files: list[tuple[str, bytes]]) -> dict:
    task = _new_task("upload", len(files))
    EXECUTOR.submit(_run_upload, store, task, files)
    return task


def _run_upload(store, task, files):
    task["status"] = "running"
    for filename, data in files:
        try:
            ext = validate_upload(filename, data)
            text = pipeline.read_document(ext, data)
            if len(text.strip()) < 30:
                raise UploadRejected("no readable text (scanned image PDF?)")
            clean_text, report = inspect_text(text)
            resume = pipeline.extract_profile(clean_text)
            _, msg = add_structured(store, resume, "upload", filename, report.to_dict())
            _say(task, "warn" if report.flagged else "ok", msg)
        except UploadRejected as exc:
            _say(task, "error", f"{filename}: rejected, {exc}")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            _say(task, "error", f"{filename}: extraction failed, {_llm_hint(exc)}")
        task["done"] += 1
    task["status"] = "done"


# ------------------------------------------------------------------ matching

def job_key(hr: HRFormData) -> str:
    return " ".join(hr.job_title.lower().split())


def _cache_key(record: dict, hr: HRFormData) -> str:
    payload = json.dumps([record["hash"], hr.model_dump(exclude={"num_candidates"}), pipeline.PROVIDER], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _result(record: dict, justification: dict) -> dict:
    r = record["resume"]
    return {
        "id": record["id"], "profile": record["profile"], "source": record["source"], "filename": record["filename"],
        "injection": record["injection"],
        "name": f"{r['name']['first_name']} {r['name']['last_name']}".strip(),
        "current_title": r["job_title"], "skills": r["skills"], "experience": r["experience"],
        "education": r["education"], "certifications": r["certifications"],
        "fit_score": justification["fit_score"], "score": justification["fit_score"] / 100,
        "justification": justification,
    }


def rank(results: list[dict]) -> list[dict]:
    # Same order as utils.rank_candidates: fit score, then number of skills
    ordered = sorted(results, key=lambda x: (x["fit_score"], len(x["skills"])), reverse=True)
    for i, r in enumerate(ordered, start=1):
        r["rank"] = i
    return ordered


def start_match(store, hr: HRFormData) -> dict:
    records = store.list_candidates()
    task = _new_task("match", len(records), job=hr.model_dump(), job_key=job_key(hr),
                     pending=[r["id"] for r in records])
    EXECUTOR.submit(_run_match, task, records, hr)
    return task


def _run_match(task, records, hr):
    task["status"] = "running"
    failures = 0
    for record in records:
        key = _cache_key(record, hr)
        try:
            if key not in _SCORE_CACHE:
                resume = ResumeData(**record["resume"])
                _SCORE_CACHE[key] = pipeline.score_candidate(resume, hr).model_dump()
            task["results"] = rank(task["results"] + [_result(record, _SCORE_CACHE[key])])
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            failures += 1
            _say(task, "error", f"{record['id']}: scoring failed, {_llm_hint(exc)}")
            if failures >= 3 and not task["results"]:
                _say(task, "error", "Stopped after 3 failures in a row.")
                break
        task["pending"].remove(record["id"])
        task["done"] += 1
    task["status"] = "done"
