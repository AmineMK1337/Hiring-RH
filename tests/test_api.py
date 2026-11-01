"""API tests in demo mode (no LLM, no MongoDB needed).

    pip install pytest httpx
    python -m pytest -q
"""
import os
import time
from pathlib import Path

import pytest

os.environ["LLM_PROVIDER"] = "demo"
os.environ["STORE"] = "json"

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("STORE_PATH", str(tmp_path / "store.json"))
    import importlib

    import api.store
    importlib.reload(api.store)
    import api.tasks
    importlib.reload(api.tasks)
    import api.main
    importlib.reload(api.main)
    from fastapi.testclient import TestClient
    return TestClient(api.main.app)


def wait(client, task_id):
    for _ in range(200):
        t = client.get(f"/api/tasks/{task_id}").json()
        if t["status"] == "done":
            return t
        time.sleep(0.05)
    raise TimeoutError


def test_health(client):
    body = client.get("/api/health").json()
    assert body["llm"]["provider"] == "demo"


def test_demo_candidates_and_injection(client):
    r = client.post("/api/candidates/demo").json()
    assert r["count"] == 13
    flagged = [c for c in client.get("/api/candidates").json() if c["injection"]["flagged"]]
    assert len(flagged) == 2
    # the injected skill never reaches the profile that is scored
    for c in flagged:
        assert not any("instructions" in s.lower() for s in c["skills"])
    # loading twice adds nothing
    assert client.post("/api/candidates/demo").json()["count"] == 13


def test_match_ranks_and_scores(client):
    client.post("/api/candidates/demo")
    task = client.post("/api/match", json={"job_title": "Backend Python Developer",
                                           "required_skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
                                           "min_experience": 3, "num_candidates": 3}).json()
    t = wait(client, task["id"])
    results = t["results"]
    assert len(results) == 13
    assert [r["rank"] for r in results] == list(range(1, 14))
    scores = [r["fit_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= r["score"] <= 1 for r in results)
    assert results[0]["profile"] == "Backend"


def test_upload_pdf_with_hidden_injection(client):
    data = (ROOT / "demo_data/cvs/attack_hidden_text.pdf").read_bytes()
    task = client.post("/api/candidates/upload", files=[("files", ("cv.pdf", data, "application/pdf"))]).json()
    t = wait(client, task["id"])
    assert "removed" in t["messages"][0]["text"]
    cand = client.get("/api/candidates").json()[0]
    assert cand["injection"]["flagged"]


def test_upload_rejects_bad_files(client):
    task = client.post("/api/candidates/upload", files=[
        ("files", ("fake.pdf", b"MZ not a pdf", "application/pdf")),
        ("files", ("run.exe", b"MZ", "application/octet-stream")),
    ]).json()
    t = wait(client, task["id"])
    assert all(m["level"] == "error" for m in t["messages"])
    assert client.get("/api/candidates").json() == []


def test_decisions_are_recorded_with_audit(client):
    client.post("/api/candidates/demo")
    entry = client.post("/api/decisions", json={"job_key": "backend", "candidate_id": "C-01",
                                                "status": "shortlisted"}).json()
    assert entry["status"] == "shortlisted"
    assert client.get("/api/decisions", params={"job_key": "backend"}).json() == {"C-01": "shortlisted"}
    assert client.get("/api/audit").json()[-1]["candidate_id"] == "C-01"


def test_match_without_candidates_is_400(client):
    assert client.post("/api/match", json={"job_title": "Anything"}).status_code == 400
