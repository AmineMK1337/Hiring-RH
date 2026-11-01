"""Storage for the orbit API: MongoDB when it is running, else a local JSON file.

Uses its own collections ("candidates", "decisions", "audit") in the same
"resumes" database as the Streamlit app, so the two never mix documents.
Set STORE=json to force the file store, STORE=mongo to require MongoDB.
"""
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGO_DB", "resumes")
JSON_PATH = Path(os.getenv("STORE_PATH", ROOT / "data" / "orbit_store.json"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JsonStore:
    kind = "json file"

    def __init__(self, path: Path = JSON_PATH):
        self.path = path
        self.lock = threading.Lock()
        self.data = {"candidates": [], "decisions": {}, "audit": []}
        if path.exists():
            self.data.update(json.loads(path.read_text(encoding="utf-8")))

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def list_candidates(self) -> list[dict]:
        return list(self.data["candidates"])

    def find_by_hash(self, content_hash: str) -> dict | None:
        return next((c for c in self.data["candidates"] if c["hash"] == content_hash), None)

    def add_candidate(self, record: dict) -> None:
        with self.lock:
            self.data["candidates"].append(record)
            self._save()

    def clear_candidates(self) -> None:
        with self.lock:
            self.data["candidates"] = []
            self.data["decisions"] = {}
            self._save()

    def set_decision(self, job_key: str, candidate_id: str, status: str) -> None:
        with self.lock:
            self.data["decisions"].setdefault(job_key, {})[candidate_id] = status
            self._save()

    def decisions(self, job_key: str) -> dict:
        return dict(self.data["decisions"].get(job_key, {}))

    def add_audit(self, entry: dict) -> None:
        with self.lock:
            self.data["audit"].append(entry)
            self.data["audit"] = self.data["audit"][-1000:]
            self._save()

    def audit(self, limit: int = 200) -> list[dict]:
        return self.data["audit"][-limit:]


class MongoStore:
    kind = "MongoDB"

    def __init__(self, client):
        db = client[DB_NAME]
        self.c = db["candidates"]
        self.d = db["decisions"]
        self.a = db["audit"]

    def list_candidates(self) -> list[dict]:
        return list(self.c.find({}, {"_id": 0}).sort("created_at", 1))

    def find_by_hash(self, content_hash: str) -> dict | None:
        return self.c.find_one({"hash": content_hash}, {"_id": 0})

    def add_candidate(self, record: dict) -> None:
        self.c.insert_one(dict(record))

    def clear_candidates(self) -> None:
        self.c.delete_many({})
        self.d.delete_many({})

    def set_decision(self, job_key: str, candidate_id: str, status: str) -> None:
        self.d.update_one({"job": job_key, "candidate_id": candidate_id},
                          {"$set": {"status": status, "at": now()}}, upsert=True)

    def decisions(self, job_key: str) -> dict:
        return {d["candidate_id"]: d["status"] for d in self.d.find({"job": job_key})}

    def add_audit(self, entry: dict) -> None:
        self.a.insert_one(dict(entry))

    def audit(self, limit: int = 200) -> list[dict]:
        return list(self.a.find({}, {"_id": 0}).sort("at", -1).limit(limit))[::-1]


def open_store():
    mode = os.getenv("STORE", "auto").lower()
    if mode != "json":
        try:
            from pymongo import MongoClient
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1500)
            client.admin.command("ping")
            return MongoStore(client)
        except Exception as exc:  # noqa: BLE001
            if mode == "mongo":
                raise
            print(f"MongoDB not reachable ({exc.__class__.__name__}), using {JSON_PATH}")
    return JsonStore()
