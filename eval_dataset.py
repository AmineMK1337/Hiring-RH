"""Run the Hiring-RH pipeline on the layan009/RESUMES-JOBS-FIT-LABELS dataset
and compare its picks with the human fit labels.

Usage:
    python eval_dataset.py                                   # provider/model from .env
    python eval_dataset.py --provider ollama --model qwen2.5:3b
    python eval_dataset.py --jd JD01 JD09                    # a subset of jobs
    python eval_dataset.py --reverse                         # also match each CV against every job offer

Every LLM result is cached under data/eval_cache/<provider>_<model>, so an
interrupted run (rate limit, Ctrl+C) picks up where it stopped.
"""
import argparse
import csv
import json
import os
import re
import statistics
import sys
import urllib.request
import zipfile

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_community.document_loaders import Docx2txtLoader
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

import llm as llm_config
from schema import ResumeData, HRFormData, Justification
from utils import justify_candidate, rank_candidates, RETRY_ON_BAD_JSON

HF_BASE = "https://huggingface.co/datasets/layan009/RESUMES-JOBS-FIT-LABELS/resolve/main"
DATA_DIR = "./data/RESUMES-JOBS-FIT-LABELS"
TOP_K = 5  # every job has exactly 5 "Good Fit" resumes
LABELS = ["Good Fit", "Potential Fit", "No Fit"]
# Bump when the justification prompt or schema changes: results are cached per version,
# while HR forms and extracted resumes are reused.
PROMPT_VERSION = "v2"
JUSTIFICATIONS = f"justifications_{PROMPT_VERSION}"

# Set in main()
llm = None
cache_dir = None


# Download & unzip the dataset
def ensure_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in ["resume_metadata.csv", "Resumes.zip"]:
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            print(f"⬇️ Downloading {name}...")
            urllib.request.urlretrieve(f"{HF_BASE}/{name}", path)
    if not os.path.isdir(os.path.join(DATA_DIR, "cv_files")):
        with zipfile.ZipFile(os.path.join(DATA_DIR, "Resumes.zip")) as z:
            z.extractall(DATA_DIR)

    with open(os.path.join(DATA_DIR, "resume_metadata.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


# Cache helpers
def cached(kind, key, model_cls, compute):
    path = os.path.join(cache_dir, kind, f"{key}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return model_cls(**json.load(f))
    result = compute()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
    return result


# What an HR user would type into the form for a given JD
class HRFormFields(BaseModel):
    job_title: str = Field(..., description="Job title")
    required_skills: list[str] = Field(..., description="Up to 10 key required skills")
    min_experience: int = Field(..., description="Minimum years of experience (0 if not stated)")
    additional_criteria: str = Field(..., description="Other important requirements, max 2 sentences")


def jd_to_hr_form(jd_id, jd_text):
    parser = PydanticOutputParser(pydantic_object=HRFormFields)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an HR assistant filling a screening form from a job description."),
        ("human", "{text}\n{format_instructions}")
    ]).partial(format_instructions=parser.get_format_instructions())
    chain = (prompt | llm_config.with_schema(llm, HRFormFields) | parser).with_retry(**RETRY_ON_BAD_JSON)
    fields = cached("hr_forms", jd_id, HRFormFields, lambda: chain.invoke({"text": jd_text}))
    return HRFormData(**fields.model_dump(), num_candidates=TOP_K, job_description=jd_text)


# Same extraction chain as app.py
def extract_resume(resume_id, docx_path):
    parser = PydanticOutputParser(pydantic_object=ResumeData)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract structured resume data from the given document."),
        ("human", "{text}\n{format_instructions}")
    ]).partial(format_instructions=parser.get_format_instructions())
    chain = (prompt | llm_config.with_schema(llm, ResumeData) | parser).with_retry(**RETRY_ON_BAD_JSON)

    def compute():
        text = "\n".join(d.page_content for d in Docx2txtLoader(docx_path).load())
        return chain.invoke({"text": text})

    return cached("resumes", resume_id, ResumeData, compute)


def count_labels(picked):
    return {label: sum(c["label"] == label for c in picked) for label in LABELS}


def evaluate_jd(jd_id, rows):
    hr_input = jd_to_hr_form(jd_id, rows[0]["job_description_text"])
    print(f"\n=== {jd_id}: {hr_input.job_title} ===")
    print(f"    skills: {', '.join(hr_input.required_skills)} | min exp: {hr_input.min_experience}y")

    candidates = []
    for i, row in enumerate(rows, start=1):
        rid = row["resume_id"]
        try:
            resume = extract_resume(rid, os.path.join(DATA_DIR, row["file_path"]))
            justification = cached(JUSTIFICATIONS, rid, Justification,
                                   lambda: justify_candidate(resume, hr_input, llm))
        except OutputParserException as e:
            # Model kept returning bad JSON for this resume: skip it, like the app does
            print(f"    ❌ {rid}: {str(e)[:200]}")
            continue
        candidates.append({"id": rid, "label": row["label"], "resume": resume,
                           "justification": justification})
        print(f"    [{i:2}/{len(rows)}] {rid} fit={justification.fit_score:3}  ({row['label']})", flush=True)

    # New ranking (LLM fit score, as in utils.find_best_candidates) vs. the original skill-count ranking
    by_score = rank_candidates(candidates)[:TOP_K]
    by_skills = sorted(candidates, key=lambda x: len(x["resume"].skills), reverse=True)[:TOP_K]

    print(f"    Top {TOP_K} by fit score:")
    for c in by_score:
        print(f"      {c['id']:9} {c['label']:13} fit={c['justification'].fit_score}")
    return {"jd": jd_id, "title": hr_input.job_title, "processed": len(candidates),
            "score": count_labels(by_score), "skills": count_labels(by_skills),
            "fit_by_label": {label: [c["justification"].fit_score for c in candidates if c["label"] == label]
                             for label in LABELS}}


def print_summary(results):
    n = len(results)
    print(f"\n=== Summary: top {TOP_K} picks per job vs labels (5 Good / 8 Potential / 7 No Fit per job) ===")
    print(f"{'JD':5} {'resumes':>7}   {'fit-score ranking':>18}   {'skill-count ranking':>19}  title")
    print(f"{'':5} {'':>7}   {'Good/Pot/No':>18}   {'Good/Pot/No':>19}")
    for r in results:
        s, k = r["score"], r["skills"]
        print(f"{r['jd']:5} {r['processed']:>7}   {s['Good Fit']:>10}/{s['Potential Fit']}/{s['No Fit']}"
              f"   {k['Good Fit']:>11}/{k['Potential Fit']}/{k['No Fit']}  {r['title']}")

    for name, key in [("fit-score ranking (new)", "score"), ("skill-count ranking (old)", "skills")]:
        precision = sum(r[key]["Good Fit"] for r in results) / (TOP_K * n)
        print(f"Precision@{TOP_K} {name:26}: {precision:.2f}")
    print(f"Precision@{TOP_K} {'random baseline':26}: 0.25")

    print("\nMean fit score by true label:")
    for label in LABELS:
        scores = [s for r in results for s in r["fit_by_label"][label]]
        if scores:
            print(f"  {label:13} {statistics.mean(scores):5.1f}  (n={len(scores)})")


# Reverse direction: score each CV against every job offer and check that its own job ranks first
def evaluate_reverse(rows, by_jd, labels):
    hr_forms = {jd_id: jd_to_hr_form(jd_id, jd_rows[0]["job_description_text"])
                for jd_id, jd_rows in sorted(by_jd.items())}
    print(f"\n=== Reverse test: each CV ({' + '.join(labels)}) vs all {len(hr_forms)} job offers ===")

    results = []
    selected = [r for r in rows if r["label"] in labels]
    for i, row in enumerate(selected, start=1):
        rid, own_jd = row["resume_id"], row["jd_id"]
        try:
            resume = extract_resume(rid, os.path.join(DATA_DIR, row["file_path"]))
            scores = {}
            for jd_id, hr_input in hr_forms.items():
                # The CV vs its own job is the same call as the forward test, so it shares that cache entry
                key = rid if jd_id == own_jd else f"{rid}__vs__{jd_id}"
                scores[jd_id] = cached(JUSTIFICATIONS, key, Justification,
                                       lambda: justify_candidate(resume, hr_input, llm)).fit_score
        except OutputParserException as e:
            print(f"    ❌ {rid}: {str(e)[:200]}")
            continue

        own = scores[own_jd]
        rank = 1 + sum(s > own for s in scores.values())
        tied = sum(s == own for jd, s in scores.items() if jd != own_jd)
        # Ties split the places they share, so an all-zero row doesn't count as a #1
        results.append({"id": rid, "label": row["label"], "rank": rank, "tied": tied,
                        "avg_rank": rank + tied / 2, "own_zero": own == 0})
        ranking = "  ".join(f"{jd}={s}{'*' if jd == own_jd else ''}"
                            for jd, s in sorted(scores.items(), key=lambda x: -x[1]))
        print(f"    [{i:2}/{len(selected)}] {rid:9} rank={rank}{' (tie)' if tied else ''}  {ranking}", flush=True)

    if results:
        print(f"\n=== Reverse summary (* = the CV's own job; random baseline top-1 = {1 / len(hr_forms):.2f}) ===")
        for label in labels:
            group = [r for r in results if r["label"] == label]
            if group:
                top1 = sum(r["rank"] == 1 and not r["tied"] for r in group) / len(group)
                top1_ties = sum(r["rank"] == 1 for r in group) / len(group)
                mean_rank = statistics.mean(r["avg_rank"] for r in group)
                own_zero = sum(r["own_zero"] for r in group)
                print(f"  {label:13} n={len(group):3}  own job ranked #1: {top1:.2f} "
                      f"(incl. ties: {top1_ties:.2f})  mean rank: {mean_rank:.2f} "
                      f"(random: {(1 + len(hr_forms)) / 2:.1f})  own job scored 0: {own_zero}")


def main():
    global llm, cache_dir

    ap = argparse.ArgumentParser()
    ap.add_argument("--jd", nargs="*", help="Job description IDs to run (default: all)")
    ap.add_argument("--provider", choices=["groq", "ollama"], default=llm_config.LLM_PROVIDER)
    ap.add_argument("--model", help="Model name (default: GROQ_MODEL / OLLAMA_MODEL from .env)")
    ap.add_argument("--reverse", action="store_true",
                    help="Also score each CV against every job offer (CV -> best job)")
    ap.add_argument("--reverse-labels", nargs="+", default=["Good Fit"], choices=LABELS,
                    help="Which CVs to use for the reverse test (default: Good Fit)")
    args = ap.parse_args()

    model = args.model or (llm_config.OLLAMA_MODEL if args.provider == "ollama" else llm_config.GROQ_MODEL)
    llm = llm_config.get_llm(args.provider, model)
    cache_dir = os.path.join("./data/eval_cache", re.sub(r"[^\w.-]", "-", f"{args.provider}_{model}"))
    print(f"🤖 {args.provider} / {model}  (cache: {cache_dir}, prompt {PROMPT_VERSION})")

    rows = ensure_dataset()
    by_jd = {}
    for r in rows:
        by_jd.setdefault(r["jd_id"], []).append(r)

    results = []
    for jd_id in sorted(args.jd or by_jd):
        try:
            results.append(evaluate_jd(jd_id, by_jd[jd_id]))
        except Exception as e:
            print(f"\n⛔ Stopped ({type(e).__name__}: {str(e)[:300]})\n   Rerun later to continue from the cache.")
            break

    if results:
        print_summary(results)

    if args.reverse:
        try:
            evaluate_reverse(rows, by_jd, args.reverse_labels)
        except Exception as e:
            print(f"\n⛔ Stopped ({type(e).__name__}: {str(e)[:300]})\n   Rerun later to continue from the cache.")


if __name__ == "__main__":
    sys.exit(main())
