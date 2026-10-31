"""Evaluate the combined screener ranking on a public labeled resume/job dataset.

Dataset: layan009/RESUMES-JOBS-FIT-LABELS (Hugging Face), resume_metadata.csv:
5 jobs x 20 resumes (5 Good, 8 Potential, 7 No Fit each). Every job's resumes
are ranked with several weightings (semantic only, combined, experience only)
and compared with the labels using the same metrics.

The dataset text comes from HTML resumes and is glued together
("Work HistoryStaff Engineer,08/2016toCurrentAcme"). prepare_resume() repairs
the glue, splits sections at their headings and puts each position on its own
line. That is preprocessing for this dataset only, not part of the screener.

Candidate ids are neutral (cand-000, ...): the dataset ids encode the label
(S/P/W) and ties are broken by candidate_id.

    python scripts/eval_screener_ranking_hf.py              # real model (E5, local)
    python scripts/eval_screener_ranking_hf.py --fake       # offline smoke run
    python scripts/eval_screener_ranking_hf.py --details    # per-job rankings
"""
import argparse
import hashlib
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from core.llm.embeddings import HashingEmbedder, SentenceTransformerEmbedder  # noqa: E402
from services.screener.config import EMBEDDING_MODEL, RankingWeights, SemanticConfig  # noqa: E402
from services.screener.graph_store.taxonomy import default_taxonomy  # noqa: E402
from services.screener.models import CandidateDocument, JobOffer, Requirement, ResumeSection  # noqa: E402
from services.screener.scoring.chunking import split_units  # noqa: E402
from services.screener.scoring.experience import parse_required_years  # noqa: E402
from services.screener.scoring.ranking_engine import rank_combined  # noqa: E402
from services.screener.scoring.semantic import SemanticIndex  # noqa: E402

DATASET = "layan009/RESUMES-JOBS-FIT-LABELS"
LABELS = ("Good Fit", "Potential Fit", "No Fit")
GAIN = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
SHORT = {"Good Fit": "G", "Potential Fit": "p", "No Fit": "."}

VARIANTS = {
    "semantic only": RankingWeights(semantic=1.0, experience=0.0, graph=0.0),
    "combined (default 0.5/0.2)": RankingWeights(),
    "combined (0.5/0.5)": RankingWeights(semantic=0.5, experience=0.5, graph=0.0),
    "experience only": RankingWeights(semantic=0.0, experience=1.0, graph=0.0),
}

# ---------------------------------------------------------------- dataset preprocessing

_GLUED_SENTENCE = re.compile(r"(?<=[a-z0-9)][.!?;:])(?=[A-Z])")
_GLUED_TO = re.compile(r"(\d{4})to(?=\d|[A-Z])")                 # "08/2016toCurrent"
_GLUED_PRESENT = re.compile(r"\b(Current|Present)(?=[A-Z])")     # "CurrentLockheed"
_HEADINGS = {
    "experience": ["Professional Experience", "Work Experience", "Work History",
                   "Employment History", "Experience"],
    "education": ["Education and Training", "Education"],
    "skills": ["Technical Skills", "Skill Highlights", "Key Skills", "Summary of Skills",
               "Core Qualifications", "Technical Qualifications", "Qualifications",
               "Areas of Expertise", "Highlights", "Skills"],
    "summary": ["Professional Summary", "Professional Profile", "Executive Profile",
                "Career Overview", "Profile", "Summary"],
}
_HEADING_KIND = {h.casefold(): kind for kind, hs in _HEADINGS.items() for h in hs}
_HEADING = re.compile(
    r"^(" + "|".join(sorted((h for hs in _HEADINGS.values() for h in hs), key=len, reverse=True))
    + r")(?=[A-Z0-9:\-–]|\s*$)"
)
# A position starts at a glue boundary either right before its date
# ("...improvements.12/2010to05/2012Engineer") or shortly before it, at its
# header ("Title,07/2015-Current", "Company-TitleCity,ST,07/2015to..."):
# break the line there.
_POSITION_START = re.compile(
    r"(?<=[a-z.)\]])(?=(?:[A-Z][^\n]{0,80}?)?(?:\d{1,2}/\d{4}\s*(?:to|-|–)"
    r"|(?:19|20)\d{2}\s*-\s*(?:Current|Present|(?:19|20)\d{2})))"
)


def repair(text: str) -> str:
    text = _GLUED_SENTENCE.sub(" ", text)
    text = _GLUED_TO.sub(r"\1 to ", text)
    return _GLUED_PRESENT.sub(r"\1 ", text)


def prepare_resume(text: str) -> list[ResumeSection]:
    sections: list[tuple[str, list[str]]] = [("summary", [])]
    for line in repair(text).split("\n"):
        line = line.strip()
        m = _HEADING.match(line)
        if m:
            sections.append((_HEADING_KIND[m.group(1).casefold()], []))
            line = line[m.end():].lstrip(":-– ")
        if line:
            sections[-1][1].append(line)
    out = []
    for kind, lines in sections:
        body = "\n".join(lines)
        if kind == "experience":
            body = _POSITION_START.sub("\n", body)
        if body.strip():
            out.append(ResumeSection(kind=kind, text=body))
    return out


def header_kind(header: str, current: str) -> str:
    h = header.casefold()
    if any(w in h for w in ("prefer", "nice to have", "bonus", "plus")):
        return "preferred"
    if any(w in h for w in ("responsib", "duties", "what you will do", "what you'll do", "role")):
        return "responsibility"
    if any(w in h for w in ("require", "qualif", "skill", "experience", "must", "education", "need")):
        return "required"
    return current


def job_to_offer(job_id: str, text: str, max_reqs: int) -> JobOffer:
    text = _GLUED_SENTENCE.sub(" ", text)
    reqs: list[Requirement] = []
    kind = "required"
    for line in text.split("\n"):
        m = re.match(r"^([A-Za-z ,&/']{3,60}):\s*", line.strip())
        if m:
            kind = header_kind(m.group(1), kind)
            line = line.strip()[m.end():]
        for unit in split_units(line):
            if len(unit) >= 15:          # drop fragments like "Benefits" or "Apply now"
                reqs.append(Requirement(text=unit[:400], kind=kind))
    title = text.strip().split("\n")[0][:120]
    return JobOffer(job_id=job_id, title=title, requirements=reqs[:max_reqs] or [Requirement(text=text[:400])])


# ---------------------------------------------------------------- metrics

def ndcg(gains: list[int]) -> float:
    discounts = 1 / np.log2(np.arange(2, len(gains) + 2))
    ideal = (np.sort(gains)[::-1] * discounts).sum()
    return float((np.array(gains) * discounts).sum() / ideal) if ideal > 0 else float("nan")


def pairwise_auc(pos: list[float], neg: list[float]) -> float:
    """P(a Good Fit scores above a No Fit); ties count half."""
    if not pos or not neg:
        return float("nan")
    p, n = np.array(pos)[:, None], np.array(neg)[None, :]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def job_metrics(ranked_labels: list[str], scores: list[float]) -> dict[str, float]:
    gains = [GAIN[lbl] for lbl in ranked_labels]
    counts = [ranked_labels.count(lbl) for lbl in LABELS]
    predicted = [lbl for lbl, n in zip(LABELS, counts) for _ in range(n)]
    k = counts[0]
    return {
        "ndcg": ndcg(gains),
        "auc": pairwise_auc([s for s, l in zip(scores, ranked_labels) if l == "Good Fit"],
                            [s for s, l in zip(scores, ranked_labels) if l == "No Fit"]),
        "p@k": sum(lbl == "Good Fit" for lbl in ranked_labels[:k]) / k if k else float("nan"),
        "groups": sum(p == lbl for p, lbl in zip(predicted, ranked_labels)) / len(ranked_labels),
    }


# ---------------------------------------------------------------- main

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true", help="bag-of-words embedder, no download")
    parser.add_argument("--as-of", type=date.fromisoformat, default=date(2024, 6, 1),
                        help="reference date for 'Current' in resumes (dataset is from ~2024)")
    parser.add_argument("--max-reqs", type=int, default=60)
    parser.add_argument("--details", action="store_true", help="per-job rankings vs labels")
    args = parser.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download

    df = pd.read_csv(hf_hub_download(DATASET, "resume_metadata.csv", repo_type="dataset"))
    # Neutral, deterministic ids that say nothing about the label
    order = sorted(df.resume_id, key=lambda rid: hashlib.sha256(rid.encode()).hexdigest())
    cid = {rid: f"cand-{i:03d}" for i, rid in enumerate(order)}
    label = {cid[r]: lbl for r, lbl in zip(df.resume_id, df.label)}
    docs = {cid[r]: CandidateDocument(candidate_id=cid[r], sections=prepare_resume(t))
            for r, t in zip(df.resume_id, df.resume_text)}
    print(f"{DATASET}: {df.jd_id.nunique()} jobs, {len(df)} resumes, labels "
          f"{df.label.value_counts().to_dict()}, as_of {args.as_of}")

    embedder = HashingEmbedder() if args.fake else SentenceTransformerEmbedder(EMBEDDING_MODEL)
    config = SemanticConfig(floor=0.0, ceiling=0.6) if args.fake else SemanticConfig()
    index = SemanticIndex(embedder, config)
    t0 = time.time()
    index.upsert(docs.values())
    print(f"model {embedder.model_id}, indexed in {time.time() - t0:.0f}s\n")

    taxonomy = default_taxonomy()
    results = {name: [] for name in VARIANTS}
    experience_rows = []
    for jd_id, rows in df.groupby("jd_id", sort=True):
        jd_text = rows.job_description_text.iloc[0]
        offer = job_to_offer(jd_id, jd_text, args.max_reqs)
        job_docs = [docs[cid[r]] for r in rows.resume_id]
        print(f"{jd_id}  {offer.title[:70]!r}")
        print(f"   required years: {parse_required_years(jd_text)} (full text), "
              f"job skills found: {sorted(taxonomy.find_skills(jd_text)) or '-'}")

        for name, weights in VARIANTS.items():
            ranking = rank_combined(offer, job_docs, index, as_of=args.as_of, weights=weights)
            ranked_labels = [label[c.candidate_id] for c in ranking.candidates]
            results[name].append(job_metrics(ranked_labels, [c.total_score for c in ranking.candidates]))
            if args.details and name in ("semantic only", "combined (default 0.5/0.2)"):
                print(f"   {name:<27} {' '.join(SHORT[lbl] for lbl in ranked_labels)}")
            if name == "combined (default 0.5/0.2)":
                experience_rows += [(label[c.candidate_id], c.experience) for c in ranking.candidates]
        print()

    print(f"{'ranking':<28}{'NDCG':>7}{'AUC G>No':>10}{'P@5 Good':>10}{'groups':>8}")
    for name, per_job in results.items():
        mean = {m: np.nanmean([r[m] for r in per_job]) for m in per_job[0]}
        print(f"{name:<28}{mean['ndcg']:>7.3f}{mean['auc']:>10.3f}{mean['p@k']:>10.3f}{mean['groups']:>8.3f}")
    print(f"{'random baseline':<28}{'~0.70':>7}{'0.500':>10}{'0.250':>10}{'~0.35':>8}")

    print("\nExperience by label (all 5 jobs; score only where the job states a requirement)")
    print(f"{'label':<15}{'n':>4}{'dated':>8}{'total y':>9}{'relevant y':>12}{'score':>7}{'meets req':>11}")
    for lbl in LABELS:
        exps = [e for l, e in experience_rows if l == lbl]
        scored = [e.score for e in exps if e.score is not None]
        print(f"{lbl:<15}{len(exps):>4}{np.mean([e.positions_dated > 0 for e in exps]):>8.0%}"
              f"{np.mean([e.total_years for e in exps]):>9.1f}{np.mean([e.relevant_years for e in exps]):>12.1f}"
              f"{np.mean(scored) if scored else float('nan'):>7.2f}"
              f"{np.mean([s == 1.0 for s in scored]) if scored else float('nan'):>11.0%}")


if __name__ == "__main__":
    main()
