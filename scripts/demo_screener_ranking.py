"""Combined ranking of the sample candidates for one job, with per-criterion scores.

    python scripts/demo_screener_ranking.py --job job-backend-python-3y
    python scripts/demo_screener_ranking.py --job job-backend-python-3y --fake   # no model download
    python scripts/demo_screener_ranking.py --job job-data-analyst --as-of 2026-11-01
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from core.llm.embeddings import HashingEmbedder, SentenceTransformerEmbedder  # noqa: E402
from services.screener.config import EMBEDDING_MODEL, SemanticConfig  # noqa: E402
from services.screener.models import CandidateDocument, JobOffer  # noqa: E402
from services.screener.scoring.ranking_engine import rank_combined  # noqa: E402
from services.screener.scoring.semantic import SemanticIndex  # noqa: E402


def fmt(score: float | None) -> str:
    return "  N/A" if score is None else f"{score:5.2f}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # French and Arabic text on Windows consoles
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default="job-backend-python-3y")
    parser.add_argument("--fake", action="store_true", help="bag-of-words embedder, no download")
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today(),
                        help="reference date for 'present' (YYYY-MM-DD), default today")
    parser.add_argument("--details", type=int, default=3, help="show details for top N")
    args = parser.parse_args()

    jobs_raw = json.loads((ROOT / "data/job_offers/sample_job_offers.json").read_text(encoding="utf-8"))
    cands_raw = json.loads(
        (ROOT / "data/resumes/sample_candidates_sanitized.json").read_text(encoding="utf-8")
    )
    jobs = {j["job_id"]: JobOffer(**j) for j in jobs_raw}
    candidates = [CandidateDocument(**c) for c in cands_raw]
    job = jobs[args.job]

    if args.fake:
        index = SemanticIndex(HashingEmbedder(), SemanticConfig(floor=0.0, ceiling=0.6))
    else:
        index = SemanticIndex(SentenceTransformerEmbedder(EMBEDDING_MODEL))
    index.upsert(candidates)
    ranking = rank_combined(job, candidates, index, as_of=args.as_of)

    required = "none stated" if ranking.required_years is None else f"{ranking.required_years:g} years"
    print(f"\n{job.title}  (semantic model: {ranking.semantic_model_id}, as of {ranking.as_of})")
    print(f"Required experience: {required}.  Configured weights: {ranking.weights_config}\n")
    print(f"{'rank':<5}{'candidate':<11}{'total':>6} {'sem':>6}{'exp':>6}{'graph':>6}  applied")
    for c in ranking.candidates:
        flag = "  [injection flag]" if c.injection_flag else ""
        print(f"{c.rank:<5}{c.candidate_id:<11}{c.total_score:>6.3f} "
              f"{fmt(c.scores.semantic):>6}{fmt(c.scores.experience):>6}{fmt(c.scores.graph):>6}  "
              f"{'+'.join(c.applied_criteria)}{flag}")

    for c in ranking.candidates[: args.details]:
        weights = ", ".join(f"{k} {w:.2f}" for k, w in c.weights_used.items())
        print(f"\n{c.candidate_id}  (weights used: {weights})")
        print(f"  experience: {c.experience.explanation}")
        print(f"  matched:    {', '.join(c.matched_skills) or '-'}")
        print(f"  missing:    {', '.join(c.missing_skills) or '-'}")
        print(f"  additional: {', '.join(c.additional_skills) or '-'}  (display only)")


if __name__ == "__main__":
    main()
