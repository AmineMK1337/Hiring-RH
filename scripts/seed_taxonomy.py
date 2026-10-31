"""Seed the screener skill taxonomy from the reference repo's SKILL_GROUPS.

    git clone --depth 1 https://github.com/arpitpandey454507/NLP-Powered-Resume-Job-Matching-Candidate-Ranking-System /tmp/ref-ranker
    python scripts/seed_taxonomy.py --source /tmp/ref-ranker

The reference file is parsed with `ast`, never imported, so no code from the
cloned repo is executed. The output keeps software, data, DevOps and frontend
skills only, drops aliases that are too short or ambiguous to match safely in
free text, adds common French spellings, and leaves "related" empty for manual
curation. Rerunning overwrites skill_taxonomy.json, including curated links.
"""
import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "backend/services/screener/graph_store/skill_taxonomy.json"
SOURCE_URL = (
    "https://github.com/arpitpandey454507/"
    "NLP-Powered-Resume-Job-Matching-Candidate-Ranking-System"
)
SOURCE_FILE = "src/utils/keyword_mapping.py"

# Reference category -> ours. Categories not listed here are dropped.
CATEGORY_MAP = {
    "programming_languages": "language",
    "machine_learning_ai": "data",
    "ml_libraries": "data",
    "big_data": "data",
    "generative_ai": "data",
    "databases": "database",
    "cloud": "cloud",
    "devops": "devops",
    "version_control": "tools",
    "backend_api": "backend",
    "methodologies": "methodology",
}

# Per-skill category, overriding CATEGORY_MAP (or keeping a skill whose
# reference category is dropped or missing)
CATEGORY_OVERRIDES = {
    "django": "backend", "flask": "backend", "fastapi": "backend",
    "nodejs": "backend", "express": "backend",
    "react": "frontend", "angular": "frontend", "vue": "frontend", "nextjs": "frontend",
    "html5": "frontend", "css3": "frontend",
    "responsive_design": "frontend", "accessibility": "frontend",
    "linux": "devops",
}

# Aliases of 2 characters or fewer are dropped (py, js, ai, cv, ...), except these
SHORT_ALIAS_ALLOWLIST = {"c#"}

# Ambiguous as plain words in English or French resumes
AMBIGUOUS_ALIASES = {
    "lambda",        # the Greek letter, or any anonymous function
    "s3",
    "rag",
    "express",       # "express interest"; express.js / expressjs are kept
    "torch",
    "spark",         # "spark interest"; apache spark / pyspark are kept
    "helm",          # "at the helm"
    "transformers",  # electrical transformers
    "vue",           # French for "view" ("point de vue")
}

# French spellings, plus safer replacements for dropped ambiguous aliases
ADDED_ALIASES = {
    "docker": ["conteneurisation"],
    "rest_api": ["api rest", "apis rest", "api restful", "apis restful"],
    "machine_learning": ["apprentissage automatique"],
    "deep_learning": ["apprentissage profond"],
    "artificial_intelligence": ["intelligence artificielle"],
    "nlp": [
        "traitement du langage naturel",
        "traitement automatique du langage naturel",
        "traitement automatique des langues",
    ],
    "computer_vision": ["vision par ordinateur"],
    "data_science": ["science des données"],
    "data_analysis": ["analyse de données", "analyse des données"],
    "statistics": ["statistiques", "analyse statistique"],
    "ci_cd": ["intégration continue", "déploiement continu", "livraison continue"],
    "nosql": ["base de données non relationnelle", "bases de données non relationnelles"],
    "agile": ["méthode agile", "méthodes agiles", "méthodologie agile"],
    "accessibility": ["accessibilité"],
    "responsive_design": ["design responsive"],
    "helm": ["helm chart", "helm charts"],
    "s3": ["aws s3"],
}

# Not in the reference repo: added by Hirewall for its own sample jobs
LOCAL_ADDITIONS = {
    "power_bi": ("data", ["power bi", "powerbi"]),
    "tableau": ("data", ["tableau", "tableau software", "tableau desktop"]),
    "excel": ("data", ["excel", "microsoft excel"]),
    "ab_testing": ("data", ["a/b testing", "ab testing", "a/b test", "a/b tests", "tests a/b"]),
    "go": ("language", ["golang", "go language", "langage go"]),
    "pytest": ("testing", ["pytest"]),
    "automated_testing": ("testing", [
        "automated tests", "automated testing", "unit tests", "unit testing",
        "integration tests", "integration testing", "tests unitaires",
        "tests d'intégration", "tests automatisés",
    ]),
}

MIT_NOTICE = (
    "Permission is hereby granted, free of charge, to any person obtaining a copy of this "
    "software and associated documentation files (the \"Software\"), to deal in the Software "
    "without restriction, including without limitation the rights to use, copy, modify, merge, "
    "publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons "
    "to whom the Software is furnished to do so, subject to the following conditions: The above "
    "copyright notice and this permission notice shall be included in all copies or substantial "
    "portions of the Software. THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, "
    "EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS "
    "FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT "
    "HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF "
    "CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE "
    "USE OR OTHER DEALINGS IN THE SOFTWARE."
)


def read_literals(path: Path, names: set[str]) -> dict:
    """Module-level dict literals from a Python file, without executing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id in names:
            found[target.id] = ast.literal_eval(value)
    missing = names - found.keys()
    if missing:
        sys.exit(f"{path}: {', '.join(sorted(missing))} not found")
    return found


def clean_aliases(aliases: list[str]) -> list[str]:
    out = []
    for alias in aliases:
        alias = " ".join(alias.lower().split())
        if len(alias) <= 2 and alias not in SHORT_ALIAS_ALLOWLIST:
            continue
        if alias in AMBIGUOUS_ALIASES or alias in out:
            continue
        out.append(alias)
    return out


def source_commit(source: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build(source: Path) -> dict:
    literals = read_literals(source / SOURCE_FILE, {"SKILL_GROUPS", "SKILL_CATEGORIES"})
    groups, categories = literals["SKILL_GROUPS"], literals["SKILL_CATEGORIES"]
    ref_category = {skill: cat for cat, skills in categories.items() for skill in skills}

    skills: dict[str, dict] = {}
    for name, aliases in groups.items():
        category = CATEGORY_OVERRIDES.get(name) or CATEGORY_MAP.get(ref_category.get(name, ""))
        if category is None:
            continue
        cleaned = clean_aliases(aliases + ADDED_ALIASES.get(name, []))
        if cleaned:
            skills[name] = {"name": name, "aliases": cleaned, "category": category, "related": []}

    for name, (category, aliases) in LOCAL_ADDITIONS.items():
        if name not in skills:
            skills[name] = {"name": name, "aliases": clean_aliases(aliases),
                            "category": category, "related": []}

    # An alias must point to one skill only
    owner: dict[str, str] = {}
    for skill in skills.values():
        for alias in skill["aliases"]:
            if alias in owner:
                sys.exit(f"alias '{alias}' used by both {owner[alias]} and {skill['name']}")
            owner[alias] = skill["name"]

    return {
        "_meta": {
            "description": "Screener skill taxonomy: canonical skills, aliases, category, "
                           "related skills (curated by hand).",
            "generated_by": "scripts/seed_taxonomy.py",
            "source": SOURCE_URL,
            "source_file": f"{SOURCE_FILE} (SKILL_GROUPS, SKILL_CATEGORIES)",
            "source_commit": source_commit(source),
            "license": "MIT",
            "copyright": "Copyright (c) 2024 AI Resume Screener",
            "license_notice": MIT_NOTICE,
            "local_additions": sorted(LOCAL_ADDITIONS),
            "changes": "Filtered to software/data/DevOps/frontend skills, short and ambiguous "
                       "aliases removed, aliases lowercased, French spellings added, "
                       "categories remapped.",
        },
        "skills": [skills[name] for name in sorted(skills)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/tmp/ref-ranker"),
                        help="local clone of the reference repo (outside this repo)")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    taxonomy = build(args.source)
    args.output.write_text(
        json.dumps(taxonomy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{len(taxonomy['skills'])} skills written to {args.output}")


if __name__ == "__main__":
    main()
