import json

from services.screener.graph_store.taxonomy import TAXONOMY_PATH, Skill, SkillTaxonomy, default_taxonomy
from services.screener.scoring.normalize import fold


def test_fold_removes_case_accents_and_arabic_marks():
    assert fold("DÉVELOPPÉ") == "developpe"
    assert fold("أفريل ٢٠٢١") == "افريل 2021"
    assert fold("حالياً") == "حاليا"


def test_taxonomy_file_credits_its_source():
    meta = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))["_meta"]
    assert meta["license"] == "MIT"
    assert "arpitpandey454507" in meta["source"]


def test_aliases_are_lowercase_and_unique():
    raw = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))["skills"]
    aliases = [a for s in raw for a in s["aliases"]]
    assert all(a == a.lower() for a in aliases)
    assert len(aliases) == len(set(aliases))
    assert all(s["related"] == [] for s in raw)


def test_alias_lookup():
    tax = default_taxonomy()
    assert tax.canonical("Postgres") == "postgresql"
    assert tax.canonical("K8s") == "kubernetes"
    assert tax.canonical("conteneurisation") == "docker"
    assert tax.canonical("not a skill") is None


def test_find_skills_in_english_french_and_arabic_text():
    tax = default_taxonomy()
    assert tax.find_skills("Built REST APIs in Python, deployed with Docker on K8s.") == {
        "rest_api", "python", "docker", "kubernetes"}
    assert tax.find_skills("Développement d'API REST, intégration continue") == {"rest_api", "ci_cd"}
    assert tax.find_skills("تطوير واجهات REST API باستعمال Python") == {"rest_api", "python"}


def test_whole_words_only():
    tax = default_taxonomy()
    assert tax.find_skills("JavaScript developer") == {"javascript"}   # not "java"
    assert tax.find_skills("point de vue, spark interest, at the helm") == set()
    assert tax.find_skills("C++ and C# services") == {"c++", "c#"}


def test_terms_for_returns_aliases():
    tax = SkillTaxonomy([Skill("docker", ("docker", "conteneurisation"), "devops")])
    assert tax.terms_for({"docker"}) == {"docker", "conteneurisation"}
