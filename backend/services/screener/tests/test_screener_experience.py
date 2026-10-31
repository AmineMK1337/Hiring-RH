from datetime import date

import pytest

from services.screener.models import ResumeSection
from services.screener.scoring.experience import (
    experience_score,
    extract_intervals,
    merge_intervals,
    parse_required_years,
    relevant_experience,
    split_positions,
    total_months,
)

AS_OF = date(2026, 11, 1)
BACKEND_TERMS = {"python", "rest api", "rest apis", "api rest", "django", "backend"}


def months(text: str) -> int:
    return total_months(extract_intervals(text, as_of=AS_OF))


def experience(*texts: str, required: float | None = 3.0):
    sections = [ResumeSection(kind="experience", text=t) for t in texts]
    return relevant_experience(sections, BACKEND_TERMS, required_years=required, as_of=AS_OF)


# ---------------------------------------------------------------- required years

@pytest.mark.parametrize("text, expected", [
    ("3+ years of Python", 3),
    ("Minimum 5 years of backend work", 5),
    ("At least 4 years of experience", 4),
    ("2-4 years of experience", 2),
    ("2 to 4 years", 2),
    ("5 years of experience, including 2 years with Kubernetes", 5),
    ("Experienced in Salesforce cloud.10+ years in SFDC", 10),   # no space after the period
])
def test_required_years_english(text, expected):
    assert parse_required_years(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("3 ans d'expérience en Python", 3),
    ("Minimum 2 ans", 2),
    ("Au moins 3 ans", 3),
    ("2 à 4 ans d'expérience", 2),
    ("Expérience : 3 ans", 3),
    ("1,5 ans d’expérience", 1.5),
])
def test_required_years_french(text, expected):
    assert parse_required_years(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("خبرة 3 سنوات", 3),
    ("3 سنوات خبرة", 3),
    ("خبرة سنتين", 2),
    ("خبرة ثلاث سنوات", 3),
    ("خبرة ٣ سنوات في بايثون", 3),
    ("خبرة لا تقل عن ٤ سنوات", 4),
    ("من 2 إلى 4 سنوات خبرة", 2),
])
def test_required_years_arabic(text, expected):
    assert parse_required_years(text) == expected


@pytest.mark.parametrize("text", [
    "Build REST APIs in Python",
    "Contrat de 3 ans",           # contract length, not experience
    "Company founded 10 years ago",
    "Experience: 60 years",       # outside 0..50
    "",
])
def test_no_requirement(text):
    assert parse_required_years(text) is None


# ---------------------------------------------------------------- date ranges

@pytest.mark.parametrize("text, expected_months", [
    ("Jan 2020 - Dec 2022", 36),
    ("January 2020 to December 2020", 12),
    ("2018 - 2020", 36),                         # whole years
    ("mars 2021 - juin 2021", 4),
    ("sept. 2018 – févr. 2019", 6),
    ("janvier 2019 à décembre 2019", 12),
    ("يناير ٢٠١٩ - ديسمبر ٢٠٢٠", 24),          # Eastern Arabic names and digits
    ("أفريل 2021 - جوان 2022", 15),              # Tunisian names
    ("جويلية 2019 إلى أوت 2020", 14),
    ("03/2021 - 05/2022", 15),
    ("2021-03 - 2022-05", 15),
])
def test_interval_extraction(text, expected_months):
    assert months(text) == expected_months


@pytest.mark.parametrize("text", [
    "Jan 2026 - Present",
    "janvier 2026 - aujourd'hui",
    "janvier 2026 - en cours",
    "جانفي 2026 - حاليا",
    "جانفي 2026 إلى الآن",
])
def test_present_in_all_languages_ends_at_as_of(text):
    assert months(text) == 11          # Jan 2026 .. Nov 2026


def test_open_range_since():
    assert months("Depuis mars 2026") == 9
    assert months("منذ 2026") == 11


@pytest.mark.parametrize("text", [
    "2022 - 2020",                     # start after end
    "1940 - 1945",                     # before 1950
    "Dec 1949 - Jan 1951",             # December is stored as year * 12 + 12
    "Jan 2027 - Present",              # starts in the future
    "13/2021 - 05/2022",               # no month 13
])
def test_impossible_dates_rejected(text):
    assert extract_intervals(text, as_of=AS_OF) == []


def test_future_end_is_cut_at_as_of():
    assert months("Jan 2026 - Dec 2028") == 11


def test_end_date_glued_to_next_word():
    # Common in text extracted from PDF/HTML resumes
    assert months("Engineer,07/2015 to 08/2016Apex Systems") == 14
    assert months("2019 - presentation skills") == 0


def test_overlapping_jobs_counted_once():
    intervals = extract_intervals("Jan 2020 - Dec 2021\nJun 2021 - Jun 2022", as_of=AS_OF)
    assert total_months(intervals) == 30           # Jan 2020 .. Jun 2022


def test_adjacent_jobs_merged():
    merged = merge_intervals([(2020 * 12 + 1, 2020 * 12 + 6), (2020 * 12 + 7, 2020 * 12 + 12)])
    assert merged == [(2020 * 12 + 1, 2020 * 12 + 12)]


# ---------------------------------------------------------------- positions

def test_split_positions_keeps_headers_and_bullets_together():
    text = (
        "Acme (fictional)\nBackend developer\nJan 2020 - Dec 2021\n- Built REST APIs in Python\n\n"
        "Call center agent | 2015 - 2019\nAnswered customer calls."
    )
    positions = split_positions(text)
    assert len(positions) == 2
    assert "Acme" in positions[0] and "REST APIs" in positions[0]
    assert "Answered customer calls" in positions[1]


# ---------------------------------------------------------------- relevant experience

def test_unrelated_experience_counts_zero():
    result = experience("Call center agent | 2010 - 2020\n- Answered customer calls.")
    assert result.total_years == 11
    assert result.relevant_years == 0
    assert result.score == 0.0


def test_only_relevant_positions_count():
    result = experience(
        "Backend developer | Jan 2022 - Dec 2023\n- Built REST APIs in Python\n\n"
        "Call center agent | 2015 - 2021\n- Answered customer calls."
    )
    assert result.relevant_years == 2
    assert result.positions_relevant == 1 and result.positions_dated == 2


def test_relevance_in_french_and_tunisian_arabic():
    fr = experience("Développeuse backend | mars 2021 - février 2024\n- API REST en Python")
    ar = experience("مطورة | من جانفي 2021 إلى ديسمبر 2023\n- تطوير REST APIs باستعمال Python")
    assert fr.relevant_years == 3
    assert ar.relevant_years == 3


def test_education_and_projects_are_ignored():
    sections = [
        ResumeSection(kind="education", text="Master in Python engineering | 2015 - 2020"),
        ResumeSection(kind="project", text="Python REST API side project | 2019 - 2024"),
    ]
    result = relevant_experience(sections, BACKEND_TERMS, required_years=3, as_of=AS_OF)
    assert result.total_years == 0 and result.score == 0.0


def test_score_none_without_requirement():
    result = experience("Backend developer | 2020 - 2022\n- Python", required=None)
    assert result.score is None
    assert result.relevant_years == 3


def test_exceeding_requirement_gives_exactly_one():
    assert experience("Backend developer | 2010 - 2024\n- Python").score == 1.0
    assert experience_score(15, 3) == 1.0
    assert experience_score(1.5, 3) == 0.5
    assert experience_score(0, 0) == 1.0


def test_older_unrelated_jobs_do_not_change_the_result():
    relevant = "Backend developer | 2021 - 2023\n- Python REST APIs\n\n"
    young = experience(relevant + "Call center agent | 2018 - 2020")
    older = experience(relevant + "Call center agent | 1995 - 2020")
    assert young.score == older.score
    assert young.relevant_years == older.relevant_years
