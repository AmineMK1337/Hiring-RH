# Adapted from https://github.com/arpitpandey454507/NLP-Powered-Resume-Job-Matching-Candidate-Ranking-System
# (MIT License, Copyright (c) 2024 AI Resume Screener)
#
# Adapted from src/scoring.py:
#   parse_required_years  <- _extract_required_experience
#   extract_intervals     <- _extract_experience_intervals
#   merge_intervals       <- _merge_intervals
#   total_months          <- _interval_months
#   relevant_experience   <- _estimate_relevant_experience, _experience_match
# Changes: French and Arabic (incl. Tunisian month names and Eastern Arabic
# digits), numeric dates, a fixed reference date instead of datetime.now(),
# relevance judged per position instead of per section, and a linear score
# with no bonus beyond the requirement.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Job-relevant experience.

Only dated positions in "experience" sections count. A position counts as
relevant when its own lines mention one of the job's skills or role terms, so
twenty years in an unrelated job add nothing. The score reaches 1.0 at the
required years and never goes higher: more years than required earn no bonus,
which would otherwise favor older candidates.

Everything is deterministic: "present" means the `as_of` date passed in by the
caller, never the system clock.
"""
import re
from collections.abc import Iterable
from datetime import date

from services.screener.models import ExperienceResult, ResumeSection
from services.screener.scoring.normalize import fold, term_pattern

MIN_START_YEAR = 1950
MAX_REQUIRED_YEARS = 50

# ---------------------------------------------------------------- required years
# All patterns run on folded text (lowercase, no accents, ASCII digits).

# Not part of a longer number or a decimal ("12.5"), but "cloud.10+ years" is fine
_NUM = r"(?<!\d)(?<!\d[.,])\d{1,2}(?:[.,]\d)?"
_UNIT = r"(?:years?|yrs?|ans|annees?|سنوات|سنين|سنة|سنه|اعوام|عاما|عام)"
_AR_UNIT = r"(?:سنوات|سنين|سنة|سنه|اعوام|عاما|عام)"
_AR_NUMBER_WORDS = {
    "ثلاث": 3, "ثلاثة": 3, "اربع": 4, "اربعة": 4, "خمس": 5, "خمسة": 5,
    "ست": 6, "ستة": 6, "سبع": 7, "سبعة": 7, "ثمان": 8, "ثماني": 8, "ثمانية": 8,
    "تسع": 9, "تسعة": 9, "عشر": 10, "عشرة": 10,
}
# Words that carry their own unit: the Arabic dual ("two years") and "one year"
_AR_UNIT_WORDS = {
    "سنتين": 2, "سنتان": 2, "عامين": 2, "عامان": 2, "سنة واحدة": 1, "عام واحد": 1,
}


def _alternation(words) -> str:
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


# Ranges are checked first, so "2-4 years" is not read as "4 years"
_RANGE = re.compile(
    rf"(?P<lo>{_NUM})\s*(?:-|–|—|to|a|et|and|الى|حتى|و)\s*(?P<hi>{_NUM})\s*\+?\s*{_UNIT}(?!\w)"
)
_QUANTITY = re.compile(
    rf"(?:(?P<num>{_NUM})\s*(?P<plus>\+)?\s*{_UNIT}"
    rf"|(?P<word>{_alternation(_AR_NUMBER_WORDS)})\s+{_AR_UNIT}"
    rf"|(?P<unit_word>{_alternation(_AR_UNIT_WORDS)}))(?!\w)"
)
_EXPERIENCE_WORD = re.compile(r"(?<!\w)(?:experiences?|exp|خبرة|الخبرة|خبره|خبرات)(?!\w)")
_MINIMUM_BEFORE = re.compile(
    r"(?:minimum(?:\s+(?:of|de))?|min\.?|at\s+least|au\s+moins|au\s+minimum|pas\s+moins\s+de"
    r"|لا\s+تقل\s+عن|ما\s+لا\s+يقل\s+عن|على\s+الاقل|كحد\s+ادنى)\s*$"
)
_MINIMUM_AFTER = re.compile(r"^\s*(?:minimum|at\s+least|au\s+moins|على\s+الاقل|كحد\s+ادنى)")
_CLAUSE_END = re.compile(r"[;\n•!?]|\.(?!\d)")
_WINDOW = 40


def _to_float(value: str) -> float:
    return float(value.replace(",", "."))


def _clause_window(text: str, start: int, end: int) -> tuple[str, str]:
    """Up to 40 characters on each side of a match, cut at sentence boundaries."""
    before = text[max(0, start - _WINDOW):start]
    boundaries = list(_CLAUSE_END.finditer(before))
    if boundaries:
        before = before[boundaries[-1].end():]
    after = text[end:end + _WINDOW]
    boundary = _CLAUSE_END.search(after)
    if boundary:
        after = after[:boundary.start()]
    return before, after


def _is_requirement(text: str, match: re.Match) -> bool:
    """A bare "3 ans" may be a contract length: require some experience context."""
    if match["plus"]:
        return True
    before, after = _clause_window(text, match.start(), match.end())
    return bool(
        _MINIMUM_BEFORE.search(before)
        or _MINIMUM_AFTER.search(after)
        or _EXPERIENCE_WORD.search(before)
        or _EXPERIENCE_WORD.search(after)
    )


def _quantity(match: re.Match) -> float:
    if match["num"]:
        return _to_float(match["num"])
    if match["word"]:
        return float(_AR_NUMBER_WORDS[match["word"]])
    return float(_AR_UNIT_WORDS[match["unit_word"]])


def parse_required_years(job_text: str) -> float | None:
    """Minimum years of experience stated in a job text, or None if there is none.

    For a range the lower bound counts. Several single values: the largest wins
    ("5+ years, including 2 with Kubernetes" -> 5). Values outside 0..50 are ignored.
    """
    if not job_text:
        return None
    text = fold(job_text)

    lows = []
    for m in _RANGE.finditer(text):
        lo, hi = _to_float(m["lo"]), _to_float(m["hi"])
        if 0 <= lo <= hi <= MAX_REQUIRED_YEARS:
            lows.append(lo)
    if lows:
        return min(lows)

    values = [
        value
        for m in _QUANTITY.finditer(text)
        if _is_requirement(text, m) and 0 <= (value := _quantity(m)) <= MAX_REQUIRED_YEARS
    ]
    return max(values) if values else None


# ---------------------------------------------------------------- date ranges

_MONTH_NAMES = {
    # English
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    # French (folded: no accents)
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12,
    "janv": 1, "fevr": 2, "fev": 2, "avr": 4, "juil": 7,
    # Arabic, Middle East
    "يناير": 1, "فبراير": 2, "مارس": 3, "أبريل": 4, "إبريل": 4, "مايو": 5,
    "يونيو": 6, "يونيه": 6, "يوليو": 7, "يوليه": 7, "أغسطس": 8, "سبتمبر": 9,
    "أكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12,
    # Arabic, Maghreb / Tunisia
    "جانفي": 1, "فيفري": 2, "أفريل": 4, "ماي": 5, "جوان": 6, "جويلية": 7,
    "جويليه": 7, "أوت": 8,
}
MONTHS = {fold(name): number for name, number in _MONTH_NAMES.items()}

_YEAR = r"(?:19|20)\d{2}"
_PRESENT = (
    r"(?:present|current|now|today|ongoing|aujourd['’]hui|ce\s+jour|actuel(?:le)?|actuellement"
    r"|en\s+cours|حاليا|الان|اليوم|الوقت\s+الحالي)"
)
_SEPARATOR = r"\s*(?:-|–|—|->|→|to|until|till|a|au|jusqu['’]?\s*(?:a|au)|الى|حتى)\s*"
_SINCE = r"(?:since|depuis|منذ)\s+"


def _point(prefix: str) -> str:
    """A date: "march 2021", "03/2021", "2021-03" or a bare year."""
    p = prefix
    return (
        rf"(?:(?<!\w)(?P<{p}_month>{_alternation(MONTHS)})\.?,?\s+(?:de\s+)?(?P<{p}_year>{_YEAR})"
        rf"|(?<!\d)(?P<{p}_num_month>\d{{1,2}})\s*[/.]\s*(?P<{p}_num_year>{_YEAR})"
        rf"|(?P<{p}_iso_year>{_YEAR})-(?P<{p}_iso_month>\d{{1,2}})"
        rf"|(?<!\d)(?P<{p}_bare_year>{_YEAR}))(?!\d)"
    )


# A numeric end date may be glued to the next word ("08/2016Apex", common in
# text extracted from PDF/HTML); a "present" word must end at a word boundary.
_DATE_RANGE = re.compile(
    rf"{_point('start')}{_SEPARATOR}(?:{_point('end')}|(?P<present>{_PRESENT})(?!\w))"
)
_OPEN_RANGE = re.compile(rf"(?<!\w){_SINCE}{_point('start')}")


def _month_index(match: re.Match, prefix: str, is_end: bool) -> int | None:
    """year * 12 + month, or None if the month is impossible."""
    if match[f"{prefix}_month"]:
        month, year = MONTHS[match[f"{prefix}_month"]], match[f"{prefix}_year"]
    elif match[f"{prefix}_num_month"]:
        month, year = int(match[f"{prefix}_num_month"]), match[f"{prefix}_num_year"]
    elif match[f"{prefix}_iso_year"]:
        month, year = int(match[f"{prefix}_iso_month"]), match[f"{prefix}_iso_year"]
    else:
        # A bare year covers the whole year: January as a start, December as an end
        month, year = (12 if is_end else 1), match[f"{prefix}_bare_year"]
    if not 1 <= month <= 12:
        return None
    return int(year) * 12 + month


def extract_intervals(text: str, *, as_of: date) -> list[tuple[int, int]]:
    """Employment periods in a text, as (start, end) month indexes (year * 12 + month).

    "present" and open ranges ("depuis mars 2021") end at `as_of`. Periods that
    start before 1950, start after `as_of`, or end before they start are dropped.
    An end date after `as_of` (a planned contract end) is cut at `as_of`.
    """
    if not text:
        return []
    folded = fold(text)
    today = as_of.year * 12 + as_of.month
    found: list[tuple[int | None, int | None]] = []
    taken: list[tuple[int, int]] = []

    for m in _DATE_RANGE.finditer(folded):
        start = _month_index(m, "start", is_end=False)
        end = today if m["present"] else _month_index(m, "end", is_end=True)
        found.append((start, end))
        taken.append(m.span())

    for m in _OPEN_RANGE.finditer(folded):
        if any(m.start() < span_end and m.end() > span_start for span_start, span_end in taken):
            continue
        found.append((_month_index(m, "start", is_end=False), today))

    intervals = []
    for start, end in found:
        if start is None or end is None:
            continue
        end = min(end, today)
        if (start - 1) // 12 < MIN_START_YEAR or start > today or start > end:
            continue
        intervals.append((start, end))
    return intervals


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent periods, so no month is counted twice."""
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def total_months(intervals: list[tuple[int, int]]) -> int:
    """Months covered, both ends included: Jan 2020 - Dec 2020 is 12 months."""
    return sum(end - start + 1 for start, end in merge_intervals(intervals))


# ---------------------------------------------------------------- relevant experience

def has_date_range(line: str) -> bool:
    folded = fold(line)
    return bool(_DATE_RANGE.search(folded) or _OPEN_RANGE.search(folded))


def split_positions(text: str) -> list[str]:
    """Split an experience section into positions.

    Blocks are separated by blank lines. In a block, each line with a date range
    starts a position; lines before the block's first dated line (title, company)
    belong to that position, other lines belong to the position above them.
    A block without any date continues the previous position.
    """
    positions: list[list[str]] = []
    for block in re.split(r"\n\s*\n", text):
        header: list[str] = []
        started = False
        for line in (ln for ln in block.splitlines() if ln.strip()):
            if has_date_range(line):
                positions.append([line] if started else header + [line])
                header, started = [], True
            elif started:
                positions[-1].append(line)
            else:
                header.append(line)
        if header:
            if positions:
                positions[-1].extend(header)
            else:
                positions.append(header)
    return ["\n".join(lines) for lines in positions]


def experience_score(relevant_years: float, required_years: float | None) -> float | None:
    """None without a requirement; otherwise relevant / required, capped at 1.0."""
    if required_years is None:
        return None
    if required_years <= 0:
        return 1.0
    return max(0.0, min(1.0, relevant_years / required_years))


def relevant_experience(
    sections: Iterable[ResumeSection],
    job_skill_terms: Iterable[str],
    *,
    required_years: float | None,
    as_of: date,
) -> ExperienceResult:
    """Relevant and total experience from the candidate's experience sections only.

    Education, projects and skills sections are ignored. A position's period
    counts as relevant only if that position's own lines mention one of
    `job_skill_terms` (skill names, aliases or role terms).
    """
    terms = term_pattern(list(job_skill_terms))
    all_periods: list[tuple[int, int]] = []
    relevant_periods: list[tuple[int, int]] = []
    dated = relevant = 0

    for section in sections:
        if section.kind != "experience":
            continue
        for position in split_positions(section.text):
            periods = extract_intervals(position, as_of=as_of)
            if not periods:
                continue
            dated += 1
            all_periods.extend(periods)
            if terms is not None and terms.search(fold(position)):
                relevant += 1
                relevant_periods.extend(periods)

    total = total_months(all_periods)
    relevant_months = min(total_months(relevant_periods), total)
    relevant_years, total_years = relevant_months / 12, total / 12
    score = experience_score(relevant_years, required_years)

    if dated == 0:
        explanation = "No dated positions found in the experience sections."
    else:
        explanation = (
            f"{relevant_years:.1f} of {total_years:.1f} years are in positions that mention "
            f"the job's skills or role ({relevant} of {dated} dated positions)."
        )
    if required_years is None:
        explanation += " The job states no experience requirement, so this is not scored."
    elif score == 1.0:
        explanation += f" Meets the {required_years:g}-year requirement."
    else:
        explanation += f" Requirement: {required_years:g} years."

    return ExperienceResult(
        relevant_years=round(relevant_years, 2),
        total_years=round(total_years, 2),
        required_years=required_years,
        score=None if score is None else round(score, 4),
        positions_dated=dated,
        positions_relevant=relevant,
        explanation=explanation,
    )
