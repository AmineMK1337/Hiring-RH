"""Text normalization shared by skill matching and experience parsing.

Everything is compared in a folded form: Unicode-normalized, case-folded,
accents and Arabic diacritics removed, Eastern Arabic digits turned into
ASCII digits. So "Développé", "developpe" and "DÉVELOPPÉ" compare equal, and
"أفريل ٢٠٢١" matches "افريل 2021".
"""
import re
import unicodedata

# Eastern Arabic (٠-٩) and Persian (۰-۹) digits -> 0-9
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_SPACES = re.compile(r"[ \t ‏‎]+")


def fold(text: str) -> str:
    """Comparable form of a text. Line breaks are kept."""
    decomposed = unicodedata.normalize("NFKD", text.translate(_DIGITS))
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return _SPACES.sub(" ", stripped.casefold())


def term_pattern(terms: list[str]) -> re.Pattern | None:
    """One regex matching any of the (folded) terms as whole words or phrases.

    Longest terms first, so "rest api" wins over "rest". Returns None if empty.
    """
    folded = sorted({fold(t).strip() for t in terms if t.strip()}, key=lambda t: (-len(t), t))
    if not folded:
        return None
    alternation = "|".join(re.escape(t) for t in folded)
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)")
