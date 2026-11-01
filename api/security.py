"""Security checks for untrusted resumes: upload validation and prompt-injection detection.

Resumes are written by people we don't know, and their text is sent to an LLM.
So before extraction we:
  1. validate the upload (extension, size, real file signature),
  2. detect instructions aimed at the AI (English, French, Arabic) and invisible characters,
  3. remove the offending lines, so they never reach the LLM,
  4. keep a flag and the excerpts, so the recruiter sees what was attempted.
"""
import re
import unicodedata
from dataclasses import asdict, dataclass, field

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
SIGNATURES = {".pdf": b"%PDF", ".docx": b"PK\x03\x04"}

HIDDEN_CHARS = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")

_PATTERNS = [
    # English
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any )?(the )?(previous|prior|above|earlier|system)",
    r"forget (all |everything )?(you|your) (were told|instructions)",
    r"\byou are now\b",
    r"^\s*(system|assistant|developer)\s*:",
    r"<\s*/?\s*(system|instructions?|prompt)\s*>",
    r"\b(rank|score|rate|place|put) (me|this candidate|them|him|her) (first|highest|at the top|10|100)",
    r"\b(give|assign) (me|this candidate|them) (a |the )?(perfect|maximum|max|highest|full) (score|marks|rating)",
    r"\bmark (me |them |this candidate )?as (hired|selected|shortlisted)",
    r"\b(admin|developer|god) mode\b",
    r"\bnew instructions?\s*:",
    r"\bsend (this|my) (resume|cv|file|data) to\b",
    # French
    r"ignore[rz]? (toutes |tous )?(les )?(instructions|consignes|regles) (precedentes|anterieures|ci-dessus)",
    r"oublie[rz]? (toutes |tous )?(les |tes |vos )?(instructions|consignes)",
    r"\b(tu es|vous etes) maintenant\b",
    r"\b(classe|classez|mets|mettez|note|notez)[- ](moi|ce candidat|cette candidate) (en )?(premier|premiere|en tete|10|100)",
    # Arabic
    r"تجاهل (كل |جميع )?(التعليمات|الاوامر)",
    r"انس (كل |جميع )?(التعليمات|الاوامر)",
    r"انت الان",
]
PATTERN = re.compile("|".join(f"(?:{p})" for p in _PATTERNS), re.IGNORECASE | re.MULTILINE)


def fold(text: str) -> str:
    """Lower case, no accents or Arabic diacritics, no hidden characters."""
    text = HIDDEN_CHARS.sub("", text)
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return re.sub(r"[ \t]+", " ", stripped.casefold())


@dataclass
class InjectionReport:
    flagged: bool = False
    hidden_chars: bool = False
    excerpts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def inspect_text(text: str) -> tuple[str, InjectionReport]:
    """Return (clean text, report). Offending lines are removed from the clean text."""
    report = InjectionReport(hidden_chars=bool(HIDDEN_CHARS.search(text or "")))
    kept = []
    for line in HIDDEN_CHARS.sub("", text or "").splitlines():
        if PATTERN.search(fold(line)):
            excerpt = line.strip()[:300]
            if excerpt not in report.excerpts:
                report.excerpts.append(excerpt)
        else:
            kept.append(line)
    report.flagged = bool(report.excerpts)
    return "\n".join(kept), report


class UploadRejected(ValueError):
    pass


def validate_upload(filename: str, data: bytes) -> str:
    """Return the lower-case extension, or raise UploadRejected."""
    name = (filename or "").lower()
    ext = next((e for e in SIGNATURES if name.endswith(e)), None)
    if ext is None:
        raise UploadRejected("only PDF and DOCX files are accepted")
    if len(data) == 0:
        raise UploadRejected("empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadRejected(f"file larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not data.startswith(SIGNATURES[ext]):
        raise UploadRejected(f"content is not a real {ext[1:].upper()} file")
    return ext
