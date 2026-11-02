"""submission/journal.py"""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "submission"

JOURNAL = "JISA"
ISSN = "2214-2126"
GUIDE_URL = ("https://www.sciencedirect.com/journal/"
             "journal-of-information-security-and-applications/publish/guide-for-authors")
HIGHLIGHTS_MIN, HIGHLIGHTS_MAX, HIGHLIGHT_MAX_CHARS = 3, 5, 85
WORD_LIMIT = None
CONTACT_EMAIL = "your-email@university.edu"

MANUAL_ITEMS = ("guide_for_authors_read", "word_limit_verified",
                "cover_letter_reviewed", "genai_declaration_written")
MANUAL_LOG = SUB / "manual_done.jsonl"

def _valid_key(key: str) -> bool:
    return key in MANUAL_ITEMS or key.startswith("doi_manual:")

def mark_done(key: str, note: str) -> None:
    if not _valid_key(key):
        raise KeyError(f"Unknown manual item: {key}")
    if not note.strip():
        raise ValueError("note is required: what was checked, and how")
    rec = {"key": key, "note": note, "at": datetime.now(timezone.utc).isoformat()}
    with MANUAL_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def done_keys() -> set[str]:
    if not MANUAL_LOG.exists():
        return set()
    lines = MANUAL_LOG.read_text(encoding="utf-8").splitlines()
    return {json.loads(l)["key"] for l in lines if l.strip()}
