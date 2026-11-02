"""submission/checks.py — does not import from src/"""
import json, math, re, time
import bibtexparser
from . import journal as J

HEADLINE = J.ROOT / "outputs" / "headline_numbers.json"
BIB = J.SUB / "references.bib"

def check_highlights(highlights: list[str]) -> None:
    hs = [h.strip() for h in highlights]
    if not (J.HIGHLIGHTS_MIN <= len(hs) <= J.HIGHLIGHTS_MAX):
        raise ValueError(f"{len(hs)} highlights; need {J.HIGHLIGHTS_MIN}-{J.HIGHLIGHTS_MAX}")
    for i, h in enumerate(hs, 1):
        if len(h) > J.HIGHLIGHT_MAX_CHARS:
            raise ValueError(f"Highlight {i}: {len(h)} chars > {J.HIGHLIGHT_MAX_CHARS}")

def load_headline() -> dict:
    h = json.loads(HEADLINE.read_text(encoding="utf-8"))
    if h.get("is_synthetic") is not False:
        raise ValueError("headline_numbers.json is synthetic or unflagged")
    bad = [k for k, v in h["values"].items() if v is None or math.isnan(float(v))]
    if bad:
        raise ValueError(f"Missing headline values: {bad}")
    return h

_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%|×|x\b)?")

def check_abstract(abstract: str) -> None:
    found = [(m.group(1), m.group(2)) for m in _NUM.finditer(abstract)
             if "." in m.group(1) or m.group(2)]
    if not found:
        raise ValueError("Abstract has no numerical result")
    vals = [float(v) for v in load_headline()["values"].values()]
    for txt, unit in found:
        x = float(txt) / (100 if unit == "%" else 1)
        d = len(txt.split(".")[1]) if "." in txt else 0
        tol = 0.5 * 10 ** -(d + (2 if unit == "%" else 0))
        if not any(abs(v - x) <= tol for v in vals):
            raise ValueError(f"'{txt}{unit or ''}' is not a headline number")

_PREFIX_TO_JOURNAL = {
    "10.1016/j.jisa.": "journal of information security and applications",
    "10.1016/j.cose.": "computers & security",
    "10.1016/j.jss.":  "journal of systems and software",
}
NO_DOI_OK = {"misc", "online", "techreport"}

def _norm(s: str) -> str:
    return re.sub(r"[^a-z&]+", " ", s.lower()).strip()

def _crossref(e: dict, doi: str) -> tuple[list[str], bool]:
    import requests
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=20,
                         headers={"User-Agent": f"refcheck (mailto:{J.CONTACT_EMAIL})"})
    except requests.RequestException:
        return [], True
    finally:
        time.sleep(1)
    if r.status_code == 404:
        return [f"{e['ID']}: DOI not found on Crossref"], False
    if r.status_code != 200:
        return [], True
    m, out = r.json()["message"], []
    year = str(m["issued"]["date-parts"][0][0])
    if e.get("year") and e["year"] != year:
        out.append(f"{e['ID']}: year {e['year']} vs Crossref {year}")
    if e.get("volume") and m.get("volume") and e["volume"] != m["volume"]:
        out.append(f"{e['ID']}: volume {e['volume']} vs Crossref {m['volume']}")
    return out, False

def check_references(online: bool = False) -> list[str]:
    with BIB.open(encoding="utf-8") as f:
        entries = bibtexparser.load(f).entries
    problems, unreachable, n_jisa = [], [], 0
    for e in entries:
        doi = e.get("doi", "").lower().strip()
        if not doi:
            if e["ENTRYTYPE"] not in NO_DOI_OK:
                problems.append(f"{e['ID']}: no DOI")
            continue
        for prefix, name in _PREFIX_TO_JOURNAL.items():
            if doi.startswith(prefix) and _norm(e.get("journal", "")) != _norm(name):
                problems.append(f"{e['ID']}: DOI belongs to '{name}', "
                                f"journal field says '{e.get('journal')}'")
        if online:
            errs, down = _crossref(e, doi)
            problems += errs
            if down:
                unreachable.append(doi)
        n_jisa += doi.startswith("10.1016/j.jisa.")
    if problems:
        raise ValueError("\n".join(problems))
    print(f"JISA references: {n_jisa} (informational only)")
    return unreachable

def check_draft(highlights, abstract) -> None:
    check_highlights(highlights)
    check_abstract(abstract)
    check_references(online=False)

def assert_ready(highlights, abstract) -> None:
    check_draft(highlights, abstract)
    done = J.done_keys()
    pending = [d for d in check_references(online=True) if f"doi_manual:{d}" not in done]
    missing = [k for k in J.MANUAL_ITEMS if k not in done] + pending
    doi = load_headline().get("zenodo_doi") or ""
    if not re.fullmatch(r"10\.5281/zenodo\.\d+", doi):
        missing.append("zenodo_doi")
    if missing:
        raise ValueError(f"Not ready: {missing}")
