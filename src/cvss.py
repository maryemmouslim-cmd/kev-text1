"""CVSS v3.0/v3.1 base score from a vector string (FIRST specification, section 7)."""
from __future__ import annotations

import math
import re

_W = {
    "AV": {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2},
    "AC": {"L": 0.77, "H": 0.44},
    "UI": {"N": 0.85, "R": 0.62},
    "C": {"H": 0.56, "L": 0.22, "N": 0.0},
    "I": {"H": 0.56, "L": 0.22, "N": 0.0},
    "A": {"H": 0.56, "L": 0.22, "N": 0.0},
}
_PR = {"U": {"N": 0.85, "L": 0.62, "H": 0.27}, "C": {"N": 0.85, "L": 0.68, "H": 0.5}}
_VEC = re.compile(r"(CVSS:3\.[01]/)?((?:[A-Z]+:[A-Z]/?)+)")


def roundup(x: float) -> float:
    """Spec v3.1 Appendix A: avoids floating-point artefacts of ceil(x*10)/10."""
    i = int(round(x * 100000))
    return i / 100000.0 if i % 10000 == 0 else (math.floor(i / 10000) + 1) / 10.0


def parse_vector(vector: str) -> dict[str, str]:
    m = _VEC.search(vector or "")
    if not m:
        raise ValueError(f"not a CVSS v3 vector: {vector!r}")
    parts = dict(p.split(":") for p in m.group(2).strip("/").split("/") if ":" in p)
    need = {"AV", "AC", "PR", "UI", "S", "C", "I", "A"}
    if not need <= parts.keys():
        raise ValueError(f"incomplete vector: {vector!r}")
    return parts


def base_score(vector: str) -> float:
    m = parse_vector(vector)
    scope_changed = m["S"] == "C"
    iss = 1 - (1 - _W["C"][m["C"]]) * (1 - _W["I"][m["I"]]) * (1 - _W["A"][m["A"]])
    impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15 if scope_changed else 6.42 * iss
    expl = 8.22 * _W["AV"][m["AV"]] * _W["AC"][m["AC"]] * _PR[m["S"]][m["PR"]] * _W["UI"][m["UI"]]
    if impact <= 0:
        return 0.0
    raw = 1.08 * (impact + expl) if scope_changed else impact + expl
    return roundup(min(raw, 10))


def severity(score: float | None) -> str | None:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return None
    return "NONE" if score == 0 else "LOW" if score < 4 else "MEDIUM" if score < 7 \
        else "HIGH" if score < 9 else "CRITICAL"


def try_base_score(vector: str | None) -> float | None:
    try:
        return base_score(vector) if vector else None
    except (ValueError, KeyError):
        return None
