"""Collect and parse the four sources, then merge on cve_id.

Parsers are pure functions on already-downloaded JSON/CSV (tested on fixtures).
fetch_* functions touch the network (run in notebook 01; not tested offline).

Day-1 checks (guide section 15): the NVD history format assumed in
parse_history_events (Description and CVSS details with dates) must be confirmed on
10 real CVEs before the full pull.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

from . import config, cvss
from .io_utils import download, get_json, manifest_event, read_json, atomic_write_json

# ===================================================================== cvelistV5
def parse_cve_record(rec: dict) -> dict:
    """One CVE JSON 5 record -> flat row. ADP containers (CISA Vulnrichment) are ignored."""
    meta = rec.get("cveMetadata", {})
    cna = rec.get("containers", {}).get("cna", {})
    descs = cna.get("descriptions", []) or []
    en = pick_english_description(descs)
    affected = cna.get("affected", []) or []
    vendors = sorted({a.get("vendor", "").strip() for a in affected
                      if a.get("vendor") and a.get("vendor").lower() not in ("n/a", "unknown")})
    products = sorted({a.get("product", "").strip() for a in affected
                       if a.get("product") and a.get("product").lower() not in ("n/a", "unknown")})
    vec = None
    for m in cna.get("metrics", []) or []:
        for key in ("cvssV3_1", "cvssV3_0"):
            if key in m and m[key].get("vectorString"):
                vec = vec or m[key]["vectorString"]
    cwes = []
    for pt in cna.get("problemTypes", []) or []:
        for d in pt.get("descriptions", []) or []:
            if d.get("cweId"):
                cwes.append(d["cweId"])
    return {
        "cve_id": meta.get("cveId"),
        "state": meta.get("state"),
        "published_date": _ts(meta.get("datePublished")),
        "date_updated": _ts(meta.get("dateUpdated")),
        "assigner": meta.get("assignerShortName"),
        "desc_cvelist_current": en,
        "lang_ok": en is not None,
        "vendors": vendors,
        "products": products,
        "vendor": vendors[0] if vendors else None,
        "cna_cvss_vector_current": vec,
        "cna_cwe_current": cwes[0] if cwes else None,
        "has_adp": bool(rec.get("containers", {}).get("adp")),
    }


def pick_english_description(descs: list[dict]) -> str | None:
    for d in descs:
        lang = (d.get("lang") or "").lower()
        if lang.startswith("en") and (d.get("value") or "").strip():
            return d["value"].strip()
    return None


def load_cvelist(repo_dir: Path, years: range) -> pd.DataFrame:
    rows = []
    for y in years:
        for p in (Path(repo_dir) / "cves" / str(y)).rglob("CVE-*.json"):
            rows.append(parse_cve_record(json.loads(p.read_text(encoding="utf-8"))))
    return pd.DataFrame(rows)


def fetch_cvelist(dest: Path = config.RAW / "cvelistV5") -> Path:
    """Shallow clone (current records). History is fetched separately if needed."""
    if not dest.exists():
        subprocess.run(["git", "clone", "--depth", "1", config.CVELIST_REPO, str(dest)], check=True)
    commit = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    manifest_event("cvelist_clone", path=str(dest), commit=commit)
    return dest


# ===================================================================== NVD change history
def parse_history_events(payload: dict) -> list[dict]:
    """cvehistory/2.0 page -> one row per (event, detail)."""
    out = []
    for item in payload.get("cveChanges", []):
        ch = item.get("change", item)
        for d in ch.get("details", []) or [{}]:
            out.append({
                "cve_id": ch.get("cveId"),
                "event": ch.get("eventName"),
                "created": _ts(ch.get("created")),
                "source": ch.get("sourceIdentifier"),
                "action": d.get("action"),
                "type": d.get("type"),
                "old": d.get("oldValue"),
                "new": d.get("newValue"),
            })
    return out


def original_description(events: pd.DataFrame) -> pd.DataFrame:
    """desc_original = first Description added, earliest event. Missing stays missing."""
    d = events[(events["type"] == "Description") & (events["action"].isin(["Added", "Changed"]))]
    d = d.sort_values(["cve_id", "created"])
    first = d.groupby("cve_id", as_index=False).first()[["cve_id", "new", "created"]]
    return first.rename(columns={"new": "desc_original", "created": "desc_original_at"})


_CVSS_TYPE = re.compile(r"CVSS V3\.[01]", re.I)
_VECTOR = re.compile(r"(?:CVSS:3\.[01]/)?AV:[NALP]/AC:[LH]/PR:[NLH]/UI:[NR]/S:[UC]/C:[NLH]/I:[NLH]/A:[NLH]")


def cvss_at_tpred(events: pd.DataFrame, t_pred: pd.Series) -> pd.DataFrame:
    """Last CVSS v3 vector added at or before t_pred, from any source (NVD or CNA)."""
    e = events[events["type"].fillna("").str.match(_CVSS_TYPE) & events["new"].notna()].copy()
    e["vector"] = e["new"].str.extract(f"({_VECTOR.pattern})", expand=False)
    e["src"] = e["new"].str.split().str[0]
    e = e.dropna(subset=["vector"]).merge(t_pred.rename("t_pred").reset_index(), on="cve_id")
    e = e[e["created"] <= e["t_pred"]].sort_values(["cve_id", "created"])
    last = e.groupby("cve_id", as_index=False).last()
    last["cvss_at_tpred"] = last["vector"].map(cvss.try_base_score)
    last["cvss_source"] = last["src"].map(lambda s: "nvd" if str(s).upper() == "NIST" else "cna")
    return last[["cve_id", "cvss_at_tpred", "cvss_source", "vector"]].rename(
        columns={"vector": "cvss_vector_at_tpred"})


def cwe_at_tpred(events: pd.DataFrame, t_pred: pd.Series) -> pd.DataFrame:
    e = events[(events["type"] == "CWE") & events["new"].notna()].copy()
    e["cwe"] = e["new"].str.extract(r"(CWE-\d+|NVD-CWE-\w+)", expand=False)
    e = e.dropna(subset=["cwe"]).merge(t_pred.rename("t_pred").reset_index(), on="cve_id")
    e = e[e["created"] <= e["t_pred"]].sort_values(["cve_id", "created"])
    return e.groupby("cve_id", as_index=False).last()[["cve_id", "cwe"]].rename(
        columns={"cwe": "cwe_at_tpred"})


def fetch_history(start: str, end: str, dest_dir: Path = config.RAW / "nvd_history") -> list[Path]:
    """All change events in [start, end], in 120-day windows, resumable per window."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    headers = {"apiKey": os.environ[config.NVD_API_KEY_ENV]} if os.environ.get(config.NVD_API_KEY_ENV) else {}
    pause = 0.7 if headers else 6.5                       # NVD public rate limits
    files = []
    s = pd.Timestamp(start)
    while s <= pd.Timestamp(end):
        e = min(s + pd.Timedelta(days=config.NVD_MAX_WINDOW_DAYS - 1), pd.Timestamp(end))
        idx = 0
        while True:
            f = dest_dir / f"{s.date()}_{e.date()}_{idx:06d}.json"
            if f.exists():                                # resume after a Colab disconnect
                page = read_json(f)
            else:
                page = get_json(config.NVD_HISTORY_API, {
                    "changeStartDate": s.strftime("%Y-%m-%dT00:00:00.000"),
                    "changeEndDate": e.strftime("%Y-%m-%dT23:59:59.999"),
                    "resultsPerPage": config.NVD_PAGE_SIZE, "startIndex": idx},
                    headers=headers, pause=pause)
                atomic_write_json(f, page)
            files.append(f)
            idx += page.get("resultsPerPage", 0)
            if idx >= page.get("totalResults", 0) or not page.get("resultsPerPage"):
                break
        s = e + pd.Timedelta(days=1)
    manifest_event("nvd_history", start=start, end=end, pages=len(files))
    return files


def parse_history_files(files: list[Path]) -> pd.DataFrame:
    rows = []
    for f in files:
        rows += parse_history_events(read_json(f))
    return pd.DataFrame(rows)


# ===================================================================== NVD CVE API (current state)
def parse_nvd_cves(payload: dict) -> list[dict]:
    """Current NVD record: used ONLY for desc_current (factor B) and nvd_status_now (forbidden)."""
    out = []
    for v in payload.get("vulnerabilities", []):
        c = v.get("cve", {})
        out.append({
            "cve_id": c.get("id"),
            "nvd_published": _ts(c.get("published")),
            "nvd_status_now": c.get("vulnStatus"),
            "desc_current": pick_english_description(c.get("descriptions", [])),
            "cvss_present_now": bool(c.get("metrics")),
        })
    return out


def fetch_nvd_cves(start: str, end: str, dest_dir: Path = config.RAW / "nvd_cves") -> list[Path]:
    """Current NVD records by publication date, in 120-day windows (resumable)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    headers = {"apiKey": os.environ[config.NVD_API_KEY_ENV]} if os.environ.get(config.NVD_API_KEY_ENV) else {}
    pause = 0.7 if headers else 6.5
    files, s = [], pd.Timestamp(start)
    while s <= pd.Timestamp(end):
        e = min(s + pd.Timedelta(days=config.NVD_MAX_WINDOW_DAYS - 1), pd.Timestamp(end))
        idx = 0
        while True:
            f = dest_dir / f"{s.date()}_{e.date()}_{idx:06d}.json"
            page = read_json(f) if f.exists() else None
            if page is None:
                page = get_json(config.NVD_CVE_API, {
                    "pubStartDate": s.strftime("%Y-%m-%dT00:00:00.000"),
                    "pubEndDate": e.strftime("%Y-%m-%dT23:59:59.999"),
                    "resultsPerPage": 2000, "startIndex": idx}, headers=headers, pause=pause)
                atomic_write_json(f, page)
            files.append(f)
            idx += page.get("resultsPerPage", 0)
            if idx >= page.get("totalResults", 0) or not page.get("resultsPerPage"):
                break
        s = e + pd.Timedelta(days=1)
    manifest_event("nvd_cves", start=start, end=end, pages=len(files))
    return files


def parse_nvd_files(files: list[Path]) -> pd.DataFrame:
    rows = []
    for f in files:
        rows += parse_nvd_cves(read_json(f))
    return pd.DataFrame(rows).drop_duplicates("cve_id", keep="last")


# ===================================================================== CISA KEV
KEV_REQUIRED = {"cveID", "dateAdded"}


def parse_kev(payload: dict) -> pd.DataFrame:
    vulns = payload.get("vulnerabilities")
    if not isinstance(vulns, list) or not vulns:
        raise ValueError("KEV JSON: 'vulnerabilities' missing or empty (schema changed?)")
    missing = KEV_REQUIRED - set(vulns[0])
    if missing:
        raise ValueError(f"KEV JSON schema changed: missing {sorted(missing)}")
    df = pd.DataFrame(vulns)
    out = pd.DataFrame({
        "cve_id": df["cveID"].str.strip(),
        "kev_date_added": pd.to_datetime(df["dateAdded"]).dt.normalize(),
        "kev_vendor_project": df.get("vendorProject"),
        "kev_product": df.get("product"),
        "kev_short_description": df.get("shortDescription"),
        "kev_ransomware": df.get("knownRansomwareCampaignUse"),
        "kev_notes": df.get("notes"),
    })
    out.attrs["catalog_version"] = payload.get("catalogVersion")
    out.attrs["date_released"] = payload.get("dateReleased")
    return out


def fetch_kev(dest: Path | None = None) -> tuple[Path, pd.Timestamp]:
    """Current catalogue. Returns the file and its download date (= label snapshot date)."""
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    dest = dest or config.RAW / "kev" / f"kev_{today.date()}.json"
    download(config.KEV_URL, dest)
    return dest, today


def kev_wayback_snapshots(start: str = "2021-11-01", end: str | None = None) -> pd.DataFrame:
    """List Wayback captures of the KEV feed (coverage must be checked on day 1)."""
    params = {"url": config.KEV_URL.split("://")[1], "from": start.replace("-", "")[:8],
              "output": "json", "filter": "statuscode:200", "collapse": "digest"}
    if end:
        params["to"] = end.replace("-", "")[:8]
    rows = get_json(config.KEV_WAYBACK_CDX, params)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "url"])
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["url"] = "https://web.archive.org/web/" + df["timestamp"] + "id_/" + df["original"]
    df["date"] = pd.to_datetime(df["timestamp"].str[:8])
    return df[["date", "timestamp", "url"]]


def fetch_kev_snapshots(cdx: pd.DataFrame, every_days: int = 30) -> list[Path]:
    """One snapshot per `every_days`; enough to detect removals and changed dateAdded."""
    picked, last = [], None
    for _, r in cdx.sort_values("date").iterrows():
        if last is None or (r["date"] - last).days >= every_days:
            picked.append(r); last = r["date"]
    files = []
    for r in picked:
        f = config.RAW / "kev_snapshots" / f"kev_{r['date'].date()}.json"
        if not f.exists():
            download(r["url"], f)
        files.append(f)
    return files


def kev_union(snapshot_files: list[Path]) -> pd.DataFrame:
    """Earliest dateAdded per CVE across snapshots, with how many snapshots listed it."""
    frames = []
    for f in snapshot_files:
        k = parse_kev(read_json(f))
        k["snapshot"] = pd.Timestamp(re.search(r"(\d{4}-\d{2}-\d{2})", f.name).group(1))
        frames.append(k)
    allk = pd.concat(frames, ignore_index=True)
    return allk.groupby("cve_id").agg(kev_date_added=("kev_date_added", "min"),
                                     n_snapshots=("snapshot", "nunique"),
                                     last_seen=("snapshot", "max")).reset_index()


# ===================================================================== FIRST EPSS
_EPSS_HEADER = re.compile(r"model_version:([^,]+),score_date:([0-9T:\-+.Z]+)")


def parse_epss_csv(raw: bytes | str) -> pd.DataFrame:
    if isinstance(raw, bytes):
        raw = gzip.decompress(raw).decode() if raw[:2] == b"\x1f\x8b" else raw.decode()
    first, _, rest = raw.partition("\n")
    m = _EPSS_HEADER.search(first)
    body = rest if first.startswith("#") else raw
    df = pd.read_csv(io.StringIO(body))
    df = df.rename(columns={"cve": "cve_id"})
    df["epss_model_version"] = _epss_version_label(m.group(1)) if m else None
    df["score_date"] = pd.Timestamp(m.group(2)[:10]) if m else pd.NaT
    return df[["cve_id", "epss", "percentile", "epss_model_version", "score_date"]]


def _epss_version_label(model_version: str) -> str:
    """'v2023.03.01' -> 'v3'. Boundaries from FIRST/Empirical release dates."""
    d = pd.Timestamp(model_version.lstrip("v").replace(".", "-"))
    if d >= pd.Timestamp("2025-03-01"):
        return "v4"
    if d >= pd.Timestamp("2023-03-01"):
        return "v3"
    if d >= pd.Timestamp("2022-02-01"):
        return "v2"
    return "v1"


def fetch_epss_for_day(day: pd.Timestamp) -> Path | None:
    day = pd.Timestamp(day).normalize()
    dest = config.RAW / "epss" / f"epss_{day.date()}.csv.gz"
    if dest.exists():
        return dest
    for tmpl in config.EPSS_DAILY_URLS:
        try:
            return download(tmpl.format(date=day.date()), dest, retries=3)
        except (FileNotFoundError, RuntimeError):
            continue
    manifest_event("epss_missing_day", day=str(day.date()))
    return None


def epss_at_tpred(cves: pd.DataFrame, max_lag: int = config.EPSS_MAX_LAG_DAYS) -> pd.DataFrame:
    """First daily file dated >= t_pred that contains the CVE; records lag and model version."""
    need = cves[["cve_id", "t_pred"]].copy()
    found = []
    for lag in range(0, max_lag + 30):                     # search past the cut-off to count lags
        if need.empty:
            break
        for day, grp in need.groupby(need["t_pred"] + pd.Timedelta(days=lag)):
            f = fetch_epss_for_day(day)
            if f is None:
                continue
            scores = parse_epss_csv(Path(f).read_bytes())
            hit = grp.merge(scores, on="cve_id")
            if len(hit):
                hit["epss_lag_days"] = lag
                found.append(hit)
        got = set(pd.concat(found)["cve_id"]) if found else set()
        need = need[~need["cve_id"].isin(got)]
    res = pd.concat(found, ignore_index=True) if found else pd.DataFrame(
        columns=["cve_id", "epss", "percentile", "epss_model_version", "epss_lag_days"])
    res = res.rename(columns={"epss": "epss_at_pred", "percentile": "epss_pct_at_pred"})
    return res[["cve_id", "epss_at_pred", "epss_pct_at_pred", "epss_model_version", "epss_lag_days"]]


# ===================================================================== merge
def build_dataset(cvelist: pd.DataFrame, history_events: pd.DataFrame, nvd_now: pd.DataFrame,
                  kev: pd.DataFrame, epss: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per CVE. t0 = cvelistV5 datePublished."""
    df = cvelist.copy()
    df["published_date"] = df["published_date"].dt.tz_localize(None) if \
        getattr(df["published_date"].dt, "tz", None) is not None else df["published_date"]
    df["published_date"] = df["published_date"].dt.normalize()
    df["t_pred"] = df["published_date"] + pd.Timedelta(days=config.T_PRED_OFFSET_DAYS)
    tp = df.set_index("cve_id")["t_pred"]
    df = df.merge(original_description(history_events), on="cve_id", how="left")
    df = df.merge(cvss_at_tpred(history_events, tp), on="cve_id", how="left")
    df = df.merge(cwe_at_tpred(history_events, tp), on="cve_id", how="left")
    df["cvss_missing_at_tpred"] = df["cvss_at_tpred"].isna()
    df["cvss_source"] = df["cvss_source"].fillna("missing")
    df["cwe"] = df["cwe_at_tpred"]
    df = df.merge(nvd_now, on="cve_id", how="left")
    df = df.merge(kev, on="cve_id", how="left")
    if epss is not None:
        df = df.merge(epss, on="cve_id", how="left")
    df["is_synthetic"] = False
    if df["cve_id"].duplicated().any():
        raise ValueError("duplicate cve_id after merge")
    return df


def _ts(x):
    if x is None:
        return pd.NaT
    t = pd.to_datetime(x, utc=True, errors="coerce")
    return t.tz_localize(None) if t is not pd.NaT and t is not None else pd.NaT
