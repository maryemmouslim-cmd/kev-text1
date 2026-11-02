"""Parsers on fixtures shaped like each documented API (no network)."""
import gzip
import json

import pandas as pd
import pytest

from src import data

CVE_REC = {
    "cveMetadata": {"cveId": "CVE-2023-0001", "state": "PUBLISHED",
                    "datePublished": "2023-03-01T10:00:00.000Z", "assignerShortName": "acme"},
    "containers": {
        "cna": {"descriptions": [{"lang": "es", "value": "Desbordamiento"},
                                 {"lang": "en-US", "value": "Overflow in Acme Router allows RCE."}],
                "affected": [{"vendor": "Acme", "product": "Router"}, {"vendor": "n/a", "product": "n/a"}],
                "metrics": [{"cvssV3_1": {"vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}}],
                "problemTypes": [{"descriptions": [{"cweId": "CWE-787"}]}]},
        "adp": [{"title": "CISA ADP Vulnrichment", "metrics": [{"other": {"content": {"options": [{"Exploitation": "active"}]}}}]}],
    },
}
HISTORY = {"cveChanges": [
    {"change": {"cveId": "CVE-2023-0001", "eventName": "New CVE Received", "created": "2023-03-01T10:15:00.000",
                "sourceIdentifier": "cve@mitre.org",
                "details": [{"action": "Added", "type": "Description", "newValue": "Overflow in Acme Router."},
                            {"action": "Added", "type": "CVSS V3.1",
                             "newValue": "acme AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H"}]}},
    {"change": {"cveId": "CVE-2023-0001", "eventName": "Initial Analysis", "created": "2023-04-20T12:00:00.000",
                "sourceIdentifier": "nvd@nist.gov",
                "details": [{"action": "Added", "type": "CVSS V3.1",
                             "newValue": "NIST AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"},
                            {"action": "Added", "type": "CWE", "newValue": "NIST CWE-787"}]}},
    {"change": {"cveId": "CVE-2023-0001", "eventName": "CVE Modified", "created": "2023-06-01T09:00:00.000",
                "sourceIdentifier": "acme",
                "details": [{"action": "Changed", "type": "Description",
                             "oldValue": "Overflow in Acme Router.",
                             "newValue": "Overflow in Acme Router. Actively exploited in the wild."}]}},
]}
KEV = {"catalogVersion": "2026.10.16", "dateReleased": "2026-10-16T12:00:00.000Z", "count": 1,
       "vulnerabilities": [{"cveID": "CVE-2023-0001", "vendorProject": "Acme", "product": "Router",
                            "dateAdded": "2023-05-02", "shortDescription": "x",
                            "knownRansomwareCampaignUse": "Unknown", "notes": ""}]}
EPSS = ("#model_version:v2023.03.01,score_date:2023-03-02T00:00:00+0000\n"
        "cve,epss,percentile\nCVE-2023-0001,0.0123,0.81\nCVE-2023-0002,0.0004,0.05\n")


def test_cve_record():
    r = data.parse_cve_record(CVE_REC)
    assert r["desc_cvelist_current"] == "Overflow in Acme Router allows RCE."
    assert r["vendors"] == ["Acme"] and r["products"] == ["Router"] and r["cna_cwe_current"] == "CWE-787"
    assert r["has_adp"] and r["published_date"] == pd.Timestamp("2023-03-01 10:00:00")


def test_rejected_and_no_english():
    rec = json.loads(json.dumps(CVE_REC))
    rec["cveMetadata"]["state"] = "REJECTED"
    rec["containers"]["cna"]["descriptions"] = [{"lang": "fr", "value": "Rejeté"}]
    r = data.parse_cve_record(rec)
    assert r["state"] == "REJECTED" and r["desc_cvelist_current"] is None and not r["lang_ok"]


def test_history_original_description_and_cvss_at_tpred():
    ev = pd.DataFrame(data.parse_history_events(HISTORY))
    orig = data.original_description(ev)
    assert orig["desc_original"].iloc[0] == "Overflow in Acme Router."
    tp = pd.Series([pd.Timestamp("2023-03-02")], index=pd.Index(["CVE-2023-0001"], name="cve_id"))
    c = data.cvss_at_tpred(ev, tp)
    assert c["cvss_at_tpred"].iloc[0] == 8.8 and c["cvss_source"].iloc[0] == "cna"
    later = data.cvss_at_tpred(ev, tp + pd.Timedelta(days=60))
    assert later["cvss_at_tpred"].iloc[0] == 9.8 and later["cvss_source"].iloc[0] == "nvd"
    assert data.cwe_at_tpred(ev, tp).empty                   # NVD CWE arrived after t_pred


def test_kev_parse_and_schema_change():
    k = data.parse_kev(KEV)
    assert k["kev_date_added"].iloc[0] == pd.Timestamp("2023-05-02")
    with pytest.raises(ValueError, match="schema"):
        data.parse_kev({"vulnerabilities": [{"cve": "CVE-1", "added": "2023"}]})
    with pytest.raises(ValueError):
        data.parse_kev({"items": []})


def test_epss_parse_gzip_and_version():
    df = data.parse_epss_csv(gzip.compress(EPSS.encode()))
    assert df["epss_model_version"].iloc[0] == "v3" and df["score_date"].iloc[0] == pd.Timestamp("2023-03-02")
    assert data._epss_version_label("v2025.03.14") == "v4"


def test_build_dataset():
    cl = pd.DataFrame([data.parse_cve_record(CVE_REC)])
    ev = pd.DataFrame(data.parse_history_events(HISTORY))
    nvd = pd.DataFrame([{"cve_id": "CVE-2023-0001", "nvd_published": pd.Timestamp("2023-03-01"),
                         "nvd_status_now": "Analyzed", "desc_current": "Overflow. Actively exploited.",
                         "cvss_present_now": True}])
    df = data.build_dataset(cl, ev, nvd, data.parse_kev(KEV))
    r = df.iloc[0]
    assert r["published_date"] == pd.Timestamp("2023-03-01") and r["t_pred"] == pd.Timestamp("2023-03-02")
    assert r["desc_original"] == "Overflow in Acme Router." and r["cvss_at_tpred"] == 8.8
    assert not r["is_synthetic"]
