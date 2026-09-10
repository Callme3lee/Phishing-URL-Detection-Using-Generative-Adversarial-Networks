from __future__ import annotations

import csv
import gzip
import hashlib
import random
import sqlite3
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .common import load_json, sha256_file, write_json
from .url_data import PSL_EXTRACTOR, canonicalize_url, encode_url, encoded_digest


ROOT = Path("data/evaluation/two_stage_url_cascade_v1")
FREEZE = Path("artifacts/experiments/two_stage_url_cascade_v1/calibration/frozen_cascade.json")
PHISHTANK = Path("data/deployment/2026-08-15-precision/raw/phishtank_online_valid.csv.gz")
TRANCO = Path("data/deployment/2026-08-15-confirmation/raw/tranco_top1m.csv.zip")
FIELDS = ("url", "label", "source", "canonical_hostname", "registered_domain", "source_id", "source_timestamp")


def registered(host: str) -> str:
    item = PSL_EXTRACTOR(host.lower())
    return ".".join(v for v in (item.domain, item.suffix) if v) or host.lower()


def stable(seed: int, value: str) -> bytes:
    return hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=16).digest()


def main() -> None:
    if ROOT.exists(): raise FileExistsError(f"Evaluation cohort already frozen at {ROOT}")
    freeze = load_json(FREEZE); seed = 2026081507
    if freeze["selection_status"] != "FROZEN BEFORE FINAL EVALUATION COHORT CONSTRUCTION":
        raise RuntimeError("Cascade rules are not frozen")
    vocabulary = load_json(Path("data/processed/vocabulary.json")); maximum = 128
    urls:set[str]=set(); hosts:set[str]=set(); domains:set[str]=set(); digests:set[bytes]=set()
    def forbid(raw: str, known_host: str="") -> None:
        try:url,host,_=canonicalize_url(raw)
        except ValueError:return
        host=(known_host or host).lower();urls.add(url);hosts.add(host);domains.add(registered(host));digests.add(encoded_digest(encode_url(url,vocabulary,maximum)))
    database=sqlite3.connect("data/processed/master.sqlite")
    for url,host in database.execute("SELECT normalized_url,hostname FROM urls"):forbid(url,host)
    database.close()
    paths=[]
    for pattern in ("data/experiments/recent_update_gan10_v1/*/processed/*manifest.csv",
                    "data/calibration/*/calibration_urls.csv","data/deployment/*/*.csv",
                    "data/external/*/external_urls.csv","data/updates/*/*.csv"):
        paths.extend(Path().glob(pattern))
    for path in sorted(set(paths)):
        with path.open("r",encoding="utf-8",errors="replace",newline="") as handle:
            for row in csv.DictReader(handle):forbid(row.get("raw_url") or row.get("url") or "",row.get("canonical_hostname", ""))

    rejects=Counter();phishing=[];seen=set()
    with gzip.open(PHISHTANK,"rt",encoding="utf-8",errors="replace",newline="") as handle:
        candidates=list(csv.DictReader(handle))
    candidates.sort(key=lambda row:stable(seed,row.get("url", "")))
    for row in candidates:
        if len(phishing)>=2000:break
        if row.get("verified","").lower()!="yes" or row.get("online","").lower()!="yes":continue
        try:url,host,_=canonicalize_url(row.get("url", ""))
        except ValueError:rejects["invalid_phishing"]+=1;continue
        domain=registered(host);digest=encoded_digest(encode_url(url,vocabulary,maximum))
        if url in urls or host in hosts or domain in domains or digest in digests:rejects["prior_phishing_identity"]+=1;continue
        if domain in seen:rejects["duplicate_phishing_domain"]+=1;continue
        seen.add(domain);phishing.append({"url":url,"label":1,"source":"PhishTank verified-online holdout",
            "canonical_hostname":host,"registered_domain":domain,"source_id":row.get("phish_id", ""),
            "source_timestamp":row.get("verification_time", "")})
    if len(phishing)<2000:raise RuntimeError(f"Only {len(phishing)} identity-disjoint phishing domains")

    legitimate=[];seen_legit=set()
    with zipfile.ZipFile(TRANCO) as archive:
        member=archive.namelist()[0]
        rows=csv.reader((line.decode("utf-8",errors="replace") for line in archive.open(member)))
        for rank,domain_value in rows:
            if len(legitimate)>=18000:break
            raw=f"https://{domain_value.strip().lower()}/"
            try:url,host,_=canonicalize_url(raw)
            except ValueError:rejects["invalid_legitimate"]+=1;continue
            domain=registered(host);digest=encoded_digest(encode_url(url,vocabulary,maximum))
            if url in urls or host in hosts or domain in domains or digest in digests or domain in seen:rejects["prior_or_phishing_legitimate_identity"]+=1;continue
            if domain in seen_legit:continue
            seen_legit.add(domain);legitimate.append({"url":url,"label":0,"source":"Tranco root-domain holdout",
                "canonical_hostname":host,"registered_domain":domain,"source_id":rank,"source_timestamp":"2026-08-15"})
    if len(legitimate)<18000:raise RuntimeError(f"Only {len(legitimate)} identity-disjoint legitimate domains")
    rows=legitimate+phishing;random.Random(seed).shuffle(rows);ROOT.mkdir(parents=True)
    output=ROOT/"evaluation_urls.csv"
    with output.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    report={"project_version":3,"dataset_name":"two_stage_url_cascade_identity_disjoint_v1",
        "locked_at_utc":datetime.now(timezone.utc).isoformat(),"selection_locked_before_collection":True,
        "rows":len(rows),"legitimate_rows":len(legitimate),"phishing_rows":len(phishing),"phishing_prevalence":len(phishing)/len(rows),
        "unique_registered_domains":len({row["registered_domain"] for row in rows}),"dataset":str(output),"dataset_sha256":sha256_file(output),
        "cascade_freeze":str(FREEZE),"cascade_freeze_sha256":sha256_file(FREEZE),
        "raw_source_hashes":{"phishtank":sha256_file(PHISHTANK),"tranco":sha256_file(TRANCO)},
        "identity_exclusions":{"urls":len(urls),"hosts":len(hosts),"registered_domains":len(domains),"encoded_inputs":len(digests)},
        "rejections":dict(rejects),"one_shot_contract":"No threshold, model, feature, or cohort selection change after inference.",
        "limitations":["This is identity-disjoint, not a later temporal snapshot; source files were already locally available.",
            "PhishTank verified/online membership is the phishing label; URLs were not visited.",
            "Tranco membership is a legitimate-domain proxy, and legitimate rows are root URLs.",
            "This URL-only test does not provide HTML, DNS, certificate, reputation, or redirect evidence."]}
    write_json(ROOT/"collection_report.json",report);print(report)


if __name__=="__main__":main()
