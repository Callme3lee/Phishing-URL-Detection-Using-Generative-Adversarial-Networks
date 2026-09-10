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


ROOT=Path("data/development/dns_secondary_v1")
PHISHTANK=Path("data/deployment/2026-08-15-precision/raw/phishtank_online_valid.csv.gz")
TRANCO=Path("data/deployment/2026-08-15-confirmation/raw/tranco_top1m.csv.zip")
POLICY=Path("artifacts/deployment_candidates/conservative_hyb_gan10_v1/policy.json")
FIELDS=("url","label","source","canonical_hostname","registered_domain","source_id","source_timestamp","split")


def registered(host: str) -> str:
    item=PSL_EXTRACTOR(host.lower());return ".".join(v for v in (item.domain,item.suffix) if v) or host.lower()


def stable(seed: int,value: str)->bytes:return hashlib.blake2b(f"{seed}:{value}".encode(),digest_size=16).digest()


def split(domain: str)->str:
    value=int.from_bytes(stable(2026081511,domain),"big")/2**128
    return "train" if value<.60 else ("calibration" if value<.80 else "holdout")


def main():
    if ROOT.exists():raise FileExistsError(f"Development cohort already exists: {ROOT}")
    vocabulary=load_json(Path("data/processed/vocabulary.json"));urls=set();hosts=set();domains=set();digests=set();maximum=128
    def forbid(raw,known_host=""):
        try:url,host,_=canonicalize_url(raw)
        except ValueError:return
        host=(known_host or host).lower();urls.add(url);hosts.add(host);domains.add(registered(host));digests.add(encoded_digest(encode_url(url,vocabulary,maximum)))
    database=sqlite3.connect("data/processed/master.sqlite")
    for url,host in database.execute("SELECT normalized_url,hostname FROM urls"):forbid(url,host)
    database.close();paths=[]
    for pattern in ("data/experiments/recent_update_gan10_v1/*/processed/*manifest.csv","data/calibration/*/calibration_urls.csv",
                    "data/deployment/*/*.csv","data/external/*/external_urls.csv","data/updates/*/*.csv","data/evaluation/*/*.csv"):
        paths.extend(Path().glob(pattern))
    for path in sorted(set(paths)):
        with path.open("r",encoding="utf-8",errors="replace",newline="") as handle:
            for row in csv.DictReader(handle):forbid(row.get("raw_url") or row.get("url") or "",row.get("canonical_hostname", ""))
    seed=2026081510;rejects=Counter();phishing=[];seen=set()
    with gzip.open(PHISHTANK,"rt",encoding="utf-8",errors="replace",newline="") as handle:candidates=list(csv.DictReader(handle))
    candidates.sort(key=lambda r:stable(seed,r.get("url", "")))
    for row in candidates:
        if len(phishing)>=6000:break
        if row.get("verified","").lower()!="yes" or row.get("online","").lower()!="yes":continue
        try:url,host,_=canonicalize_url(row.get("url", ""))
        except ValueError:rejects["invalid_phishing"]+=1;continue
        domain=registered(host);digest=encoded_digest(encode_url(url,vocabulary,maximum))
        if url in urls or host in hosts or domain in domains or digest in digests:rejects["prior_identity_phishing"]+=1;continue
        if domain in seen:rejects["duplicate_phishing_domain"]+=1;continue
        seen.add(domain);phishing.append({"url":url,"label":1,"source":"PhishTank development-only verified-online",
            "canonical_hostname":host,"registered_domain":domain,"source_id":row.get("phish_id", ""),"source_timestamp":row.get("verification_time", ""),"split":split(domain)})
    if len(phishing)<3000:raise RuntimeError(f"Insufficient identity-disjoint phishing domains: {len(phishing)}")
    legitimate=[];seen_legit=set();target=max(12000,2*len(phishing))
    with zipfile.ZipFile(TRANCO) as archive:
        member=archive.namelist()[0];rows=csv.reader((line.decode("utf-8",errors="replace") for line in archive.open(member)))
        for rank,value in rows:
            if len(legitimate)>=target:break
            try:url,host,_=canonicalize_url(f"https://{value.strip().lower()}/")
            except ValueError:rejects["invalid_legitimate"]+=1;continue
            domain=registered(host);digest=encoded_digest(encode_url(url,vocabulary,maximum))
            if url in urls or host in hosts or domain in domains or digest in digests or domain in seen:rejects["prior_or_phishing_legitimate"]+=1;continue
            if domain in seen_legit:continue
            seen_legit.add(domain);legitimate.append({"url":url,"label":0,"source":"Tranco development-only root proxy",
                "canonical_hostname":host,"registered_domain":domain,"source_id":rank,"source_timestamp":"2026-08-15","split":split(domain)})
    if len(legitimate)<target:raise RuntimeError(f"Insufficient legitimate domains: {len(legitimate)}/{target}")
    rows=legitimate+phishing;random.Random(seed).shuffle(rows);ROOT.mkdir(parents=True);dataset=ROOT/"development_urls.csv"
    with dataset.open("w",encoding="utf-8",newline="") as handle:writer=csv.DictWriter(handle,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    report={"project_version":3,"dataset_name":"dns_secondary_development_v1","created_at_utc":datetime.now(timezone.utc).isoformat(),
        "purpose":"development-only DNS secondary verifier; forbidden for final confirmation claims","rows":len(rows),"legitimate_rows":len(legitimate),"phishing_rows":len(phishing),
        "split_counts":dict(Counter(row["split"] for row in rows)),"split_label_counts":{s:{str(y):sum(row["split"]==s and row["label"]==y for row in rows) for y in (0,1)} for s in ("train","calibration","holdout")},
        "unique_registered_domains":len(rows),"dataset":str(dataset),"dataset_sha256":sha256_file(dataset),"policy":str(POLICY),"policy_sha256":sha256_file(POLICY),
        "identity_exclusions":{"urls":len(urls),"hosts":len(hosts),"registered_domains":len(domains),"encoded_inputs":len(digests)},"rejections":dict(rejects),
        "source_hashes":{"phishtank":sha256_file(PHISHTANK),"tranco":sha256_file(TRANCO)},
        "limitations":["Development sources are same-date local snapshots, not future temporal evidence.","Tranco is a legitimate proxy and PhishTank membership supplies phishing labels.","Every row has a distinct registered domain and is identity-disjoint from protected training, calibration, external and 20K evaluation sets."]}
    write_json(ROOT/"collection_report.json",report);print(report)


if __name__=="__main__":main()
