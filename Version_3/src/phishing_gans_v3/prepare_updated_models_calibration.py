from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .common import load_json, sha256_file, write_json
from .url_data import PSL_EXTRACTOR, canonicalize_url, encode_url, encoded_digest


ROOT = Path("data/calibration/updated_models_2026-08-15_v1")
FIELDS = ("url", "label", "source", "structure_category", "canonical_hostname", "registered_domain", "source_id", "source_timestamp")


def registered(host: str) -> str:
    item = PSL_EXTRACTOR(host.lower())
    return ".".join(value for value in (item.domain, item.suffix) if value) or host.lower()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def stable(seed: int, value: str) -> bytes:
    return hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=16).digest()


def structure(raw: str) -> str:
    from urllib.parse import urlsplit
    parsed = urlsplit(raw if "://" in raw else "http://" + raw.lstrip("/"))
    if parsed.query: return "query"
    depth = len([value for value in parsed.path.split("/") if value])
    return "root" if depth == 0 else ("shallow" if depth == 1 else "deep")


def main() -> None:
    if ROOT.exists(): raise FileExistsError(f"Calibration already frozen at {ROOT}")
    ROOT.mkdir(parents=True)
    seed=2026081505; vocabulary=load_json(Path("data/processed/vocabulary.json")); maximum=128
    forbidden_urls:set[str]=set();forbidden_hosts:set[str]=set();forbidden_domains:set[str]=set();forbidden_hashes:set[bytes]=set()
    identity_paths=[]
    patterns=("data/experiments/recent_update_gan10_v1/*/processed/*manifest.csv",
              "data/calibration/*/calibration_urls.csv","data/deployment/*/*.csv","data/external/*/external_urls.csv",
              "data/updates/*/*.csv")
    for pattern in patterns: identity_paths.extend(Path().glob(pattern))
    def forbid(raw:str,provided_host:str="") -> None:
        try:url,host,_=canonicalize_url(raw)
        except ValueError:return
        host=(provided_host or host).lower();forbidden_urls.add(url);forbidden_hosts.add(host);forbidden_domains.add(registered(host))
        forbidden_hashes.add(encoded_digest(encode_url(url,vocabulary,maximum)))
    for path in sorted(set(identity_paths)):
        for row in read_csv(path):forbid(row.get("raw_url") or row.get("url") or "",row.get("canonical_hostname", ""))

    phishtank=Path("data/deployment/2026-08-15-precision/raw/phishtank_online_valid.csv.gz")
    tranco=Path("data/deployment/2026-08-15-confirmation/raw/tranco_top1m.csv.zip")
    with gzip.open(phishtank,"rt",encoding="utf-8",errors="replace",newline="") as handle: source_phish=list(csv.DictReader(handle))
    active_phish_domains=set();phish_candidates=[];rejects=Counter();seen=set()
    for row in source_phish:
        try:url,host,_=canonicalize_url(row.get("url", ""))
        except ValueError:rejects["phishing_invalid"]+=1;continue
        domain=registered(host);active_phish_domains.add(domain)
        if row.get("verified","").lower()!="yes" or row.get("online","").lower()!="yes":continue
        digest=encoded_digest(encode_url(url,vocabulary,maximum))
        if url in forbidden_urls or host in forbidden_hosts or domain in forbidden_domains or digest in forbidden_hashes:
            rejects["phishing_prior_identity"]+=1;continue
        if domain in seen:rejects["phishing_duplicate_domain"]+=1;continue
        seen.add(domain);phish_candidates.append({"url":url,"label":1,"source":"PhishTank verified online",
            "structure_category":structure(url),"canonical_hostname":host,"registered_domain":domain,
            "source_id":row.get("phish_id", ""),"source_timestamp":row.get("verification_time", "")})
    phish_candidates.sort(key=lambda row:stable(seed,str(row["url"])))
    phishing=phish_candidates[:2500]
    if len(phishing)<2500:raise RuntimeError(f"Only {len(phishing)} phishing calibration domains")

    path_candidates=[]
    for cache in Path("data/development").glob("*/cache/*.jsonl"):
        for line in cache.read_text(encoding="utf-8",errors="replace").splitlines():
            try:
                record=json.loads(line);raw=record.get("url", "");url,host,_=canonicalize_url(raw);domain=registered(host)
            except (json.JSONDecodeError,ValueError):continue
            category=structure(raw)
            if category=="root":continue
            digest=encoded_digest(encode_url(url,vocabulary,maximum))
            if (url in forbidden_urls or host in forbidden_hosts or domain in forbidden_domains or digest in forbidden_hashes
                    or domain in active_phish_domains):continue
            path_candidates.append((stable(seed+1,url),url,host,domain,category,record))
    path_candidates.sort(key=lambda row:row[0]);selected_urls=set();selected_domains=Counter();category_counts=Counter();paths=[]
    caps={"shallow":150,"deep":300,"query":150}
    for _,url,host,domain,category,record in path_candidates:
        if len(paths)>=sum(caps.values()):break
        if category_counts[category]>=caps[category] or selected_domains[domain]>=25 or url in selected_urls:continue
        selected_urls.add(url);selected_domains[domain]+=1;category_counts[category]+=1
        paths.append({"url":url,"label":0,"source":"Common Crawl cached URL Index","structure_category":category,
            "canonical_hostname":host,"registered_domain":domain,"source_id":record.get("digest", ""),"source_timestamp":record.get("timestamp", "")})
    if len(paths)<500:raise RuntimeError(f"Only {len(paths)} strict path calibration URLs")

    legitimate=list(paths);legit_domains={str(row["registered_domain"]) for row in paths};legit_urls={str(row["url"]) for row in paths}
    with zipfile.ZipFile(tranco) as archive:
        member=archive.namelist()[0]
        rows=csv.reader((line.decode("utf-8",errors="replace") for line in archive.open(member)))
        for rank,domain_value in rows:
            if len(legitimate)>=20600:break
            raw=f"https://{domain_value.strip().lower()}/"
            try:url,host,_=canonicalize_url(raw)
            except ValueError:continue
            domain=registered(host);digest=encoded_digest(encode_url(url,vocabulary,maximum))
            if (url in forbidden_urls or host in forbidden_hosts or domain in forbidden_domains or digest in forbidden_hashes
                    or domain in active_phish_domains or domain in legit_domains or url in legit_urls):continue
            legit_domains.add(domain);legit_urls.add(url);legitimate.append({"url":url,"label":0,"source":"Tranco root proxy",
                "structure_category":"root","canonical_hostname":host,"registered_domain":domain,
                "source_id":rank,"source_timestamp":"2026-08-15"})
    if len(legitimate)<20600:raise RuntimeError(f"Only {len(legitimate)} legitimate calibration URLs")
    all_rows=legitimate+phishing;random.Random(seed).shuffle(all_rows)
    dataset=ROOT/"calibration_urls.csv"
    with dataset.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS);writer.writeheader();writer.writerows(all_rows)
    checkpoint_roots={"real_only":Path("artifacts/experiments/recent_update_gan10_v1/real_only/hyb_ms"),
                      "gan_10pct":Path("artifacts/experiments/recent_update_gan10_v1/gan_10pct/hyb_ms")}
    checkpoints={condition:[{"seed":s,"path":str(root/f"seed_{s}/checkpoint.pt"),
                             "sha256":sha256_file(root/f"seed_{s}/checkpoint.pt")} for s in (42,43,44)]
                 for condition,root in checkpoint_roots.items()}
    report={"project_version":3,"dataset_name":"updated_models_calibration_2026-08-15_v1",
        "frozen_at_utc":datetime.now(timezone.utc).isoformat(),"purpose":"new threshold calibration for updated real-only and +10% GAN ensembles",
        "rows":len(all_rows),"legitimate_rows":len(legitimate),"phishing_rows":len(phishing),
        "structure_counts":dict(Counter(str(row["structure_category"]) for row in all_rows)),
        "legitimate_structure_counts":dict(Counter(str(row["structure_category"]) for row in legitimate)),
        "unique_urls":len({str(row["url"]) for row in all_rows}),"unique_registered_domains":len({str(row["registered_domain"]) for row in all_rows}),
        "identity_exclusion":{"urls":len(forbidden_urls),"hosts":len(forbidden_hosts),"registered_domains":len(forbidden_domains),"encoded_inputs":len(forbidden_hashes)},
        "dataset":str(dataset),"dataset_sha256":sha256_file(dataset),"checkpoints":checkpoints,
        "selection_rule":"three-seed mean FP32 probabilities; maximize F1 subject to FPR <= 0.25% and recall >= 80%; if infeasible, report failure and best recall under FPR cap",
        "probability_contract":"model forward pass may use FP16 autocast; sigmoid is computed in FP32",
        "protected_confirmation_rows_used":0,"prior_evaluation_rows_used":0,"training_rows_used":0,
        "source_hashes":{"phishtank":sha256_file(phishtank),"tranco":sha256_file(tranco)},"rejections":dict(rejects),
        "limitations":["Tranco membership is a legitimate-domain proxy, not page-level manual adjudication.",
                       f"Common Crawl connectivity failed during a new collection attempt; strict unseen cached records supply {len(paths)} path URLs across {len(selected_domains)} registered domains.",
                       "The phishing snapshot is identity-disjoint but was downloaded on the same date as training development, so this is calibration rather than temporal confirmation."]}
    write_json(ROOT/"collection_report.json",report);print(json.dumps(report,indent=2))


if __name__=="__main__":main()
