from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .common import sha256_file,write_json


def parse_args():
    p=argparse.ArgumentParser(description="Reuse cached GAN-routed DNS evidence and isolate missing real-only routed rows.")
    p.add_argument("--scored",type=Path,required=True);p.add_argument("--existing",type=Path,required=True)
    p.add_argument("--missing-input",type=Path,required=True);p.add_argument("--missing-evidence",type=Path)
    p.add_argument("--merged",type=Path,required=True);return p.parse_args()


def read(path):return list(csv.DictReader(path.open("r",encoding="utf-8",newline="")))


def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as handle:writer=csv.DictWriter(handle,fieldnames=list(rows[0].keys()));writer.writeheader();writer.writerows(rows)


def main():
    args=parse_args();scored=read(args.scored);required=[r for r in scored if r["risk_tier"]=="secondary_review"];existing=read(args.existing);cache={r["url"]:r for r in existing}
    missing=[r for r in required if r["url"] not in cache]
    if args.missing_input.exists():
        prior=read(args.missing_input)
        if [r["url"] for r in prior]!=[r["url"] for r in missing]:raise RuntimeError("Frozen missing-input identity/order mismatch")
    else:write(args.missing_input,missing)
    if args.missing_evidence is None or not args.missing_evidence.exists():
        print(json.dumps({"required":len(required),"reused":len(required)-len(missing),"missing":len(missing),"missing_input":str(args.missing_input)},indent=2));return
    for row in read(args.missing_evidence):
        if row["url"] in cache:raise RuntimeError("Missing-evidence file overlaps existing cache")
        cache[row["url"]]=row
    merged=[cache[r["url"]] for r in required]
    if len(merged)!=len(required):raise RuntimeError("Incomplete merged DNS cache")
    write(args.merged,merged)
    payload={"required_rows":len(required),"reused_rows":len(required)-len(missing),"newly_queried_rows":len(missing),"scored":str(args.scored),"scored_sha256":sha256_file(args.scored),
        "existing_cache":str(args.existing),"existing_cache_sha256":sha256_file(args.existing),"missing_input":str(args.missing_input),"missing_input_sha256":sha256_file(args.missing_input),
        "missing_evidence":str(args.missing_evidence),"missing_evidence_sha256":sha256_file(args.missing_evidence),"merged":str(args.merged),"merged_sha256":sha256_file(args.merged),
        "cache_contract":"Overlapping domains reuse byte-identical prior DNS observations; only real-only-exclusive routed identities are newly queried."}
    write_json(args.merged.with_suffix(".report.json"),payload);print(json.dumps(payload,indent=2))


if __name__=="__main__":main()
