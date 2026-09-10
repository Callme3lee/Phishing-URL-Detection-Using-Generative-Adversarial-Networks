from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .common import sha256_file, write_json


QUERIES=(("host_a","hostname","A"),("host_aaaa","hostname","AAAA"),("host_cname","hostname","CNAME"),
         ("domain_mx","domain","MX"),("domain_ns","domain","NS"))
STATUS=re.compile(r"status:\s*([A-Z]+)")
QUERY_TIME=re.compile(r"Query time:\s*(\d+)\s*msec")


def parse_args():
    p=argparse.ArgumentParser(description="Collect live DNS-only evidence for secondary-review URLs.")
    p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--tier",default="secondary_review");p.add_argument("--workers",type=int,default=40);p.add_argument("--timeout",type=float,default=4.0)
    return p.parse_args()


def query(name: str,record_type: str,timeout:float)->dict:
    started=time.perf_counter()
    command=["dig","+time=1","+tries=1","+noall","+answer","+comments","+stats",name,record_type]
    try:
        result=subprocess.run(command,capture_output=True,text=True,timeout=timeout,check=False)
        text=result.stdout;status_match=STATUS.search(text);status=status_match.group(1) if status_match else ("PROCESS_ERROR" if result.returncode else "UNKNOWN")
        answers=[];ttls=[]
        for line in text.splitlines():
            if not line or line.startswith(";"):continue
            parts=line.split()
            if len(parts)>=5 and parts[3].upper()==record_type:answers.append(" ".join(parts[4:]));
            if len(parts)>=5:
                try:ttls.append(int(parts[1]))
                except ValueError:pass
        timing=QUERY_TIME.search(text);milliseconds=float(timing.group(1)) if timing else 1000*(time.perf_counter()-started)
        return {"status":status,"count":len(answers),"minimum_ttl":min(ttls) if ttls else 0,"maximum_ttl":max(ttls) if ttls else 0,
                "query_ms":milliseconds,"timeout":0,"process_error":int(result.returncode!=0),"answers":answers}
    except subprocess.TimeoutExpired:
        return {"status":"TIMEOUT","count":0,"minimum_ttl":0,"maximum_ttl":0,"query_ms":1000*timeout,"timeout":1,"process_error":0,"answers":[]}


def collect(index:int,row:dict,timeout:float)->tuple[int,dict]:
    hostname=row["canonical_hostname"].strip().lower();domain=row["registered_domain"].strip().lower();results={}
    for prefix,target,kind in QUERIES:results[prefix]=query(hostname if target=="hostname" else domain,kind,timeout)
    features={}
    for prefix,_,_ in QUERIES:
        value=results[prefix];features[f"{prefix}_count"]=value["count"];features[f"{prefix}_minimum_ttl"]=value["minimum_ttl"]
        features[f"{prefix}_query_ms"]=round(value["query_ms"],3);features[f"{prefix}_nxdomain"]=int(value["status"]=="NXDOMAIN")
        features[f"{prefix}_timeout"]=value["timeout"]
    features.update({"dns_address_exists":int(results["host_a"]["count"]+results["host_aaaa"]["count"]>0),
        "dns_mail_exists":int(results["domain_mx"]["count"]>0),"dns_nameserver_exists":int(results["domain_ns"]["count"]>0),
        "dns_total_timeouts":sum(results[p]["timeout"] for p,_,_ in QUERIES),
        "dns_total_nxdomain":sum(results[p]["status"]=="NXDOMAIN" for p,_,_ in QUERIES)})
    audit={prefix:{"status":value["status"],"answers":value["answers"]} for prefix,value in results.items()}
    return index,{**row,**features,"dns_audit_json":json.dumps(audit,separators=(",",":"),sort_keys=True)}


def main():
    args=parse_args()
    if args.output.exists():raise FileExistsError(f"DNS evidence already exists: {args.output}")
    rows=list(csv.DictReader(args.input.open("r",encoding="utf-8",newline="")));selected=[row for row in rows if not args.tier or row.get("risk_tier")==args.tier]
    if not selected:raise RuntimeError(f"No rows selected for tier {args.tier!r}")
    started=time.perf_counter();collected=[None]*len(selected)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(collect,i,row,args.timeout) for i,row in enumerate(selected)]
        for completed,future in enumerate(as_completed(futures),1):
            index,value=future.result();collected[index]=value
            if completed%100==0:print(json.dumps({"completed":completed,"total":len(selected),"elapsed_seconds":time.perf_counter()-started}),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True);fields=list(collected[0].keys())
    with args.output.open("w",encoding="utf-8",newline="") as handle:writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(collected)
    timeout_total=sum(int(row["dns_total_timeouts"]) for row in collected);nxdomain_rows=sum(int(row["dns_total_nxdomain"])>0 for row in collected)
    report={"project_version":3,"evidence":"live DNS records only","collected_at_utc":datetime.now(timezone.utc).isoformat(),"input":str(args.input),"input_sha256":sha256_file(args.input),
        "tier_filter":args.tier,"selected_rows":len(selected),"output":str(args.output),"output_sha256":sha256_file(args.output),"workers":args.workers,"per_query_timeout_seconds":args.timeout,
        "query_types":[f"{target}:{kind}" for _,target,kind in QUERIES],"total_queries":len(selected)*len(QUERIES),"timeout_queries":timeout_total,"rows_with_any_nxdomain":nxdomain_rows,
        "elapsed_seconds":time.perf_counter()-started,"security_scope":"No HTTP connection, page visit, redirect following, TLS handshake, WHOIS/RDAP call, or phishing-content retrieval was performed.",
        "limitations":["DNS is mutable and this collection is a timestamped snapshot.","Resolution failure may reflect blocking or transient network failure rather than maliciousness.","Availability can become a source shortcut; holdout and later temporal evaluation are required."]}
    write_json(args.output.with_suffix(".report.json"),report);print(json.dumps(report,indent=2))


if __name__=="__main__":main()
