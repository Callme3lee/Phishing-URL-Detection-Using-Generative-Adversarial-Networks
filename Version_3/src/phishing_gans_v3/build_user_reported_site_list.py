from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


SOURCE = Path("0.5.1 Results")
OUTPUT = Path("../Documentation/user_reported_sites_2026-09-07")
TIER_ORDER = {
    "unknown": -1, "allow": 0, "trusted": 0, "secondary_review": 1,
    "ip_review": 1, "caution": 2, "warning": 3, "block": 4,
}
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f-]{18,}$", re.I)
LONG_TOKEN = re.compile(r"^(?=.{18,}$)(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9_.-]+$")
SESSION = re.compile(r"^cpsess\d+$", re.I)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sanitize_url(raw: str) -> str:
    """Remove material that should not be reproduced in a shareable test list."""
    parsed = urlsplit(raw)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{hostname}{port}"
    parts = [unquote(part) for part in parsed.path.split("/")]
    sanitized: list[str] = []
    previous = ""
    for part in parts:
        redact = (
            "..." in part
            or UUID.fullmatch(part) is not None
            or LONG_TOKEN.fullmatch(part) is not None
            or SESSION.fullmatch(part) is not None
            or previous.lower() in {"d", "folders", "file", "c", "skill"}
        )
        sanitized.append("[redacted-id]" if redact and part else part)
        previous = part
    path = "/".join(quote(part, safe="[]:@-._~") for part in sanitized)
    return urlunsplit((parsed.scheme.lower() or "https", netloc, path or "/", "", ""))


def load_reports() -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    manifest: list[dict] = []
    for path in sorted(SOURCE.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        feedback = payload.get("feedbackReports", [])
        manifest.append({
            "source": path.name, "sha256": sha256(path),
            "extension_version": payload.get("extensionVersion"),
            "feedback_records": len(feedback),
        })
        for item in feedback:
            raw_url = item.get("reportedUrl", "")
            parsed = urlsplit(raw_url)
            hostname = (parsed.hostname or "").lower().rstrip(".")
            if not hostname:
                continue
            records.append({
                "source": path.stem,
                "claim": item.get("claim", "unknown"),
                "hostname": hostname,
                "sanitized_url": sanitize_url(raw_url),
                "effective_tier": item.get("effectiveTier", "unknown"),
                "model_tier": item.get("modelTier", "unknown"),
                "score": float(item["score"]) if item.get("score") is not None else None,
            })
    return records, manifest


def aggregate_hosts(records: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        grouped[row["hostname"]].append(row)
    output: list[dict] = []
    for hostname, rows in grouped.items():
        claims = {row["claim"] for row in rows}
        label = "Both claims" if len(claims) > 1 else (
            "Missed phishing" if "missed_phishing" in claims else "False alarm"
        )
        raw_tier = max((row["model_tier"] for row in rows), key=lambda value: TIER_ORDER.get(value, -1))
        scores = [row["score"] for row in rows if row["score"] is not None]
        output.append({
            "hostname": hostname,
            "user_reported_category": label,
            "report_count": len(rows),
            "highest_raw_model_tier": raw_tier,
            "maximum_score": max(scores) if scores else "",
            "source_exports": "; ".join(sorted({row["source"] for row in rows})),
        })
    category_order = {"False alarm": 0, "Both claims": 1, "Missed phishing": 2}
    return sorted(output, key=lambda row: (category_order[row["user_reported_category"]], row["hostname"]))


def deduplicate_patterns(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(row["claim"], row["sanitized_url"])].append(row)
    output: list[dict] = []
    for (claim, url), rows in grouped.items():
        tier = max((row["model_tier"] for row in rows), key=lambda value: TIER_ORDER.get(value, -1))
        scores = [row["score"] for row in rows if row["score"] is not None]
        output.append({
            "user_claim": "Missed phishing" if claim == "missed_phishing" else "False alarm",
            "sanitized_test_url": url,
            "raw_model_tier": tier,
            "maximum_score": max(scores) if scores else "",
            "report_count": len(rows),
            "source_exports": "; ".join(sorted({row["source"] for row in rows})),
        })
    return sorted(output, key=lambda row: (row["user_claim"] == "Missed phishing", row["sanitized_test_url"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def page_header(fig, title: str, subtitle: str = "") -> None:
    fig.text(.06, .955, title, fontsize=20, weight="bold", color="#17324D", va="top")
    if subtitle:
        fig.text(.06, .918, subtitle, fontsize=9.5, color="#526777", va="top")
    fig.lines.append(plt.Line2D([.06, .94], [.895, .895], transform=fig.transFigure, color="#2C7A7B", lw=2))


def table_pages(pdf: PdfPages, title: str, subtitle: str, headers: list[str], rows: list[list[str]],
                widths: list[float], rows_per_page: int) -> None:
    for page, start in enumerate(range(0, len(rows), rows_per_page), 1):
        chunk = rows[start:start + rows_per_page]
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        suffix = f" — page {page}" if len(rows) > rows_per_page else ""
        page_header(fig, title + suffix, subtitle)
        axis = fig.add_axes([.055, .07, .89, .80]); axis.axis("off")
        table = axis.table(cellText=chunk, colLabels=headers, cellLoc="left", colLoc="left",
                           colWidths=widths, loc="upper left", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False); table.set_fontsize(7.4)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#D6E0E5"); cell.set_linewidth(.45)
            if row == 0:
                cell.set_facecolor("#17324D"); cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            elif row % 2 == 0:
                cell.set_facecolor("#F3F7F8")
            if col == 0 and row > 0:
                cell.get_text().set_weight("bold")
        fig.text(.94, .035, f"Rows {start + 1}–{start + len(chunk)} of {len(rows)}",
                 fontsize=8, color="#687A86", ha="right")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def make_pdf(path: Path, records: list[dict], hosts: list[dict], patterns: list[dict]) -> None:
    false_records = sum(row["claim"] == "false_alarm" for row in records)
    missed_records = sum(row["claim"] == "missed_phishing" for row in records)
    false_hosts = sum(row["user_reported_category"] in {"False alarm", "Both claims"} for row in hosts)
    missed_hosts = sum(row["user_reported_category"] in {"Missed phishing", "Both claims"} for row in hosts)
    with PdfPages(path, metadata={"Title": "Websites Reported During Extension User Evaluation"}) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.patches.append(plt.Rectangle((0, .78), 1, .22, transform=fig.transFigure, color="#17324D"))
        fig.text(.06, .91, "Websites Reported by Users", fontsize=28, weight="bold", color="white")
        fig.text(.06, .845, "Extension evaluation test list · privacy-redacted", fontsize=14, color="#DDEBED")
        cards = [
            ("151", "feedback records"), (str(len(hosts)), "distinct hostnames"),
            (str(false_hosts), "false-alarm hosts"), (str(missed_hosts), "missed-phishing hosts"),
        ]
        for index, (value, label) in enumerate(cards):
            x = .06 + index * .225
            fig.text(x, .68, value, fontsize=25, weight="bold", color="#2C7A7B")
            fig.text(x, .635, label, fontsize=10, color="#526777")
        text = (
            f"The source exports contain {false_records} user-submitted false-alarm reports and "
            f"{missed_records} user-submitted missed-phishing reports. These are user claims, not independently "
            "verified ground-truth labels. Repeated reports are retained in the counts and consolidated in the lists.\n\n"
            "For privacy, the PDF removes query strings, fragments, credentials, document identifiers, conversation "
            "identifiers, session identifiers and other long path tokens. The original exports remain unchanged."
        )
        fig.text(.06, .52, text, fontsize=12, color="#273B49", linespacing=1.5, va="top", wrap=True)
        fig.text(.06, .25, "SAFE TESTING", fontsize=12, weight="bold", color="#A33A3A")
        fig.text(.06, .205,
                 "Paste strings into the extension's Offline checker. Do not open or navigate to entries marked\n"
                 "Missed phishing. A redacted pattern may not reproduce the original exact score.",
                 fontsize=12, color="#273B49", linespacing=1.5, va="top")
        fig.text(.06, .07, "Generated from the JSON exports in Version_3/0.5.1 Results · 7 September 2026",
                 fontsize=9, color="#687A86")
        pdf.savefig(fig); plt.close(fig)

        host_rows = []
        for index, row in enumerate(hosts, 1):
            score = "—" if row["maximum_score"] == "" else f"{100 * float(row['maximum_score']):.2f}%"
            host_rows.append([str(index), row["hostname"], row["user_reported_category"],
                              str(row["report_count"]), row["highest_raw_model_tier"], score])
        table_pages(pdf, "Reported website list",
                    "Category is the user's submitted claim; it is not independent verification.",
                    ["#", "Hostname", "User claim", "Reports", "Highest tier", "Max score"],
                    host_rows, [.045, .36, .19, .09, .15, .11], 28)

        pattern_rows = []
        for index, row in enumerate(patterns, 1):
            score = "—" if row["maximum_score"] == "" else f"{100 * float(row['maximum_score']):.2f}%"
            pattern_rows.append([str(index), row["sanitized_test_url"], row["user_claim"],
                                 row["raw_model_tier"], score])
        table_pages(pdf, "Privacy-redacted URL patterns",
                    "Use the Offline checker only. [redacted-id] replaces potentially identifying path content.",
                    ["#", "Sanitized URL pattern", "User claim", "Tier", "Score"],
                    pattern_rows, [.04, .56, .17, .12, .09], 22)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {OUTPUT}")
    records, manifest = load_reports()
    if not records:
        raise RuntimeError("No feedback records found")
    hosts = aggregate_hosts(records)
    patterns = deduplicate_patterns(records)
    OUTPUT.mkdir(parents=True)
    write_csv(OUTPUT / "reported_hosts.csv", hosts)
    write_csv(OUTPUT / "sanitized_test_urls.csv", patterns)
    summary = {
        "source_directory": str(SOURCE),
        "source_manifest": manifest,
        "feedback_records": len(records),
        "false_alarm_records": sum(row["claim"] == "false_alarm" for row in records),
        "missed_phishing_records": sum(row["claim"] == "missed_phishing" for row in records),
        "distinct_hostnames": len(hosts),
        "false_alarm_hostnames": sum(row["user_reported_category"] in {"False alarm", "Both claims"} for row in hosts),
        "missed_phishing_hostnames": sum(row["user_reported_category"] in {"Missed phishing", "Both claims"} for row in hosts),
        "distinct_sanitized_patterns": len(patterns),
        "labels_independently_verified": False,
        "privacy": "Query, fragment and identifying path tokens are removed from generated outputs.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    make_pdf(OUTPUT / "user_reported_websites_test_list.pdf", records, hosts, patterns)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
