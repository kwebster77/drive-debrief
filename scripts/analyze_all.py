"""Batch-analyse every sample drive and build an index page.

    python scripts/analyze_all.py [ROOT] [--out DIR]

Walks ROOT (default sample_data/) for .csv/.gpx/.kml/.kmz/.json drives,
runs the full debrief on each, writes one HTML report per drive, and an
index.html linking them with score + fault summary. This is the whole
pipeline in one command — the automation entry point.
"""
from __future__ import annotations

import argparse
import glob
import html
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from drive_debrief.pipeline import run_debrief  # noqa: E402

EXTS = (".csv", ".gpx", ".kml", ".kmz", ".json")


def _find(root: str):
    files = []
    for ext in EXTS:
        files += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    return sorted(set(files))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="sample_data")
    ap.add_argument("--out", default=os.path.join("out", "reports"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows, ok, failed = [], 0, 0
    for path in _find(args.root):
        name = os.path.relpath(path, args.root).replace(os.sep, "__")
        out_html = os.path.join(args.out, name + ".html")
        try:
            result = run_debrief(path, out_html=out_html)
            s, a = result["summary"], result["assessment"]
            verdict = a["verdict"]
            rows.append((name + ".html", os.path.basename(path), s["score"], s["grade"],
                         len(result["events"]), verdict, a["passed"]))
            ok += 1
            print(f"OK   {s['score']:>3}/{s['grade']}  {len(result['events']):>2} ev  {verdict:<28} {path}")
        except Exception as exc:
            rows.append((None, os.path.basename(path), "-", "-", "-", f"ERROR: {exc}", False))
            failed += 1
            print(f"FAIL {path}: {exc}")

    # Index page.
    def _row(link, src, score, grade, events, verdict, passed):
        colour = "#18794e" if passed else "#cd2b31"
        report = f'<a href="{link}">report</a>' if link else ""
        return (f"<tr><td>{html.escape(src)}</td>"
                f"<td class='mono'>{score}</td><td>{grade}</td><td class='mono'>{events}</td>"
                f"<td style='color:{colour}'>{html.escape(str(verdict))}</td>"
                f"<td>{report}</td></tr>")

    body = "".join(_row(*r) for r in rows)
    index = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>drive-debrief — batch</title>
<style>body{{font-family:-apple-system,sans-serif;background:#f4f5f7;color:#11181c;margin:0}}
.wrap{{max-width:900px;margin:0 auto;padding:28px 20px}}h1{{font-size:22px}}
table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e6e8eb;border-radius:12px;overflow:hidden}}
th,td{{text-align:left;padding:9px 12px;font-size:13px;border-bottom:1px solid #eef0f2}}
th{{background:#fafbfc;color:#687076}}.mono{{font-variant-numeric:tabular-nums}}a{{color:#0091ff}}</style></head>
<body><div class="wrap"><h1>drive-debrief — {ok} drives analysed{', '+str(failed)+' failed' if failed else ''}</h1>
<table><thead><tr><th>Drive</th><th>Score</th><th>Grade</th><th>Events</th><th>Mock test</th><th></th></tr></thead>
<tbody>{body}</tbody></table></div></body></html>"""
    index_path = os.path.join(args.out, "index.html")
    with open(index_path, "w", encoding="utf-8") as fh:
        fh.write(index)

    print(f"\n{ok} analysed, {failed} failed. Index: {index_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
