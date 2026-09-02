"""`drive-debrief-progress` — render the trend report from saved history.

    drive-debrief-progress                       # reads drive_history.json
    drive-debrief-progress -H hist.json -o p.html
"""
from __future__ import annotations

import argparse
import sys

from .history import DEFAULT_HISTORY, load_history
from .progress import build_progress_html


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="drive-debrief-progress", description="Render driving-progress trends from saved drives.")
    p.add_argument("-H", "--history", default=DEFAULT_HISTORY, help="history JSON (default: drive_history.json)")
    p.add_argument("-o", "--out", default="progress.html", help="output HTML (default: progress.html)")
    p.add_argument("--title", default="Driving progress")
    args = p.parse_args(argv)

    entries = load_history(args.history)
    if not entries:
        print(f"No drives found in {args.history}. Run some drives with --save first.", file=sys.stderr)

    html = build_progress_html(entries, title=args.title)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"{len(entries)} drive(s) → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
