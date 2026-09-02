"""Command-line entry point.

    drive-debrief drive.csv                 -> writes drive.debrief.html
    drive-debrief drive.csv -o out.html     -> custom output path
    drive-debrief drive.csv --json          -> also print summary JSON to stdout
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

from .events import Thresholds
from .history import DEFAULT_HISTORY, append_run, build_entry
from .pipeline import run_debrief


def _default_out(csv_path: str) -> str:
    base, _ = os.path.splitext(csv_path)
    return base + ".debrief.html"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="drive-debrief", description="Turn a recorded practice drive into a coaching debrief.")
    p.add_argument("input", help="CSV of the drive (t, lat, lon, [speed]); phyphox/SensorLog exports accepted")
    p.add_argument("-o", "--out", help="output HTML path (default: <input>.debrief.html)")
    p.add_argument("--title", default="Practice-drive debrief", help="report title")
    p.add_argument("--json", action="store_true", help="print the summary as JSON to stdout")
    p.add_argument("--no-html", action="store_true", help="skip writing HTML (use with --json)")
    p.add_argument("--save", action="store_true", help="append this drive to the progress history")
    p.add_argument("--history", default=DEFAULT_HISTORY, help="history JSON path (with --save)")
    p.add_argument("--label", help="name for this drive in the history (default: file name)")
    p.add_argument("--brake-g", type=float, default=Thresholds.brake_g)
    p.add_argument("--accel-g", type=float, default=Thresholds.accel_g)
    p.add_argument("--lateral-g", type=float, default=Thresholds.lateral_g)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.exists(args.input):
        print(f"error: no such file: {args.input}", file=sys.stderr)
        return 2

    thresholds = Thresholds(brake_g=args.brake_g, accel_g=args.accel_g, lateral_g=args.lateral_g)
    out_html = None if args.no_html else (args.out or _default_out(args.input))

    try:
        result = run_debrief(args.input, out_html=out_html, thresholds=thresholds, title=args.title)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    s = result["summary"]
    n_events = len(result["events"])

    # When --json is requested, keep stdout pure JSON so it can be piped;
    # send the human-readable summary to stderr.
    human = sys.stderr if args.json else sys.stdout
    a = result["assessment"]
    print(f"Score {s['score']}/100 (grade {s['grade']})  ·  "
          f"{s['distance_km']} km  ·  {n_events} event(s) flagged", file=human)
    print(f"Mock test: {a['verdict']}  ·  "
          f"{a['minors']} driving / {a['serious']} serious / {a['dangerous']} dangerous", file=human)
    for e in result["events"]:
        mm = int(e["t_peak"] // 60)
        ss = int(e["t_peak"] % 60)
        print(f"  {mm:02d}:{ss:02d}  {e['label']:<20} {e['peak_value']}{e['unit']:<3} [{e['severity']}]", file=human)
    if args.save:
        label = args.label or os.path.splitext(os.path.basename(args.input))[0]
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        entry = build_entry(label, stamp, result)
        history = append_run(entry, args.history)
        print(f"Saved '{label}' to {args.history} ({len(history)} drive(s) tracked)", file=human)

    if result["report_path"]:
        print(f"\nReport: {result['report_path']}", file=human)
    if args.json:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
