#!/usr/bin/env bash
# One-shot: set up, generate sample data, run the debrief, open the report.
# Works from a clean checkout with only python3 available.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

echo "==> Installing dependencies"
$PY -m pip install -q -r requirements.txt

echo "==> Generating a sample drive"
$PY scripts/make_synthetic.py

echo "==> Running the debrief"
mkdir -p out
PYTHONPATH=src $PY -m drive_debrief sample_data/sample_drive.csv -o out/debrief.html

echo
echo "Done. Open: out/debrief.html"
