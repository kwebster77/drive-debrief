"""Write a demo drive CSV to sample_data/.

    python scripts/make_synthetic.py                 # noisy, realistic demo
    python scripts/make_synthetic.py --clean         # noise-free
"""
from __future__ import annotations

import argparse
import os
import sys

# Make the package importable when run straight from the repo.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from drive_debrief.synth import generate_drive  # noqa: E402
from drive_debrief.io import to_gpx  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--out", default="sample_data/sample_drive.csv")
    p.add_argument("--clean", action="store_true", help="no GPS noise (deterministic)")
    p.add_argument("--noise", type=float, default=3.0, help="GPS jitter in metres")
    p.add_argument("--no-gpx", action="store_true", help="don't also write a .gpx sample")
    args = p.parse_args()

    noise = 0.0 if args.clean else args.noise
    speed_noise = 0.0 if args.clean else 0.3
    course_noise = 0.0 if args.clean else 2.0
    df = generate_drive(
        noise_m=noise, speed_noise_mps=speed_noise, course_noise_deg=course_noise
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} samples -> {args.out} (noise={noise} m)")

    if not args.no_gpx:
        gpx_path = os.path.splitext(args.out)[0] + ".gpx"
        with open(gpx_path, "w", encoding="utf-8") as fh:
            fh.write(to_gpx(df))
        print(f"Wrote GPX sample     -> {gpx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
