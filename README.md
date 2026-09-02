# 🚗 drive-debrief

**Record a practice drive → get a coaching debrief.** Point it at a GPS log
of a learner drive and it produces a clean HTML report: a route map, a
smoothness score, and a timestamped list of every harsh brake, harsh
acceleration, hard corner and hesitation — with a plain-English coaching
note for each.

```
Score 87/100 (grade B)  ·  0.43 km  ·  4 event(s) flagged
  00:22  Harsh braking        0.51g   [moderate]
  00:23  Stop                 7.0s    [minor]
  00:33  Harsh acceleration   0.35g   [minor]
  00:38  Hard cornering       0.38g   [minor]
```

## Why this needed to be built (and isn't a prompt)

The value is **signal processing on a sensor stream**, not text generation:
GPS-derived speed and heading → longitudinal/lateral acceleration → event
detection with noise rejection. An LLM + a markdown file can't do any of
that reliably. It's pure-Python, CPU-only, no network — so it runs
identically on a laptop or in a minimal cloud sandbox container.

The robustness comes from doing the telematics *properly*:

- **Longitudinal g from Doppler speed, not differentiated position.** Phone
  logs report speed directly; differentiating noisy GPS position at speed
  produces garbage. (We fall back to derived speed if none is present.)
- **Heading from device course, speed-gated.** A 3 m GPS wobble looks like a
  wild turn when you're crawling, so yaw is zeroed below walking pace and a
  reported course is preferred over a bearing differentiated from position.
- **Minimum-duration + merge gating** so a one-sample spike can't invent an
  event.

## Quick start

**One command** (clean checkout, only `python3` needed):

```bash
./run.sh            # installs deps, makes a sample drive, writes out/debrief.html
```

**Or with make:**

```bash
make setup          # pip install deps + package
make demo           # sample data + debrief -> out/debrief.html
make test           # run the unit tests
```

**Or in a container (the sandbox path):**

```bash
make docker-build
make docker-run     # report lands in ./out/debrief.html
```

Then open `out/debrief.html`.

## Use your own drive

Record a drive with a free phone sensor logger and export CSV:

- **Android** — [phyphox](https://phyphox.org/) → *Location (GPS)* experiment
- **iOS** — [SensorLog](https://apps.apple.com/app/sensorlog/id388014573)

Then:

```bash
drive-debrief my_drive.csv -o my_debrief.html
```

The loader is forgiving about headers — it recognises phyphox and SensorLog
column names automatically. The only required fields are **time, latitude,
longitude**; **speed** and **course** are used if present (recommended).

Canonical CSV, if you're generating your own:

```csv
t,lat,lon,speed,course
0,51.5326,-0.1050,0.0,90
1,51.5326,-0.1049,2.7,90
...
```

## What it detects

| Event | Trigger (default) |
|---|---|
| Harsh braking | longitudinal deceleration > 0.35 g |
| Harsh acceleration | longitudinal acceleration > 0.30 g |
| Hard cornering | lateral acceleration > 0.35 g |
| Stop / long stop | stationary ≥ 3 s / ≥ 25 s |

Thresholds are tunable: `drive-debrief drive.csv --brake-g 0.3 --lateral-g 0.3`.

## CLI

```
drive-debrief INPUT.csv [-o OUT.html] [--json] [--no-html]
                        [--brake-g G] [--accel-g G] [--lateral-g G]
```

`--json` prints a machine-readable summary to **stdout** (human summary goes
to stderr, so it pipes cleanly).

## How it's organised

```
src/drive_debrief/
  io.py          load + normalise CSV (phyphox / SensorLog aware)
  geo.py         haversine / bearing (scalar + vectorised)
  kinematics.py  speed, heading, longitudinal & lateral g  ← the robust core
  events.py      threshold + duration + merge event detection
  scoring.py     smoothness score & summary
  report.py      self-contained HTML + inline SVG map (no network)
  pipeline.py    CSV in → report + summary out
  cli.py         command line
  synth.py       synthetic drive with *known* events (demo + test ground truth)
tests/           unit tests (maths, detectors, noise-robustness)
Dockerfile       CPU-only image for a cloud sandbox
```

## Testing

```bash
make test
```

The synthetic generator builds a drive from an explicit speed + yaw profile,
so the tests assert the detectors recover events at known times — including a
run with realistic GPS jitter to prove noise-robustness.

## Roadmap

- Speed-limit overlay (OSM `maxspeed`) → speeding detection
- Optional vision garnish: run a small model on dashcam frames to catch
  red-light stops, fused with the speed trace
- Trend across multiple drives (is the learner improving?)

---

*Practice feedback, not a substitute for a qualified instructor.*
