.PHONY: setup demo test docker-build docker-run clean

PY ?= python3
# Run from ./src without requiring an install (harmless once installed too).
export PYTHONPATH := src

setup:            ## install deps + package (editable)
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

sample:           ## (re)generate the demo drive CSV
	$(PY) scripts/make_synthetic.py

demo: sample      ## generate sample data + run a debrief -> out/debrief.html
	mkdir -p out
	$(PY) -m drive_debrief sample_data/sample_drive.csv -o out/debrief.html

progress: sample  ## save 3 drives + render the trend -> out/progress.html
	mkdir -p out
	rm -f out/demo_history.json
	$(PY) -m drive_debrief sample_data/sample_drive.csv --no-html --save --history out/demo_history.json --label "Week 1" --brake-g 0.25 --lateral-g 0.25
	$(PY) -m drive_debrief sample_data/sample_drive.csv --no-html --save --history out/demo_history.json --label "Week 2" --brake-g 0.32
	$(PY) -m drive_debrief sample_data/sample_drive.csv --no-html --save --history out/demo_history.json --label "Week 3"
	$(PY) -m drive_debrief.progress_cli -H out/demo_history.json -o out/progress.html

samples:          ## generate the full sample-data library (many routes/times/formats)
	$(PY) scripts/generate_samples.py

batch: samples    ## analyse every sample drive -> out/reports/index.html
	$(PY) scripts/analyze_all.py

web:              ## serve the web app on :8000 (upload CSV/GPX or paste a video link)
	$(PY) -m pip install -q -e ".[web]"
	$(PY) -m drive_debrief.webapp

test:             ## run the unit tests
	$(PY) -m pytest -q

docker-build:
	docker build -t drive-debrief .

docker-run:       ## run the containerised demo, report lands in ./out
	mkdir -p out
	docker run --rm -v "$$PWD/out:/app/out" drive-debrief

clean:
	rm -rf out *.debrief.html .pytest_cache **/__pycache__
