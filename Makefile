.PHONY: setup demo test docker-build docker-run clean

PY ?= python3

setup:            ## install deps + package (editable)
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

sample:           ## (re)generate the demo drive CSV
	$(PY) scripts/make_synthetic.py

demo: sample      ## generate sample data + run a debrief -> out/debrief.html
	mkdir -p out
	$(PY) -m drive_debrief sample_data/sample_drive.csv -o out/debrief.html

test:             ## run the unit tests
	$(PY) -m pytest -q

docker-build:
	docker build -t drive-debrief .

docker-run:       ## run the containerised demo, report lands in ./out
	mkdir -p out
	docker run --rm -v "$$PWD/out:/app/out" drive-debrief

clean:
	rm -rf out *.debrief.html .pytest_cache **/__pycache__
