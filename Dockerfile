# Minimal image with the web app on :8000 — drops straight into a cloud sandbox.
FROM python:3.11-slim

# ffmpeg is needed for the video-frame sampling path.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY scripts ./scripts
COPY sample_data ./sample_data

EXPOSE 8000

# Default: serve the web app (upload CSV/GPX or paste a video link).
# For the CLI instead: docker run --entrypoint drive-debrief <img> sample_data/sample_drive.csv
CMD ["drive-debrief-web"]
