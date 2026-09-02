# Minimal, CPU-only image — drops straight into any cloud sandbox.
FROM python:3.11-slim

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install the package.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# Sample data + generator so the image can demo itself with no inputs.
COPY scripts ./scripts
COPY sample_data ./sample_data

# Default: analyse the bundled sample drive and write a report.
ENTRYPOINT ["drive-debrief"]
CMD ["sample_data/sample_drive.csv", "-o", "/app/out/debrief.html", "--json"]
