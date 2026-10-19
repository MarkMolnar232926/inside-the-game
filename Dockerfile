# PitchLens API + web UI. Runs fully offline (mock mode) unless Azure settings are provided.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[azure]"
COPY assets ./assets
COPY web ./web
COPY data/synthetic ./data/synthetic
COPY scripts ./scripts
# Regenerate the deterministic synthetic match (tracking parquet is not committed) and the light bundle.
RUN python -m pitchlens.synth.generate --seed 6 --narrative late_comeback --out data/synthetic/demo > /dev/null
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "pitchlens.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
