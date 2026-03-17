FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Patch OS packages to reduce base image CVEs.
RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Install runtime deps with pinned versions.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only runtime files.
COPY app /app/app
COPY data /app/data
COPY models /app/models

RUN useradd -u 10001 -r -s /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER 10001

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
