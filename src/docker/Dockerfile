FROM python:3.11-slim

WORKDIR /app

# System deps (small but useful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first (better caching)
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# If using spaCy, install a small English model (optional but recommended)
# RUN python -m spacy download en_core_web_sm

# Copy code
COPY . /app

# Ensure src layout works
ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "kdsh.main"]


