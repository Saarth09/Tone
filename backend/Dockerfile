FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p /app/data/chroma

ENV PYTHONUNBUFFERED=1
ENV DEMO_MODE=true
ENV DATABASE_URL=sqlite+aiosqlite:///./data/tone.db
ENV CHROMA_PATH=./data/chroma
ENV ANONYMIZED_TELEMETRY=False
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
