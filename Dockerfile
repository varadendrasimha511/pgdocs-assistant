FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/models

WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so the first request is not a download
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY app ./app

RUN useradd --create-home appuser && chown -R appuser /srv /models
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]