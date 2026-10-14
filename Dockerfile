FROM python:3.11-slim

# System packages are added here as the pipeline needs them
# (ffmpeg for video, libreoffice-impress for PPTX), to keep early builds fast.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Download the embedding model while building, so the first question is not slow
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/home/user/.cache/fastembed')"

COPY --chown=user . .
RUN mkdir -p data/runtime

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]