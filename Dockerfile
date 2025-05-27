FROM python:3.13.3-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SENTENCE_TRANSFORMERS_HOME="/app/assets/embedding_model"

RUN groupadd -r appgroup && useradd -r -g appgroup -ms /bin/bash appuser

WORKDIR /app

RUN mkdir -p /app/assets/embedding_model \
             /app/assets/markdown \
             /app/assets/chunks \
             /app/assets/vector_store && \
    chown -R appuser:appgroup /app

COPY --chown=appuser:appgroup requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appgroup frontend/ ./frontend/
COPY --chown=appuser:appgroup rag/ ./rag/
COPY --chown=appuser:appgroup data_processing/ ./data_processing/
COPY --chown=appuser:appgroup assets/embedding_model/BAAI/bge-base-en-v1.5/ ./assets/embedding_model/BAAI/bge-base-en-v1.5/

USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]