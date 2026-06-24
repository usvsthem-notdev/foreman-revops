FROM python:3.11-slim

# Security: run as non-root
RUN groupadd -r foreman && useradd -r -g foreman foreman

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY app.py .
COPY .streamlit/ ./.streamlit/

# Data volume for the SQLite database
RUN mkdir -p /app/data && chown -R foreman:foreman /app/data
VOLUME ["/app/data"]

ENV FOREMAN_DB_PATH=/app/data/foreman.db

USER foreman

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
