FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt \
 && useradd -m nids
COPY . .
RUN mkdir -p artifacts && chown -R nids:nids /app
USER nids
EXPOSE 8501 8000
# Default: dashboard. Override command for API / training (see docker-compose.yml).
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
