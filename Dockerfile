FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

RUN mkdir -p uploads output && chmod -R 777 uploads output

EXPOSE 7860

CMD ["sh", "-c", "gunicorn server:app --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 180"]
