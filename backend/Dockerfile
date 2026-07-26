FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt .

RUN pip install --no-cache-dir \
    -r requirements.txt

COPY backend/ .

ENV ARTIFACT_DIR=/app/artifacts

EXPOSE 10000

CMD ["sh", "start.sh"]