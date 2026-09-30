FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN mkdir -p /app/input /app/output

ENV INPUT_DIR=/app/input
ENV OUTPUT_DIR=/app/output

CMD ["python", "app/main.py"]