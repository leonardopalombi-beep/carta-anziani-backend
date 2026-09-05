FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./

ENV PORT=8001
EXPOSE 8001

CMD ["sh", "-c", "python -m uvicorn server:app --host 0.0.0.0 --port ${PORT:-8001}"]
