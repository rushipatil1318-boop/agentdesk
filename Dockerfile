FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY agentdesk ./agentdesk
COPY data ./data
COPY eval ./eval
COPY scripts ./scripts
EXPOSE 8000
CMD uvicorn agentdesk.api:app --host 0.0.0.0 --port 8000
