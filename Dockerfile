FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY mcp_server.py .
COPY agent_client.py .
COPY a2a_server.py .

CMD ["python", "agent_client.py"]