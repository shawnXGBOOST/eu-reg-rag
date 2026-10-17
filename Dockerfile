FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "eu_reg_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
