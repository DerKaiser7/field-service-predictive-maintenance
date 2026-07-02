FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==2.18.0 psycopg2-binary==2.9.10

EXPOSE 5000

ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
