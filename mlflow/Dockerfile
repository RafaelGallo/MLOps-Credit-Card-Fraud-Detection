FROM python:3.12-slim

# psycopg2-binary: o MLflow guarda experimentos e o registro de modelos no Postgres
RUN pip install --no-cache-dir "mlflow==3.17.2" "psycopg2-binary>=2.9,<3"

WORKDIR /mlflow
EXPOSE 5000

# o comando completo (backend, artefatos, hosts) fica no docker-compose.yml
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
