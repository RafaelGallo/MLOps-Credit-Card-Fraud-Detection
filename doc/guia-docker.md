# Guia: Airflow, MLflow e app com Docker

## Pré-requisitos

- Docker Desktop (ou Docker Engine) com o Compose v2, **em execução**.
- Pelo menos 8 GB de RAM livres para o Docker. O Random Forest usa todos os núcleos disponíveis.
- A base em `input/credit_card_fraud.csv`.

## Subir

```bash
cd docker
cp .env.example .env
```

Edite `docker/.env`: troque as senhas e gere a chave do Airflow:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Cole a saída em `AIRFLOW_FERNET_KEY`. Depois:

```bash
docker compose up --build
```

A primeira vez baixa as imagens e leva alguns minutos. Quando tudo estiver saudável:

| Serviço | Endereço | Login |
|---|---|---|
| Airflow | http://localhost:8080 | `AIRFLOW_USER` / `AIRFLOW_PASSWORD` do `.env` |
| MLflow | http://localhost:5000 | sem login (somente local) |
| App Streamlit | http://localhost:8501 | sem login |

## Rodar o pipeline

1. No Airflow, abra a DAG `fraude_cartao_pipeline`.
2. **Trigger DAG w/ config** (botão de play com seta). Ajuste os parâmetros, ou deixe os padrões.
3. Acompanhe as tarefas no *Grid*. Os resultados aparecem no MLflow (`fraude-cartao`).
4. Se o modelo for promovido, o app passa a usá-lo na próxima interação, sem reiniciar.

Para um teste rápido, aponte `DADOS_DIR` do `.env` para uma pasta com um `credit_card_fraud.csv` menor (precisa ter
as colunas `unix_time, amt, city_pop, lat, long, merch_lat, merch_long, category, gender, state, is_fraud`).

## Serviços

| Serviço | Imagem | Função |
|---|---|---|
| `postgres` | `postgres:16` | Bancos `airflow` e `mlflow` (o segundo é criado por `docker/postgres-init/`) |
| `mlflow` | `mlflow/Dockerfile` (Python 3.12 + MLflow 3.17.2) | Tracking e Model Registry |
| `airflow-init` | `docker/Dockerfile.airflow` | Migra o banco e cria o usuário; roda e termina |
| `airflow-webserver`, `airflow-scheduler` | `docker/Dockerfile.airflow` (Airflow 2.10.5, `LocalExecutor`) | Interface e agendador |
| `app` | `docker/Dockerfile.app` | Streamlit com o modelo do repositório, ou o campeão publicado |

## Volumes e pastas montadas

| No contêiner | Origem | Observação |
|---|---|---|
| `/opt/airflow/dags` | `airflow/dags/` | Alterações na DAG valem sem rebuild |
| `/opt/airflow/src` | `src/` (somente leitura) | Código do pipeline, no `PYTHONPATH` |
| `/opt/airflow/input` | `DADOS_DIR` (somente leitura) | A base de dados |
| `/opt/airflow/work` | `retreino/airflow/` | Parquet e divisões intermediárias (fora do git) |
| `/opt/airflow/producao` e `/producao` | `mlflow/producao/` | Onde a DAG publica o campeão; o app lê |
| volume `postgres-dados` | Docker | Bancos do Airflow e do MLflow |
| volume `mlflow-artefatos` | Docker | Artefatos dos modelos |

## Comandos úteis

```bash
docker compose ps                      # estado dos serviços
docker compose logs -f mlflow          # logs de um serviço
docker compose down                    # para tudo, mantém os dados
docker compose down -v                 # para e APAGA bancos e artefatos
docker compose up --build app          # reconstrói só o app (após mudar o código)
```

## Problemas comuns

- **`error during connect` / `dockerDesktopLinuxEngine`**: o Docker Desktop não está aberto.
- **Airflow não sobe e fala de Fernet**: faltou `AIRFLOW_FERNET_KEY` no `.env`.
- **`Invalid Host header` no MLflow**: o nome usado para acessar não está em `--allowed-hosts` (no `docker-compose.yml`).
- **Permissão negada em `retreino/airflow` (Linux)**: defina `AIRFLOW_UID` com o resultado de `id -u`.
- **Build do app muito lento**: confira se o `.dockerignore` da raiz está presente; sem ele o Docker tenta enviar
  os 10 GB de `input/`.
- **Mudou o banco do Postgres e não aplicou**: o script de `postgres-init/` só roda na criação do volume. Use
  `docker compose down -v` para recriar (apaga os dados).

## Segurança

Os valores de `.env.example` são só para uso local. O `.env` está no `.gitignore`. Não exponha as portas 5000 e
8080 na internet; o MLflow não tem autenticação.
