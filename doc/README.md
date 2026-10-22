# Documentação: detecção de fraude em cartão de crédito

Projeto de ciência de dados sobre 34,6 milhões de transações de cartão (jan/2019 a dez/2020, dados sintéticos),
com análise exploratória, modelo de classificação, app Streamlit e uma esteira de MLOps com Airflow, MLflow e Docker.

## Índice

| Documento | Para quê |
|---|---|
| [modelo-e-dados.md](modelo-e-dados.md) | Base, método, resultados e limitações do modelo |
| [pipeline-airflow.md](pipeline-airflow.md) | A DAG: tarefas, parâmetros, arquivos gerados e promoção do modelo |
| [mlflow.md](mlflow.md) | O que é registrado, o registro de modelos e o alias `champion` |
| [guia-docker.md](guia-docker.md) | Como subir Airflow, MLflow e o app com `docker compose` |
| [app-streamlit.md](app-streamlit.md) | As abas do app, o mapa, execução local e deploy |

## Visão geral

```
 input/credit_card_fraud.csv (~10 GB, só na sua máquina)
            │  montada somente leitura
            ▼
 ┌─────────────────────────── Airflow (DAG fraude_cartao_pipeline) ───────────────────────────┐
 │ validar_base → converter_parquet → preparar_amostra → dividir_dados → treinar_avaliar → promover │
 └───────────────────────────────────────────────┬──────────────────────────────┬─────────────┘
                                                 │ métricas, parâmetros,        │ se vencer o campeão:
                                                 │ modelo                       │ publica o pacote
                                                 ▼                              ▼
                                  MLflow (Postgres + artefatos)       mlflow/producao/modelo_final.joblib
                                  experimento "fraude-cartao"                   │
                                  registro "fraude_cartao_random_forest"        ▼
                                  alias champion                       App Streamlit (porta 8501)
```

Portas locais: Airflow `8080`, MLflow `5000`, app `8501`.

## Estrutura das pastas

| Pasta ou arquivo | Conteúdo |
|---|---|
| `app/streamlit_app.py` | Interface do app (3 abas) |
| `src/modelo.py` | Carga do modelo, validação de lote, predição e métricas (sem Streamlit) |
| `src/mapa.py` | Mapas de bolhas por local do cliente |
| `src/pipeline_treino.py` | Passos do pipeline de treino, usados pela DAG e executáveis sozinhos |
| `airflow/dags/fraude_pipeline.py` | A DAG do Airflow |
| `docker/` | `docker-compose.yml`, imagens do Airflow e do app, `.env.example` |
| `mlflow/` | Imagem do servidor MLflow e a pasta `producao/` (modelo campeão publicado) |
| `notebook/`, `py/` | Notebooks de EDA e de modelagem e o `.py` exportado do EDA |
| `models/` | Modelo final do repositório (`modelo_final.joblib`); os demais arquivos são locais |
| `input/` | Base de dados (não vai para o git) |
| `retreino/` | Execuções locais dos notebooks e dados intermediários do Airflow (não vão para o git) |
| `output/` | CSV de exemplo para o app |
| `doc/` | Esta documentação |

## Decisões de projeto

- **O pipeline de treino é código Python comum** (`src/pipeline_treino.py`). A DAG só orquestra, então o mesmo
  código roda no Airflow, num terminal ou num teste.
- **Os passos trocam arquivos, não dados pelo XCom**, porque a base tem dezenas de milhões de linhas.
- **O app nunca é sobrescrito pelo treino.** A DAG publica o campeão em `mlflow/producao/`, e o app prefere esse
  arquivo, usando o modelo do repositório como alternativa.
- **Mesma versão do scikit-learn em todo lugar** (`1.5.2`): o modelo é um pickle e só carrega com segurança na série
  que o gerou.
