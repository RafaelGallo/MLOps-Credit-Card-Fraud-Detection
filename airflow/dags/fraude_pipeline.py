"""DAG do pipeline de fraude: base -> amostra -> treino -> avaliação -> MLflow -> promoção.

Dispara manualmente (schedule=None): o pipeline completo usa a base de ~10 GB e leva dezenas de minutos.
A lógica de cada passo está em src/pipeline_treino.py; aqui só há a orquestração.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.operators.python import get_current_context

import pipeline_treino as pt  # src/ está no PYTHONPATH do container

ENTRADA = Path(os.environ.get("FRAUDE_ENTRADA", "/opt/airflow/input"))  # pasta com credit_card_fraud.csv
TRABALHO = Path(os.environ.get("FRAUDE_TRABALHO", "/opt/airflow/work"))  # Parquet e divisões intermediárias
PRODUCAO = Path(os.environ.get("FRAUDE_PRODUCAO", "/opt/airflow/producao"))  # modelo campeão lido pelo app
CSV = ENTRADA / "credit_card_fraud.csv"
PARQUET = TRABALHO / "credit_card_fraud_full.parquet"  # reaproveitado entre execuções
PASTA_EXECUCAO = TRABALHO / "execucao"

PESADA = {"execution_timeout": timedelta(hours=4), "retries": 0}


@dag(
    dag_id="fraude_cartao_pipeline",
    description="Treina o Random Forest de fraude, registra no MLflow e promove se superar o campeão.",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["fraude", "mlflow"],
    params={
        "n_iter": Param(8, type="integer", minimum=1, maximum=50, description="Combinações testadas na busca aleatória."),
        "fp_custo": Param(10.0, type="number", minimum=0, description="Custo (US$) de revisar um falso alarme."),
        "amostra_um_em": Param(5, type="integer", minimum=1, description="Usa 1 a cada N transações (5 = 20%)."),
        "so_se_melhor": Param(True, type="boolean", description="Só promove se o custo no teste for menor que o do campeão."),
        "refazer_parquet": Param(False, type="boolean", description="Reconverte o CSV mesmo com Parquet atualizado."),
    },
    doc_md=__doc__,
)
def fraude_cartao_pipeline():
    @task(retries=1)
    def validar_base() -> dict:
        return pt.validar_base(CSV)

    @task(**PESADA)
    def converter_parquet(_base: dict) -> dict:
        refazer = get_current_context()["params"]["refazer_parquet"]
        return pt.converter_parquet(CSV, PARQUET, forcar=refazer)

    @task(**PESADA)
    def preparar_amostra(convertido: dict) -> dict:
        um_em = get_current_context()["params"]["amostra_um_em"]
        return pt.preparar_amostra(Path(convertido["parquet"]), PASTA_EXECUCAO, um_em=um_em)

    @task(**PESADA)
    def dividir_dados(_amostra: dict) -> dict:
        return pt.dividir_dados(PASTA_EXECUCAO)

    @task(**PESADA)
    def treinar_avaliar(divisao: dict) -> dict:
        contexto = get_current_context()
        p = contexto["params"]
        return pt.treinar_avaliar(
            PASTA_EXECUCAO, pt.tracking_uri_padrao(), n_iter=p["n_iter"], fp_custo=p["fp_custo"],
            metadados={"airflow_dag_run": contexto["run_id"], "periodo_teste_inicio": divisao["periodo_teste_inicio"]},
        )

    @task
    def promover(resultado: dict) -> dict:
        so_se_melhor = get_current_context()["params"]["so_se_melhor"]
        return pt.promover(resultado, pt.tracking_uri_padrao(), PRODUCAO, so_se_melhor=so_se_melhor)

    base = validar_base()
    convertido = converter_parquet(base)
    amostra = preparar_amostra(convertido)
    divisao = dividir_dados(amostra)
    resultado = treinar_avaliar(divisao)
    promover(resultado)


fraude_cartao_pipeline()
