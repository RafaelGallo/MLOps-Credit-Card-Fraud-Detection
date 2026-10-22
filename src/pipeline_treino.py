"""Pipeline de treino do modelo de fraude: base -> amostra -> treino -> avaliação -> MLflow.

Reproduz o método do notebook de modelagem em passos independentes. Cada função é uma
tarefa da DAG (airflow/dags/fraude_pipeline.py) e também roda sozinha, fora do Airflow.
Os passos trocam dados por arquivos em `pasta`, nunca por XCom, porque a base é grande.
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import OrdinalEncoder

NUM_COLS = ["amt", "city_pop", "lat", "long", "merch_lat", "merch_long"]
CAT_COLS = ["category", "gender", "state"]
COLUNAS_CSV = ["unix_time"] + NUM_COLS + CAT_COLS + ["is_fraud"]  # sem colunas pessoais (ssn, cc_num, nome...)
DATA_CORTE = datetime(2020, 12, 24)  # depois disso os rótulos de fraude estão incompletos
NOME_MODELO = "fraude_cartao_random_forest"
FP_CUSTO_PADRAO = 10.0  # custo assumido de revisar uma compra legítima (US$)

# espaço de busca do Random Forest, igual ao do notebook (Parte 15)
ESPACO_RF = {
    "n_estimators": [100, 200, 300],
    "max_depth": [6, 8, 10, 14, 20],
    "min_samples_leaf": [20, 50, 100, 200, 500],
    "max_features": ["sqrt", 0.5, 0.8],
    "max_samples": [0.3, 0.5, 0.8],
    "class_weight": [None, "balanced_subsample"],
}


# ------------------------------------------------------------------ 1. base
def validar_base(csv: Path) -> dict:
    """Confere se a base existe e tem as colunas que o pipeline usa."""
    csv = Path(csv)
    if not csv.exists():
        raise FileNotFoundError(f"Base não encontrada: {csv}. Coloque credit_card_fraud.csv na pasta de dados.")
    amostra = pl.read_csv(csv, n_rows=1000)
    ausentes = [c for c in COLUNAS_CSV if c not in amostra.columns]
    if ausentes:
        raise ValueError(f"Colunas ausentes na base: {ausentes}")
    return {"csv": str(csv), "tamanho_gb": round(csv.stat().st_size / 1e9, 2)}


def converter_parquet(csv: Path, parquet: Path, forcar: bool = False) -> dict:
    """CSV -> Parquet só com as colunas do modelo. Pula se o Parquet já está atualizado."""
    csv, parquet = Path(csv), Path(parquet)
    if parquet.exists() and parquet.stat().st_mtime >= csv.stat().st_mtime and not forcar:
        pulado = True
    else:
        pulado = False
        parquet.parent.mkdir(parents=True, exist_ok=True)
        temporario = parquet.with_suffix(".tmp")
        (
            pl.scan_csv(csv, infer_schema_length=10000)
            .select(COLUNAS_CSV)
            .with_columns(
                pl.col("is_fraud").cast(pl.Boolean),
                pl.from_epoch("unix_time", time_unit="s").alias("trans_dt"),
            )
            .sink_parquet(temporario)
        )
        temporario.replace(parquet)  # só aparece com o nome final quando está completo
    linhas = pl.scan_parquet(parquet).select(pl.len()).collect().item()
    return {"parquet": str(parquet), "linhas": int(linhas), "reaproveitado": pulado}


# ------------------------------------------------------------------ 2. amostra
def preparar_amostra(parquet: Path, pasta: Path, um_em: int = 5) -> dict:
    """Amostra de 1 em `um_em` linhas (hash, mantém a taxa real de fraude), antes da data de corte, ordenada no tempo."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    chave = pl.struct(["trans_dt", "amt", "lat", "long"])
    amostra = (
        pl.scan_parquet(parquet)
        .filter(pl.col("trans_dt") < pl.lit(DATA_CORTE))
        .select(["trans_dt"] + NUM_COLS + CAT_COLS + ["is_fraud"])
        .filter(chave.hash(seed=5) % um_em == 0)
        .sort("trans_dt")
        .collect()
    )
    destino = pasta / "amostra.parquet"
    amostra.write_parquet(destino)
    return {
        "amostra": str(destino), "linhas": amostra.height,
        "taxa_fraude": float(amostra["is_fraud"].mean()),
        "inicio": str(amostra["trans_dt"].min()), "fim": str(amostra["trans_dt"].max()),
    }


# ------------------------------------------------------------------ 3. divisão
def _codificar(df: pd.DataFrame, codificador: OrdinalEncoder) -> pd.DataFrame:
    """Numéricas + categorias como códigos inteiros (desconhecida = -1), mesma regra do app."""
    matriz = np.hstack([df[NUM_COLS].to_numpy(dtype=float), codificador.transform(df[CAT_COLS].astype(str))])
    return pd.DataFrame(matriz, columns=NUM_COLS + CAT_COLS)


def dividir_dados(pasta: Path, razao_subamostragem: int = 10, semente: int = 42) -> dict:
    """Divisão no tempo: teste = últimos 20%; do treino, últimos 20% viram validação; subamostragem 1:10 só no ajuste."""
    pasta = Path(pasta)
    amostra = pd.read_parquet(pasta / "amostra.parquet")
    n_teste = math.ceil(len(amostra) * 0.2)  # igual a train_test_split(test_size=0.2, shuffle=False)
    treino_total, teste = amostra.iloc[: len(amostra) - n_teste], amostra.iloc[len(amostra) - n_teste:]
    corte = int(len(treino_total) * 0.8)
    ajuste, validacao = treino_total.iloc[:corte], treino_total.iloc[corte:]

    rng = np.random.default_rng(semente)
    y_aj = ajuste["is_fraud"].astype(int).to_numpy()
    idx_fraude = np.where(y_aj == 1)[0]
    idx_legit = np.where(y_aj == 0)[0]
    idx_legit = rng.choice(idx_legit, size=min(razao_subamostragem * len(idx_fraude), len(idx_legit)), replace=False)
    ajuste_sub = ajuste.iloc[np.sort(np.concatenate([idx_fraude, idx_legit]))]

    codificador = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(ajuste_sub[CAT_COLS].astype(str))
    resumo = {}
    for nome, df in (("ajuste", ajuste_sub), ("validacao", validacao), ("teste", teste)):
        saida = _codificar(df, codificador)
        saida["is_fraud"] = df["is_fraud"].astype(int).to_numpy()
        saida.to_parquet(pasta / f"{nome}.parquet", index=False)
        resumo[nome] = {"linhas": len(saida), "taxa_fraude": float(saida["is_fraud"].mean())}
    joblib.dump(codificador, pasta / "codificador.joblib")
    resumo["periodo_teste_inicio"] = str(teste["trans_dt"].min())
    resumo["periodo_validacao_inicio"] = str(validacao["trans_dt"].min())
    return resumo


# ------------------------------------------------------------------ 4. treino e avaliação
def custo_total(y: np.ndarray, previsto: np.ndarray, valores: np.ndarray, fp_custo: float) -> float:
    """Valor das fraudes que passaram + custo de revisão de cada falso alarme (regra do notebook)."""
    return float(valores[(y == 1) & ~previsto].sum() + fp_custo * ((y == 0) & previsto).sum())


def escolher_limiar(proba: np.ndarray, y: np.ndarray, valores: np.ndarray, fp_custo: float) -> tuple[float, float]:
    """Limiar de menor custo na validação, testando quantis de 10% a 0,01% de alertas."""
    limiares = np.unique(np.quantile(proba, np.linspace(0.90, 0.9999, 250)))
    custos = [custo_total(y, proba >= t, valores, fp_custo) for t in limiares]
    melhor = int(np.argmin(custos))
    return float(limiares[melhor]), float(custos[melhor])


def treinar_avaliar(pasta: Path, tracking_uri: str, experimento: str = "fraude-cartao", n_iter: int = 8,
                    fp_custo: float = FP_CUSTO_PADRAO, semente: int = 42, metadados: dict | None = None) -> dict:
    """Busca aleatória do Random Forest, limiar por custo, avaliação no teste e registro no MLflow."""
    import mlflow
    import mlflow.sklearn

    pasta = Path(pasta)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experimento)

    ajuste, validacao, teste = (pd.read_parquet(pasta / f"{n}.parquet") for n in ("ajuste", "validacao", "teste"))
    X_aj, y_aj = ajuste[NUM_COLS + CAT_COLS], ajuste["is_fraud"].to_numpy()
    X_val, y_val = validacao[NUM_COLS + CAT_COLS], validacao["is_fraud"].to_numpy()
    X_tst, y_tst = teste[NUM_COLS + CAT_COLS], teste["is_fraud"].to_numpy()
    codificador = joblib.load(pasta / "codificador.joblib")

    with mlflow.start_run(run_name=f"rf-{datetime.now():%Y%m%d-%H%M}") as run:
        mlflow.log_params({"n_iter": n_iter, "fp_custo": fp_custo, "semente": semente, "algoritmo": "RandomForest",
                           "linhas_ajuste": len(ajuste), "linhas_validacao": len(validacao), "linhas_teste": len(teste)})
        if metadados:
            mlflow.set_tags({k: str(v) for k, v in metadados.items()})

        # busca aleatória, nota = PR-AUC na validação (nunca no teste)
        tentativas = []
        for i, params in enumerate(ParameterSampler(ESPACO_RF, n_iter=n_iter, random_state=semente)):
            inicio = time.time()
            with mlflow.start_run(run_name=f"tentativa-{i}", nested=True):
                candidato = RandomForestClassifier(n_jobs=-1, random_state=semente, **params).fit(X_aj, y_aj)
                pr_auc = float(average_precision_score(y_val, candidato.predict_proba(X_val)[:, 1]))
                mlflow.log_params({k: str(v) for k, v in params.items()})
                mlflow.log_metrics({"pr_auc_validacao": pr_auc, "tempo_s": round(time.time() - inicio, 1)})
            tentativas.append((pr_auc, params))
        pr_auc_val, melhores_params = max(tentativas, key=lambda t: t[0])
        mlflow.log_params({f"melhor_{k}": str(v) for k, v in melhores_params.items()})

        modelo = RandomForestClassifier(n_jobs=-1, random_state=semente, **melhores_params).fit(X_aj, y_aj)
        proba_val = modelo.predict_proba(X_val)[:, 1]
        limiar, custo_val = escolher_limiar(proba_val, y_val, X_val["amt"].to_numpy(), fp_custo)

        # avaliação no teste, que não participou de nenhuma escolha
        proba_tst = modelo.predict_proba(X_tst)[:, 1]
        previsto = proba_tst >= limiar
        valores = X_tst["amt"].to_numpy()
        vp, fp = int((previsto & (y_tst == 1)).sum()), int((previsto & (y_tst == 0)).sum())
        fn = int((~previsto & (y_tst == 1)).sum())
        custo_tst = custo_total(y_tst, previsto, valores, fp_custo)
        custo_nada = float(valores[y_tst == 1].sum())
        metricas = {
            "pr_auc_validacao": pr_auc_val, "custo_validacao_us": custo_val, "limiar": limiar,
            "teste_pr_auc": float(average_precision_score(y_tst, proba_tst)),
            "teste_roc_auc": float(roc_auc_score(y_tst, proba_tst)),
            "teste_pr_auc_valor_sozinho": float(average_precision_score(y_tst, valores)),
            "teste_fraudes_detectadas": vp, "teste_fraudes_perdidas": fn, "teste_falsos_alarmes": fp,
            "teste_recall": vp / max(vp + fn, 1), "teste_precisao": vp / max(vp + fp, 1),
            "teste_custo_total_us": custo_tst, "teste_custo_sem_modelo_us": custo_nada,
            "teste_reducao_custo": 1 - custo_tst / custo_nada if custo_nada else float("nan"),
        }
        mlflow.log_metrics(metricas)

        # pacote no mesmo formato que o app Streamlit carrega (src/modelo.py)
        pacote = {"modelo": modelo, "nome": "Random Forest", "limiar": limiar, "custo_falso_alarme": fp_custo,
                  "codificador": codificador, "num_cols": NUM_COLS, "cat_cols": CAT_COLS}
        caminho_pacote = pasta / "modelo_final.joblib"
        joblib.dump(pacote, caminho_pacote)
        mlflow.log_artifact(str(caminho_pacote), "pacote")
        # cloudpickle: o formato padrão (skops) recusa as árvores do Random Forest como tipo não confiável
        info = mlflow.sklearn.log_model(modelo, name="model", input_example=X_val.head(5),
                                        serialization_format="cloudpickle")

    return {"run_id": run.info.run_id, "model_uri": info.model_uri, "pacote": str(caminho_pacote),
            "metricas": metricas, "melhores_params": {k: str(v) for k, v in melhores_params.items()}}


# ------------------------------------------------------------------ 5. promoção
def promover(resultado: dict, tracking_uri: str, pasta_producao: Path, so_se_melhor: bool = True) -> dict:
    """Registra o modelo no MLflow e, se vencer o campeão atual (menor custo no teste), o promove e o publica para o app."""
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    mlflow.set_tracking_uri(tracking_uri)
    cliente = MlflowClient()
    versao = mlflow.register_model(resultado["model_uri"], NOME_MODELO)
    custo_novo = resultado["metricas"]["teste_custo_total_us"]

    custo_campeao, versao_campea = None, None
    try:
        campea = cliente.get_model_version_by_alias(NOME_MODELO, "champion")
        versao_campea = int(campea.version)
        custo_campeao = cliente.get_run(campea.run_id).data.metrics.get("teste_custo_total_us")
    except MlflowException:
        pass  # ainda não há campeão

    promover_agora = (not so_se_melhor) or custo_campeao is None or custo_novo < custo_campeao
    if promover_agora:
        cliente.set_registered_model_alias(NOME_MODELO, "champion", versao.version)
        destino = Path(pasta_producao)
        destino.mkdir(parents=True, exist_ok=True)
        temporario = destino / "modelo_final.joblib.tmp"
        temporario.write_bytes(Path(resultado["pacote"]).read_bytes())
        temporario.replace(destino / "modelo_final.joblib")  # troca atômica: o app nunca lê um arquivo pela metade
    cliente.set_model_version_tag(NOME_MODELO, versao.version, "promovido", str(promover_agora).lower())
    return {"versao": int(versao.version), "promovido": promover_agora, "custo_novo": custo_novo,
            "custo_campeao": custo_campeao, "versao_campea_anterior": versao_campea}


def tracking_uri_padrao() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
