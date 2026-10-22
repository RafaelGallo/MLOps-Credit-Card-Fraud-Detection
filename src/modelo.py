"""Lógica do modelo de fraude: carga do pacote, validação, predição e métricas.

Não depende do Streamlit, então pode ser testada e reutilizada fora do app.
O pacote é o `modelo_final.joblib` gerado no notebook de modelagem (Parte 17):
modelo, limiar de custo, codificador das categorias e as listas de colunas.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent  # raiz do repositório (este arquivo fica em src/)
CAMINHO_MODELO = RAIZ / "models" / "modelo_final.joblib"
CAMINHO_EXEMPLO = RAIZ / "output" / "transacoes_novas.csv"

# faixas aceitas por coluna numérica; fora delas a linha é descartada do lote
LIMITES = {
    "amt": (0.0, 1e12),  # teto finito: inf e valores acima do float32 derrubariam o lote inteiro
    "city_pop": (0.0, 1e12),
    "lat": (-90.0, 90.0),
    "long": (-180.0, 180.0),
    "merch_lat": (-90.0, 90.0),
    "merch_long": (-180.0, 180.0),
}


CAMINHO_BASE = RAIZ / "input" / "credit_card_fraud.csv"  # base completa (~10 GB), só existe na máquina local


def amostrar_base(n: int, caminho: Path = CAMINHO_BASE, semente: int = 42, tamanho_bloco: int = 1_000_000) -> pd.DataFrame:
    """Sorteia n transações da base completa, lendo em blocos e só as colunas do modelo + is_fraud.

    As colunas pessoais da base (ssn, cc_num, nome, endereço) nunca são carregadas.
    """
    colunas = ["amt", "city_pop", "lat", "long", "merch_lat", "merch_long", "category", "gender", "state", "is_fraud"]
    tamanho_linha = max(len(linha) for linha in caminho.open("rb").readlines(200_000)) / 1.5  # bytes por linha, aproximado
    total_estimado = caminho.stat().st_size / max(tamanho_linha, 1)
    fracao = min(1.0, 1.5 * n / total_estimado)  # folga para depois cortar exatamente em n
    partes = [
        bloco.sample(frac=fracao, random_state=semente + i)
        for i, bloco in enumerate(pd.read_csv(caminho, usecols=colunas, chunksize=tamanho_bloco))
    ]
    amostra = pd.concat(partes, ignore_index=True)
    if len(amostra) > n:
        amostra = amostra.sample(n=n, random_state=semente)
    return amostra.reset_index(drop=True)


def caminho_modelo() -> Path:
    """Modelo que o app usa: o campeão publicado pela DAG (FRAUDE_MODELO_PATH), se existir; senão o do repositório."""
    publicado = os.environ.get("FRAUDE_MODELO_PATH")
    if publicado and Path(publicado).exists():
        return Path(publicado)
    return CAMINHO_MODELO


def carregar_pacote(caminho: Path | None = None) -> dict:
    """Carrega o pacote salvo no notebook (modelo, limiar, codificador, colunas)."""
    return joblib.load(caminho or caminho_modelo())


def colunas_modelo(pacote: dict) -> list[str]:
    """As 9 colunas originais, na ordem em que o modelo foi treinado."""
    return list(pacote["num_cols"]) + list(pacote["cat_cols"])


def categorias(pacote: dict) -> dict[str, list[str]]:
    """Valores de cada coluna categórica conhecidos pelo codificador."""
    return {
        coluna: [str(v) for v in valores]
        for coluna, valores in zip(pacote["cat_cols"], pacote["codificador"].categories_)
    }


def montar_matriz(df: pd.DataFrame, pacote: dict) -> pd.DataFrame:
    """Mesma codificação do treino: numéricas + categóricas ordinais (desconhecida = -1)."""
    num_cols, cat_cols = pacote["num_cols"], pacote["cat_cols"]
    codificadas = pacote["codificador"].transform(df[cat_cols].astype(str))
    matriz = np.hstack([df[num_cols].to_numpy(dtype=float), codificadas])
    return pd.DataFrame(matriz, columns=num_cols + cat_cols, index=df.index)


def pontuar(df: pd.DataFrame, pacote: dict, tamanho_lote: int = 100_000) -> np.ndarray:
    """Score de fraude (0 a 1) de cada linha, calculado em blocos para limitar a memória."""
    if df.empty:
        return np.empty(0)
    matriz = montar_matriz(df, pacote)
    modelo = pacote["modelo"]
    partes = [
        modelo.predict_proba(matriz.iloc[i : i + tamanho_lote])[:, 1]
        for i in range(0, len(matriz), tamanho_lote)
    ]
    return np.concatenate(partes)


def validar_lote(df: pd.DataFrame, pacote: dict) -> tuple[pd.DataFrame | None, dict]:
    """Converte os tipos e separa as linhas que o modelo consegue avaliar.

    Devolve (linhas_validas, relatorio). Sem as 9 colunas, devolve (None, relatorio).
    Categorias que o modelo não conhece não invalidam a linha: viram -1, como no treino,
    e são listadas no relatório.
    """
    num_cols, cat_cols = pacote["num_cols"], pacote["cat_cols"]
    ausentes = [c for c in colunas_modelo(pacote) if c not in df.columns]
    relatorio = {"linhas": len(df), "colunas_ausentes": ausentes, "invalidas": 0, "motivos": {}, "desconhecidas": {}}
    if ausentes:
        return None, relatorio

    limpo = df.copy()
    invalida = pd.Series(False, index=limpo.index)
    for coluna in num_cols:
        limpo[coluna] = pd.to_numeric(limpo[coluna], errors="coerce")
        minimo, maximo = LIMITES[coluna]
        ruim = ~limpo[coluna].between(minimo, maximo)  # NaN também cai aqui
        if ruim.any():
            relatorio["motivos"][coluna] = int(ruim.sum())
        invalida |= ruim
    for coluna in cat_cols:
        vazia = limpo[coluna].isna() | (limpo[coluna].astype(str).str.strip() == "")
        if vazia.any():
            relatorio["motivos"][coluna] = int(vazia.sum())
        invalida |= vazia
        limpo[coluna] = limpo[coluna].astype(str).str.strip()

    limpo = limpo.loc[~invalida]
    conhecidas = categorias(pacote)
    for coluna in cat_cols:
        novas = sorted(set(limpo[coluna]) - set(conhecidas[coluna]))
        if novas:
            relatorio["desconhecidas"][coluna] = novas
    relatorio["invalidas"] = int(invalida.sum())
    return limpo, relatorio


def metricas(y_real: pd.Series, score: np.ndarray, valores: pd.Series, limiar: float, custo_falso_alarme: float) -> dict:
    """Matriz de confusão, precisão, recall e custo (mesma regra do notebook).

    Custo = valor das fraudes que passaram + custo de revisão de cada falso alarme.
    """
    real = np.asarray(y_real).astype(bool)
    alerta = np.asarray(score) >= limiar
    valores = np.asarray(valores, dtype=float)
    vp = int((alerta & real).sum())
    fp = int((alerta & ~real).sum())
    fn = int((~alerta & real).sum())
    vn = int((~alerta & ~real).sum())
    custo_sem_modelo = float(valores[real].sum())
    custo = float(valores[~alerta & real].sum()) + custo_falso_alarme * fp
    return {
        "vp": vp, "fp": fp, "fn": fn, "vn": vn,
        "precisao": vp / (vp + fp) if vp + fp else float("nan"),
        "recall": vp / (vp + fn) if vp + fn else float("nan"),
        "custo": custo,
        "custo_sem_modelo": custo_sem_modelo,
        "economia": (1 - custo / custo_sem_modelo) if custo_sem_modelo else float("nan"),
    }


def distancia_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distância (haversine) entre o titular e o estabelecimento, só como contexto."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(6371.0 * 2 * np.arcsin(np.sqrt(a)))


def curva_valor(transacao: dict, pacote: dict, valores: np.ndarray) -> np.ndarray:
    """Score da mesma transação variando só o valor da compra (análise de sensibilidade)."""
    base = pd.DataFrame([transacao] * len(valores))
    base["amt"] = np.asarray(valores, dtype=float)
    return pontuar(base, pacote)
