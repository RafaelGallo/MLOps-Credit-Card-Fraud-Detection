"""Mapas de bolhas dos alertas por local do cliente, adaptados do notebook de modelagem (células 108 a 110).

Mesmo visual do notebook: Scattergeo nos EUA, destaque em laranja e apoio em cinza.
Não usa tiles externos nem token. Sem dependência do Streamlit.

Dois modos:
- mapa_score: todos os locais, coloridos pelo score de risco (0 a 1);
- mapa_resultado: só os alertas, fraude detectada (laranja) contra falso alarme (cinza). Exige `is_fraud`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

LAT, LON = "lat", "long"  # local do cliente, como no notebook (o do estabelecimento muda a cada compra)
MAX_LOCAIS = 20_000  # acima disso o navegador fica pesado; ficam os locais de maior score
TAMANHO_MAX = 40

COR_SECUNDARIA = "#94A3B8"  # apoio neutro, igual ao notebook
COR_DESTAQUE = "#E69F00"  # destaque, igual ao notebook
RAMPA_SCORE = [(0.0, "#FDEBC8"), (0.5, COR_DESTAQUE), (1.0, "#7A4B00")]  # um hue (âmbar), do claro ao escuro


def _virgula(valor: float, casas: int = 2) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def _tamanho(n_alertas: pd.Series) -> np.ndarray:
    """Tamanho da bolha: 4 + 3 por alerta, como no notebook, com teto para não cobrir o mapa."""
    return np.minimum(4 + 3 * n_alertas.to_numpy(dtype=float), TAMANHO_MAX)


def por_local(pontos: pd.DataFrame, limiar: float) -> pd.DataFrame:
    """Uma linha por local do cliente: transações, alertas e maior score. Fica com os MAX_LOCAIS mais arriscados."""
    locais = (
        pontos.assign(_alerta=pontos["score_fraude"] >= limiar)
        .groupby([LAT, LON], as_index=False)
        .agg(n=("score_fraude", "size"), n_alertas=("_alerta", "sum"), score_max=("score_fraude", "max"))
    )
    return locais.nlargest(MAX_LOCAIS, "score_max")


def escopo(pontos: pd.DataFrame) -> str:
    """'usa' (projeção com Alasca e Havaí) quando quase tudo está nos EUA; senão o mundo todo."""
    if pontos.empty:
        return "usa"
    dentro = pontos[LAT].between(18, 72) & pontos[LON].between(-180, -66)
    return "usa" if dentro.mean() >= 0.95 else "world"


def _barra_de_cores(limiar: float) -> dict:
    marcas = sorted({0.0, 0.25, 0.5, 0.75, 1.0, round(limiar, 2)})
    textos = [_virgula(m) + (" limiar" if m == round(limiar, 2) else "") for m in marcas]
    return dict(title=dict(text="Score máximo no local"), tickvals=marcas, ticktext=textos,
                len=0.8, thickness=14, outlinewidth=0)


def _base(fig: go.Figure, regiao: str, escuro: bool, altura: int) -> go.Figure:
    fig.update_geos(
        scope=regiao, showland=True, landcolor="#2A303C" if escuro else "#F1F5F9",
        showsubunits=True, subunitcolor="#0E1117" if escuro else "white",
        showcountries=True, countrycolor="#0E1117" if escuro else "white",
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(height=altura, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def mapa_score(locais: pd.DataFrame, limiar: float, regiao: str = "usa", escuro: bool = False, altura: int = 560) -> go.Figure:
    """Cada local colorido pelo maior score (0 a 1). Locais com alerta ficam maiores e por cima."""
    locais = locais.sort_values("score_max")  # os maiores scores são desenhados por último
    com_alerta = locais["n_alertas"] > 0
    contorno = "#0E1117" if not escuro else "#F0EFEC"
    fig = go.Figure()
    barra_ja_usada = False
    for parte, opacidade, borda in ((locais[~com_alerta], 0.6, 0.0), (locais[com_alerta], 0.9, 0.5)):
        if parte.empty:
            continue
        fig.add_trace(go.Scattergeo(
            lat=parte[LAT], lon=parte[LON], mode="markers",
            marker=dict(
                size=_tamanho(parte["n_alertas"]), opacity=opacidade, color=parte["score_max"],
                colorscale=RAMPA_SCORE, cmin=0, cmax=1, line=dict(width=borda, color=contorno),
                showscale=not barra_ja_usada, colorbar=_barra_de_cores(limiar),
            ),
            customdata=np.column_stack([parte["n"], parte["n_alertas"], parte["score_max"]]),
            hovertemplate="%{customdata[0]:,.0f} transações · %{customdata[1]:,.0f} alertas<br>"
                          "score máximo %{customdata[2]:.3f}<extra></extra>",
        ))
        barra_ja_usada = True
    fig.update_layout(showlegend=False)
    return _base(fig, regiao, escuro, altura)


def mapa_resultado(pontos: pd.DataFrame, limiar: float, regiao: str = "usa", escuro: bool = False, altura: int = 560) -> go.Figure:
    """Só os alertas: fraudes detectadas (laranja, por cima) e falsos alarmes (cinza, ao fundo). Exige `is_fraud` 0/1."""
    alertas = pontos[pontos["score_fraude"] >= limiar]
    detectadas = alertas[alertas["is_fraud"] == 1].groupby([LAT, LON]).size().reset_index(name="n")
    falsos = alertas[alertas["is_fraud"] == 0].groupby([LAT, LON]).size().reset_index(name="n")
    detectadas, falsos = detectadas.nlargest(MAX_LOCAIS, "n"), falsos.nlargest(MAX_LOCAIS, "n")

    fig = go.Figure()
    # falsos alarmes primeiro para ficarem atrás; a bolha cresce com o número de falsos alarmes do local
    fig.add_trace(go.Scattergeo(
        lat=falsos[LAT], lon=falsos[LON], mode="markers", name="Falso alarme",
        marker=dict(size=_tamanho(falsos["n"]), color=COR_SECUNDARIA, opacity=0.5,
                    line=dict(width=0.5, color="white")),
        text=falsos["n"].astype(str) + " falso(s) alarme(s)", hoverinfo="text",
    ))
    # fraudes detectadas por último para ficarem por cima
    fig.add_trace(go.Scattergeo(
        lat=detectadas[LAT], lon=detectadas[LON], mode="markers", name="Fraude detectada",
        marker=dict(size=_tamanho(detectadas["n"]), color=COR_DESTAQUE, opacity=0.85,
                    line=dict(width=0.5, color="black")),
        text=detectadas["n"].astype(str) + " fraude(s)", hoverinfo="text",
    ))
    fig = _base(fig, regiao, escuro, altura)
    fig.update_layout(showlegend=True, margin=dict(l=0, r=0, t=50, b=0),
                      title="Fraudes previstas pelo modelo por local do cliente")
    return fig
