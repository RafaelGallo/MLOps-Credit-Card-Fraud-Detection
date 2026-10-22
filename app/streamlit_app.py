"""App Streamlit: detecção de fraude em cartão de crédito com Random Forest.

Executar localmente (na raiz do repositório):  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# o Streamlit só põe a pasta do script (app/) no sys.path; o módulo do modelo fica em src/
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import mapa as mp  # noqa: E402
import modelo as ml  # noqa: E402

st.set_page_config(page_title="Detecção de fraude em cartão", page_icon="💳", layout="wide")

MAX_LINHAS = 300_000
COLUNAS_PT = {
    "amt": "Valor (US$)", "city_pop": "População da cidade", "lat": "Latitude do titular",
    "long": "Longitude do titular", "merch_lat": "Latitude do estabelecimento",
    "merch_long": "Longitude do estabelecimento", "category": "Categoria", "gender": "Gênero", "state": "Estado",
}
EXEMPLOS = {
    "Compra online de valor alto (NY)": dict(
        amt=950.0, city_pop=40000, lat=40.7, long=-73.8, merch_lat=40.9, merch_long=-73.2,
        category="shopping_net", gender="M", state="NY"),
    "Compra de supermercado (CA)": dict(
        amt=25.0, city_pop=250000, lat=34.1, long=-117.6, merch_lat=34.2, merch_long=-117.3,
        category="grocery_pos", gender="F", state="CA"),
}

# Cores: um hue (azul) para a série, cinza para o contexto, vermelho de status só no limiar de alerta.
AZUL_CLARO, AZUL_ESCURO = "#2a78d6", "#3987e5"
CINZA, CRITICO = "#898781", "#d03b3b"
TRILHO, GRADE = "rgba(137,135,129,0.25)", "rgba(137,135,129,0.18)"


# ------------------------------------------------------------------ utilidades
def _versao_streamlit() -> tuple[int, int]:
    partes = st.__version__.split(".")
    return int(partes[0]), int(partes[1])


# versões novas trocam use_container_width por width="stretch"; as antigas só conhecem o primeiro
LARGURA = {"width": "stretch"} if _versao_streamlit() >= (1, 52) else {"use_container_width": True}
CONFIG_GRAFICO = {"displayModeBar": False}  # a barra de ferramentas do Plotly cobria o rótulo do limiar


def tema_escuro() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # versões sem st.context.theme
        return False


def azul() -> str:
    return AZUL_ESCURO if tema_escuro() else AZUL_CLARO


def md_seguro(texto: str) -> str:
    """Texto vindo do CSV não pode virar Markdown (links, imagens) nas mensagens."""
    return "".join("\\" + c if c in "\\`*_{}[]()#+-.!|<>~$" else c for c in str(texto))


def inteiro(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", ".")


def dolar(v: float) -> str:
    return "US$ " + inteiro(v)


def pct(x: float, casas: int = 1) -> str:
    return "–" if x != x else f"{x * 100:.{casas}f}%".replace(".", ",")


def decimal(x: float, casas: int = 3) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def estilizar(fig: go.Figure, altura: int, margem_direita: int = 10) -> go.Figure:
    """Grade e eixos discretos, fundo transparente (herda o tema claro/escuro do app)."""
    fig.update_layout(
        height=altura, margin=dict(l=0, r=margem_direita, t=28, b=0), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor=GRADE, zeroline=False, linecolor=GRADE)
    fig.update_yaxes(gridcolor=GRADE, zeroline=False, linecolor=GRADE)
    return fig


def tabela_do_grafico(df: pd.DataFrame, chave: str) -> None:
    """Todo gráfico tem uma tabela equivalente: o tooltip nunca é a única forma de ler um valor."""
    if st.checkbox("Ver dados do gráfico", key=chave):
        st.dataframe(df, hide_index=True, **LARGURA)


@st.cache_resource(show_spinner="Carregando o modelo...")
def _carregar(caminho: str, modificado_em: float) -> dict:
    """O horário de modificação entra na chave do cache: um modelo novo da DAG é recarregado sozinho."""
    return ml.carregar_pacote(Path(caminho))


def carregar_pacote() -> dict:
    caminho = ml.caminho_modelo()
    return _carregar(str(caminho), caminho.stat().st_mtime)


@st.cache_data(show_spinner=False, max_entries=2)
def sortear_base(n: int) -> bytes:
    """Amostra da base local em CSV (só colunas do modelo + is_fraud); o resultado fica em cache."""
    return ml.amostrar_base(n).to_csv(index=False).encode("utf-8")


class ArquivoInvalido(ValueError):
    """Problema do arquivo enviado que o usuário consegue corrigir."""


@st.cache_data(show_spinner="Avaliando as transações...", max_entries=2, ttl=3600)
def avaliar_arquivo(conteudo: bytes) -> tuple[pd.DataFrame, dict]:
    """Lê o CSV, descarta linhas inválidas e acrescenta o score de cada transação."""
    pacote = carregar_pacote()
    try:
        df = pd.read_csv(io.BytesIO(conteudo), nrows=MAX_LINHAS + 1)  # não lê o arquivo todo se passar do limite
    except Exception as erro:
        raise ArquivoInvalido("Não consegui ler o arquivo como CSV (separador vírgula, UTF-8).") from erro
    if len(df.columns) == 1 and any(sep in str(df.columns[0]) for sep in (";", "\t")):
        raise ArquivoInvalido("O arquivo parece usar ponto e vírgula (ou tab) como separador. "
                              "Salve como CSV com separador vírgula e ponto decimal.")
    if len(df) > MAX_LINHAS:
        raise ArquivoInvalido(f"O arquivo tem mais de {inteiro(MAX_LINHAS)} linhas, que é o limite.")
    validas, relatorio = ml.validar_lote(df, pacote)
    if validas is None:
        raise ArquivoInvalido("Faltam colunas obrigatórias: " + ", ".join(relatorio["colunas_ausentes"]) + ".")
    if validas.empty:
        detalhe = ", ".join(f"{COLUNAS_PT.get(c, c)}: {inteiro(n)}" for c, n in relatorio["motivos"].items())
        raise ArquivoInvalido(f"Nenhuma linha válida para avaliar. Valores ausentes ou fora da faixa por coluna: {detalhe}.")
    validas = validas.assign(score_fraude=ml.pontuar(validas, pacote).round(5))
    return validas, relatorio


# ------------------------------------------------------------------ gráficos
def medidor(score: float, limiar: float) -> go.Figure:
    """Barra de 0 a 1 com o score preenchido e o limiar marcado."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[1], y=[""], orientation="h", marker_color=TRILHO, width=0.5, hoverinfo="skip"))
    fig.add_trace(go.Bar(
        x=[score], y=[""], orientation="h", marker_color=azul(), width=0.5,
        hovertemplate="Score %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=limiar, line_color=CRITICO, line_width=3, annotation_text=f"limiar {decimal(limiar, 2)}",
                  annotation_position="top", annotation_font_color=CRITICO)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1], showgrid=False)
    fig.update_yaxes(visible=False)
    return estilizar(fig, 110)


def curva_sensibilidade(valores: np.ndarray, scores: np.ndarray, atual: tuple[float, float], limiar: float) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=valores, y=scores, mode="lines", line=dict(color=azul(), width=2),
        hovertemplate="US$ %{x:,.2f}<br>score %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=limiar, line_color=CRITICO, line_width=2, annotation_text=f"limiar {decimal(limiar, 2)}",
                  annotation_position="top left", annotation_font_color=CRITICO)
    fig.add_trace(go.Scatter(
        x=[atual[0]], y=[atual[1]], mode="markers", marker=dict(color=azul(), size=11, line=dict(color="white", width=2)),
        hovertemplate="Esta transação<br>US$ %{x:,.2f}<br>score %{y:.3f}<extra></extra>",
    ))
    fig.update_xaxes(type="log", title_text="Valor da compra (US$, escala log)")
    fig.update_yaxes(range=[0, 1.02], title_text="Score de risco")
    return estilizar(fig, 300)


def faixas_de_score(score: np.ndarray) -> pd.DataFrame:
    arestas = np.linspace(0, 1, 21)
    contagem, _ = np.histogram(score, bins=arestas)
    rotulos = [f"{decimal(a, 2)}–{decimal(b, 2)}" for a, b in zip(arestas[:-1], arestas[1:])]
    return pd.DataFrame({"inicio": arestas[:-1], "Faixa de score": rotulos, "Transações": contagem})


def histograma(faixas: pd.DataFrame, limiar: float) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=faixas["inicio"] + 0.025, y=faixas["Transações"], width=0.044, marker_color=azul(),
        customdata=faixas["Faixa de score"], hovertemplate="Score %{customdata}<br>%{y:,} transações<extra></extra>",
    ))
    fig.add_vline(x=limiar, line_color=CRITICO, line_width=2, annotation_text=f"limiar {decimal(limiar, 2)}",
                  annotation_position="top", annotation_font_color=CRITICO)
    fig.update_xaxes(range=[0, 1], title_text="Score de risco")
    fig.update_yaxes(type="log", title_text="Transações (escala log)")
    return estilizar(fig, 320)


def comparar_cenarios(cenarios: pd.DataFrame) -> go.Figure:
    """Ênfase: o cenário escolhido em azul, os demais em cinza."""
    cores = [azul() if destaque else CINZA for destaque in cenarios["destaque"]]
    fig = go.Figure(go.Bar(
        y=cenarios["Cenário"], x=cenarios["Custo (US$)"], orientation="h", marker_color=cores, width=0.55,
        text=[dolar(v) for v in cenarios["Custo (US$)"]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>US$ %{x:,.0f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, cenarios["Custo (US$)"].max() * 1.25], title_text="Custo no conjunto de teste (US$)")
    return estilizar(fig, 260, margem_direita=90)


# ------------------------------------------------------------------ app
pacote = carregar_pacote()
COLUNAS = ml.colunas_modelo(pacote)
CATEGORIAS = ml.categorias(pacote)

with st.sidebar:
    st.title("💳 Fraude em cartão")
    st.caption("Random Forest para transações de cartão de crédito. Dados de jan/2019 a dez/2020.")
    st.subheader("Regra de decisão")
    limiar = st.slider(
        "Limiar de alerta", min_value=0.05, max_value=0.99,
        value=round(round(pacote["limiar"] / 0.005) * 0.005, 3), step=0.005,  # o limiar do modelo, arredondado ao passo
        help=f"Transações com score igual ou acima do limiar viram alerta. {decimal(pacote['limiar'], 3)} foi o "
             "limiar de menor custo na validação, assumindo US$ 10 por falso alarme.",
    )
    custo_fp = st.number_input(
        "Custo de revisar um falso alarme (US$)", min_value=0.0, value=float(pacote["custo_falso_alarme"]), step=1.0,
        help="Usado nas estimativas de custo da análise em lote. Mudar o valor não recalcula o limiar ideal.",
    )
    st.divider()
    st.warning("Projeto de portfólio com **dados sintéticos**. Os resultados não representam fraude real.", icon="⚠️")

aba_unica, aba_lote, aba_sobre = st.tabs(["Transação única", "Análise em lote (CSV)", "Sobre o modelo"])

# ---- transação única
with aba_unica:
    for coluna, valor in next(iter(EXEMPLOS.values())).items():
        st.session_state.setdefault(f"tx_{coluna}", valor)

    def aplicar_exemplo() -> None:
        escolhido = st.session_state.get("exemplo")
        if escolhido:
            for coluna, valor in EXEMPLOS[escolhido].items():
                st.session_state[f"tx_{coluna}"] = valor

    esquerda, direita = st.columns([1.1, 1], gap="large")
    with esquerda:
        st.selectbox("Carregar um exemplo", list(EXEMPLOS), index=None, placeholder="Escolher um exemplo...",
                     key="exemplo", on_change=aplicar_exemplo)
        st.markdown("**Compra**")
        c1, c2 = st.columns(2)
        c1.number_input("Valor da compra (US$)", min_value=0.0, max_value=1_000_000.0, step=1.0, format="%.2f", key="tx_amt")
        c2.selectbox("Categoria do estabelecimento", CATEGORIAS["category"], key="tx_category")
        st.markdown("**Titular do cartão**")
        c1, c2 = st.columns(2)
        c1.selectbox("Gênero", CATEGORIAS["gender"], key="tx_gender")
        c2.selectbox("Estado", CATEGORIAS["state"], key="tx_state")
        c1, c2 = st.columns(2)
        c1.number_input("População da cidade", min_value=0, step=1000, key="tx_city_pop")
        c1, c2 = st.columns(2)
        c1.number_input("Latitude", min_value=-90.0, max_value=90.0, step=0.01, format="%.4f", key="tx_lat")
        c2.number_input("Longitude", min_value=-180.0, max_value=180.0, step=0.01, format="%.4f", key="tx_long")
        st.markdown("**Estabelecimento**")
        c1, c2 = st.columns(2)
        c1.number_input("Latitude ", min_value=-90.0, max_value=90.0, step=0.01, format="%.4f", key="tx_merch_lat")
        c2.number_input("Longitude ", min_value=-180.0, max_value=180.0, step=0.01, format="%.4f", key="tx_merch_long")

    transacao = {coluna: st.session_state[f"tx_{coluna}"] for coluna in COLUNAS}
    score = float(ml.pontuar(pd.DataFrame([transacao]), pacote)[0])
    alerta = score >= limiar

    with direita:
        st.markdown("**Score de risco**")
        st.markdown(
            f"<div style='font-size:3.4rem;font-weight:600;line-height:1.1'>{decimal(score)}</div>",
            unsafe_allow_html=True,
        )
        if alerta:
            st.error("🚨 **ALERTA de fraude.** Enviar para revisão: o score está igual ou acima do limiar.")
        else:
            st.success("✅ **Liberar.** O score está abaixo do limiar de alerta.")
        st.plotly_chart(medidor(score, limiar), config=CONFIG_GRAFICO, **LARGURA)
        km = ml.distancia_km(transacao["lat"], transacao["long"], transacao["merch_lat"], transacao["merch_long"])
        st.caption(
            f"Distância entre titular e estabelecimento: {inteiro(km)} km (informativo; não é uma variável do modelo). "
            "O score ordena o risco, mas **não é uma probabilidade calibrada**: o treino usou undersampling das "
            "transações legítimas."
        )
        if transacao["amt"] > 1250:
            st.caption("⚠️ Nos dados de treino não há fraude acima de ~US$ 1.250, então valores maiores "
                       "tendem a ser liberados pelo modelo.")

    with st.expander("Como o valor da compra muda o score?"):
        valores = np.round(np.geomspace(1, 5000, 80), 2)
        scores = ml.curva_valor(transacao, pacote, valores)
        st.plotly_chart(curva_sensibilidade(valores, scores, (transacao["amt"], score), limiar),
                        config=CONFIG_GRAFICO, **LARGURA)
        st.caption("Só o valor varia; os demais campos ficam como preenchidos acima. "
                   "No estudo, o valor da compra concentra quase todo o sinal do modelo.")
        tabela_do_grafico(pd.DataFrame({"Valor (US$)": valores, "Score": scores.round(4)}), "tab_curva")

# ---- análise em lote (função para poder usar return sem interromper as outras abas)
def renderizar_lote() -> None:
    st.markdown(
        "Envie um CSV com as colunas abaixo. Linhas inválidas são descartadas e listadas no resumo. "
        f"Limite: {inteiro(MAX_LINHAS)} linhas. Se houver uma coluna `is_fraud` (0/1), o app também mede a "
        "qualidade do modelo no seu arquivo."
    )
    st.code(", ".join(COLUNAS), language=None)
    arquivo = st.file_uploader("Arquivo CSV", type=["csv"])
    usar_exemplo = st.checkbox("Usar o arquivo de exemplo (5.000 transações)", value=False)
    if ml.CAMINHO_EXEMPLO.exists():
        st.download_button("Baixar o arquivo de exemplo", ml.CAMINHO_EXEMPLO.read_bytes(),
                           file_name="transacoes_exemplo.csv", mime="text/csv")

    if ml.CAMINHO_BASE.exists():  # só na máquina local: a base completa (~10 GB) não vai para o deploy
        st.markdown("**Base de dados local** (`input/credit_card_fraud.csv`)")
        b1, b2 = st.columns([1, 1])
        n_base = b1.number_input("Transações a sortear", min_value=1_000, max_value=MAX_LINHAS, value=50_000, step=10_000)
        b2.write("")
        if b2.button("Sortear da base (leva ~1 min)"):
            with st.spinner("Lendo a base completa em blocos..."):
                st.session_state["amostra_base"] = sortear_base(int(n_base))
        if "amostra_base" in st.session_state:
            st.caption(f"Usando a amostra sorteada da base ({inteiro(len(st.session_state['amostra_base'].splitlines()) - 1)} "
                       "transações, com `is_fraud`). Enviar um arquivo substitui a amostra.")

    conteudo = None
    if arquivo is not None:
        conteudo = arquivo.getvalue()
    elif "amostra_base" in st.session_state:
        conteudo = st.session_state["amostra_base"]
    elif usar_exemplo and ml.CAMINHO_EXEMPLO.exists():
        conteudo = ml.CAMINHO_EXEMPLO.read_bytes()

    if conteudo is None:
        st.info("Envie um arquivo ou marque a opção de exemplo para ver os resultados.")
    else:
        try:
            resultado, relatorio = avaliar_arquivo(conteudo)
        except ArquivoInvalido as erro:
            st.error(str(erro))
            return
        except Exception:
            st.error("Não foi possível avaliar o arquivo. Confira se ele segue o formato descrito acima.")
            return

        if relatorio["invalidas"]:
            detalhe = ", ".join(f"{COLUNAS_PT.get(c, c)}: {inteiro(n)}" for c, n in relatorio["motivos"].items())
            st.warning(f"{inteiro(relatorio['invalidas'])} de {inteiro(relatorio['linhas'])} linhas foram descartadas "
                       f"(valor ausente ou fora da faixa válida). Por coluna: {detalhe}.")
        for coluna, novas in relatorio["desconhecidas"].items():
            st.warning(f"Valores de **{COLUNAS_PT.get(coluna, coluna)}** que o modelo não conhece "
                       f"({', '.join(md_seguro(v) for v in novas[:8])}{'…' if len(novas) > 8 else ''}) "
                       "foram tratados como categoria desconhecida.")

        alertas = resultado["score_fraude"] >= limiar
        n, n_alertas = len(resultado), int(alertas.sum())
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Transações avaliadas", inteiro(n))
        k2.metric("Alertas", inteiro(n_alertas))
        k3.metric("Taxa de alerta", pct(n_alertas / n, 2))
        k4.metric("Custo de revisão estimado", dolar(n_alertas * custo_fp),
                  help="Alertas × custo por falso alarme. É um teto: parte dos alertas serão fraudes reais.")
        st.caption(f"Valor total das transações em alerta: {dolar(resultado.loc[alertas, 'amt'].sum())}.")

        # mapa adaptado do notebook (células 108 a 110): resultado só existe se o CSV trouxer is_fraud
        rotulo_mapa = pd.to_numeric(resultado["is_fraud"], errors="coerce") if "is_fraud" in resultado.columns else None
        com_rotulo = rotulo_mapa is not None and rotulo_mapa.isin([0, 1]).any()
        st.markdown("**Mapa por local do cliente**")
        modos = ["Resultado: fraude × falso alarme", "Score de risco (0 a 1)"] if com_rotulo else ["Score de risco (0 a 1)"]
        modo = st.radio("Mostrar", modos, horizontal=True, key="modo_mapa")
        regiao = mp.escopo(resultado)
        if modo.startswith("Resultado"):
            base = resultado.assign(is_fraud=rotulo_mapa)[rotulo_mapa.isin([0, 1])]
            fig_mapa = mp.mapa_resultado(base, limiar, regiao, tema_escuro())
            st.caption("Só os alertas (score ≥ limiar): laranja = fraude detectada, cinza = falso alarme. "
                       "O tamanho da bolha é o número de alertas no local.")
        else:
            locais = mp.por_local(resultado, limiar)
            fig_mapa = mp.mapa_score(locais, limiar, regiao, tema_escuro())
            st.caption(f"Cada ponto é um local de cliente, colorido pelo maior score nele ({inteiro(len(locais))} locais). "
                       "Pontos maiores têm alertas; os de score alto ficam por cima.")
        st.plotly_chart(fig_mapa, config=CONFIG_GRAFICO, **LARGURA)

        st.markdown("**Distribuição dos scores**")
        faixas = faixas_de_score(resultado["score_fraude"].to_numpy())
        st.plotly_chart(histograma(faixas, limiar), config=CONFIG_GRAFICO, **LARGURA)
        tabela_do_grafico(faixas[["Faixa de score", "Transações"]], "tab_hist")

        st.markdown("**Transações de maior risco**")
        colunas_topo = [c for c in resultado.columns if c != "score_fraude"]
        topo = resultado.sort_values("score_fraude", ascending=False).head(200)
        st.dataframe(topo[["score_fraude"] + colunas_topo], hide_index=True, **LARGURA)

        saida = resultado.assign(alerta=alertas.astype(int)).sort_values("score_fraude", ascending=False)
        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button("Baixar todas com score", saida.to_csv(index=False).encode("utf-8"),
                           file_name="transacoes_com_score.csv", mime="text/csv")
        d2.download_button("Baixar só os alertas", saida[saida["alerta"] == 1].to_csv(index=False).encode("utf-8"),
                           file_name="alertas_de_fraude.csv", mime="text/csv")

        if "is_fraud" in resultado.columns:
            rotulo = pd.to_numeric(resultado["is_fraud"], errors="coerce")
            validos = rotulo.isin([0, 1])
            st.markdown("**Avaliação com os rótulos do arquivo (`is_fraud`)**")
            if not validos.any() or rotulo[validos].nunique() < 2:
                st.info("Para avaliar, o arquivo precisa ter transações legítimas (0) e fraudes (1).")
            else:
                if not validos.all():
                    st.warning(f"{inteiro((~validos).sum())} linhas com `is_fraud` fora de 0/1 ficaram de fora desta avaliação.")
                base = resultado[validos]
                r = ml.metricas(rotulo[validos], base["score_fraude"], base["amt"], limiar, custo_fp)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Recall (fraudes capturadas)", pct(r["recall"]))
                m2.metric("Precisão dos alertas", pct(r["precisao"]))
                m3.metric("Custo com o modelo", dolar(r["custo"]), delta=f"{pct(r['economia'])} de economia", delta_color="normal")
                m4.metric("Custo sem modelo", dolar(r["custo_sem_modelo"]))
                st.dataframe(
                    pd.DataFrame(
                        {"Previsto: fraude": [r["vp"], r["fp"]], "Previsto: legítima": [r["fn"], r["vn"]]},
                        index=["Real: fraude", "Real: legítima"],
                    ),
                    **LARGURA,
                )
                st.caption("Custo = valor das fraudes que passaram + custo de revisão × falsos alarmes (mesma regra do notebook).")


with aba_lote:
    renderizar_lote()

# ---- sobre o modelo
with aba_sobre:
    st.subheader("O problema")
    st.markdown(
        "Fraude em cartão gera perda direta (estornos e reembolsos) e custo de revisão manual. É um evento raro: "
        "cerca de **0,55%** de 34,6 milhões de transações (jan/2019 a dez/2020), então regras fixas e revisão "
        "manual não escalam. O objetivo é **pontuar cada transação no momento da autorização** para que o time de "
        "fraude revise só as de maior risco."
    )
    st.subheader("Como o modelo foi construído")
    st.markdown(
        "- **Variáveis:** apenas as 9 colunas originais (valor, população da cidade, coordenadas do titular e do "
        "estabelecimento, categoria, gênero e estado). Sem identificadores pessoais e sem variáveis criadas.\n"
        "- **Divisão cronológica:** treino e teste separados no tempo; o teste cobre de 14/08 a 23/12/2020 "
        "(1.358.891 transações, 7.042 fraudes, 0,52%).\n"
        "- **Desbalanceamento:** undersampling das legítimas (1:10) apenas no treino; validação e teste mantêm a taxa real.\n"
        "- **Modelo:** Random Forest, o melhor entre 10 algoritmos comparados (PR-AUC no teste de 0,2835, cerca de "
        "54 vezes a linha de base aleatória de 0,0052). Em seguida os hiperparâmetros foram ajustados (200 árvores, "
        "profundidade 6, mínimo de 20 linhas por folha) e o limiar foi escolhido num conjunto de validação separado "
        "(PR-AUC de 0,3625 nele; não é comparável ao 0,2835 do teste).\n"
        "- **Limiar:** escolhido na validação pelo menor custo, assumindo US$ 10 por falso alarme. Resultado: 0,395."
    )

    st.subheader("Resultado no conjunto de teste")
    cenarios = pd.DataFrame(
        [
            ("Sem modelo", 0, 7042, 0, 3810073, False),
            ("Random Forest original, corte 0,5", 4985, 2057, 120011, 1446207, False),
            ("Ajustado, corte 0,5", 4843, 2199, 102030, 1290984, False),
            ("Ajustado, corte por custo (0,395)", 4997, 2045, 108357, 1298802, True),
        ],
        columns=["Cenário", "Fraudes detectadas", "Fraudes perdidas", "Falsos alarmes", "Custo (US$)", "destaque"],
    )
    st.plotly_chart(comparar_cenarios(cenarios), config=CONFIG_GRAFICO, **LARGURA)
    exibir = cenarios.drop(columns="destaque").copy()
    for coluna in exibir.columns[1:]:
        exibir[coluna] = exibir[coluna].map(inteiro)
    st.dataframe(exibir, hide_index=True, **LARGURA)
    st.caption("Custo = valor das fraudes não detectadas + US$ 10 por falso alarme. Números do notebook de modelagem. "
               "No teste, o corte por custo (escolhido na validação) ficou 0,6% acima do custo do corte 0,5 do modelo "
               "ajustado: a diferença é pequena.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraudes detectadas", "71%", help="4.997 de 7.042 no teste.")
    c2.metric("Precisão dos alertas", "4%", help="Cerca de 22 falsos alarmes para cada fraude detectada.")
    c3.metric("Redução de custo", "66%", help="Em relação a não usar modelo.")

    st.subheader("Limitações")
    st.markdown(
        "- **Os dados são sintéticos** (por exemplo, não há fraude acima de ~US$ 1.250 e há titulares de 16 e 17 anos). "
        "Os números não valem como estimativa do mundo real.\n"
        "- **O valor da compra carrega quase todo o sinal:** um modelo só com `amt` chegou a 0,3366 de PR-AUC na "
        "validação, contra 0,3625 com as 9 colunas.\n"
        "- **US$ 10 por falso alarme é uma suposição.** Limiar e economia dependem dele; use o painel lateral para testar outros valores.\n"
        "- **Uma única divisão dos dados e semente fixa.** O teste foi consultado mais de uma vez durante a comparação dos modelos.\n"
        "- **Padrões de fraude mudam:** o modelo precisa ser reavaliado periodicamente."
    )
    with st.expander("Detalhes técnicos do modelo carregado"):
        parametros = pacote["modelo"].get_params()
        st.json({k: parametros[k] for k in ("n_estimators", "max_depth", "min_samples_leaf", "max_features", "max_samples", "class_weight")}
                | {"limiar": round(pacote["limiar"], 4), "variáveis": COLUNAS})
