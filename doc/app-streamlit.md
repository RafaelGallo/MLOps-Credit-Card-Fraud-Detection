# App Streamlit

Arquivo principal: `app/streamlit_app.py`. A lógica fica em `src/modelo.py` (modelo, validação, métricas) e
`src/mapa.py` (mapas), sem dependência do Streamlit.

## Abas

**Transação única**: preencha os 9 campos (ou escolha um exemplo) e veja o score de risco, a decisão (alerta ou
liberar), o medidor com o limiar e como o valor da compra muda o score. O score **não é uma probabilidade calibrada**.

**Análise em lote (CSV)**: envie um CSV com as 9 colunas (limite de 300.000 linhas e 50 MB). O app descarta
linhas inválidas e lista os motivos, mostra alertas, taxa de alerta e custo de revisão, o mapa, a distribuição dos scores,
as transações de maior risco e permite baixar os resultados. Se o arquivo tiver a coluna `is_fraud` (0/1), mede também
recall, precisão e custo.

- **Mapa por local do cliente**, adaptado do notebook de modelagem. Dois modos: *Score de risco (0 a 1)*, que
  funciona com qualquer CSV, e *Resultado: fraude × falso alarme*, que exige `is_fraud` (laranja = fraude
  detectada, cinza = falso alarme; o tamanho da bolha é o número de alertas no local).
- **Base de dados local**: se `input/credit_card_fraud.csv` existir, aparece um botão que sorteia N transações
  da base (~1 min para 50.000). Só as colunas do modelo e `is_fraud` são lidas. No deploy esse botão não aparece.

**Sobre o modelo**: problema, método, resultados no teste e limitações. Os números dessa aba são **fixos no código**
e correspondem ao modelo de `models/modelo_final.joblib`. Se você promover outro campeão, atualize o texto.

## Painel lateral

- **Limiar de alerta**: começa no limiar do modelo, arredondado a 0,005. Transações com score igual ou acima dele
  viram alerta.
- **Custo de revisar um falso alarme**: usado nas estimativas de custo do lote. Não recalcula o limiar ideal.

## Qual modelo o app usa

1. O arquivo apontado por `FRAUDE_MODELO_PATH`, se existir (no Docker: o campeão publicado pela DAG em
   `mlflow/producao/modelo_final.joblib`);
2. senão, `models/modelo_final.joblib` do repositório.

O app recarrega o modelo quando o horário de modificação do arquivo muda.

## Executar localmente

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Deploy no Streamlit Community Cloud

1. Suba para o GitHub o que o app precisa (o `.gitignore` já bloqueia o CSV de ~10 GB e os modelos pesados):
   `app/`, `src/modelo.py`, `src/mapa.py`, `requirements.txt`, `.streamlit/config.toml`,
   `models/modelo_final.joblib` e `output/transacoes_novas.csv`.
2. Em [share.streamlit.io](https://share.streamlit.io), crie o app com o repositório, a branch `main` e o arquivo
   principal `app/streamlit_app.py`.
3. Em **Advanced settings**, selecione **Python 3.12**. A versão só muda recriando o app.

O Community Cloud não roda Airflow nem MLflow: nele o app usa sempre o modelo do repositório.

## Configuração

`.streamlit/config.toml`: limite de upload de 50 MB, proteção XSRF, sem telemetria e cor de destaque dos temas
claro e escuro. O app segue o tema do sistema.
