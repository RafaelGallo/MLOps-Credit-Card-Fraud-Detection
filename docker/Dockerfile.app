FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY .streamlit .streamlit
COPY app app
COPY src src
# modelo incluído no repositório e CSV de exemplo; o modelo campeão do MLflow, quando existe, vem por volume
COPY models/modelo_final.joblib models/modelo_final.joblib
COPY output/transacoes_novas.csv output/transacoes_novas.csv

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
