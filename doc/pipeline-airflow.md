# Pipeline no Airflow

A DAG `fraude_cartao_pipeline` (arquivo `airflow/dags/fraude_pipeline.py`) treina o Random Forest, registra tudo no
MLflow e promove o modelo se ele superar o campeão atual. Ela é disparada **manualmente**
(`schedule=None`), porque o pipeline completo usa a base de ~10 GB. A lógica de cada passo está em
`src/pipeline_treino.py`; a DAG só orquestra.

## Tarefas

```
validar_base → converter_parquet → preparar_amostra → dividir_dados → treinar_avaliar → promover
```

| Tarefa | O que faz | Saída (em `retreino/airflow/`) |
|---|---|---|
| `validar_base` | Confere se `credit_card_fraud.csv` existe e tem as colunas usadas | (nenhuma) |
| `converter_parquet` | CSV para Parquet só com as colunas do modelo. **Pula** se o Parquet já está mais novo que o CSV | `credit_card_fraud_full.parquet` |
| `preparar_amostra` | Antes de 24/12/2020, 1 transação a cada N (hash), ordenada no tempo | `execucao/amostra.parquet` |
| `dividir_dados` | Teste = últimos 20%; validação = últimos 20% do treino; subamostragem 1:10 no ajuste; ajusta o codificador | `execucao/ajuste.parquet`, `validacao.parquet`, `teste.parquet`, `codificador.joblib` |
| `treinar_avaliar` | Busca aleatória do Random Forest, escolha do limiar por custo, avaliação no teste, registro no MLflow | `execucao/modelo_final.joblib` |
| `promover` | Registra uma versão no MLflow e, se vencer o campeão, define o alias `champion` e publica o pacote | `mlflow/producao/modelo_final.joblib` |

Os passos passam entre si apenas dicionários pequenos pelo XCom (caminhos e contagens). Os dados ficam em arquivos.

## Parâmetros (Trigger DAG w/ config)

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `n_iter` | 8 | Combinações testadas na busca aleatória (como no notebook) |
| `fp_custo` | 10.0 | Custo (US$) de revisar um falso alarme; entra na escolha do limiar e no custo do teste |
| `amostra_um_em` | 5 | Usa 1 a cada N transações (5 = 20% da base) |
| `so_se_melhor` | true | Só promove se o custo no teste for menor que o do campeão |
| `refazer_parquet` | false | Reconverte o CSV mesmo com Parquet atualizado |

## Regra de promoção

O novo modelo vira `champion` quando:

- ainda não existe campeão, **ou**
- `so_se_melhor` está desligado, **ou**
- o **custo total no teste** do novo modelo é menor que o do campeão atual.

O custo é "valor das fraudes não detectadas + `fp_custo` × falsos alarmes". Só é justo comparar custos obtidos no
mesmo conjunto de teste e com o mesmo `fp_custo`, então mantenha `amostra_um_em` e `fp_custo` ao comparar execuções.
Quando promove, o pacote é copiado para `mlflow/producao/modelo_final.joblib` por troca atômica, e o app passa a
usá-lo sem reiniciar.

## Recursos

O notebook de modelagem completo, na base inteira, levou ~25 min num notebook com 12 núcleos e 16 GB de RAM.
A DAG treina só o Random Forest (sem GPU), então deve ser mais rápida. A conversão do CSV de ~10 GB para Parquet
acontece uma vez e é reaproveitada. Cada tarefa pesada tem limite de 4 horas.

## Rodar o pipeline sem o Airflow

Os passos funcionam em qualquer Python com as dependências de `docker/requirements-airflow.txt`:

```python
import sys; sys.path.insert(0, "src")
from pathlib import Path
import pipeline_treino as pt

pt.validar_base(Path("input/credit_card_fraud.csv"))
pt.converter_parquet(Path("input/credit_card_fraud.csv"), Path("retreino/airflow/full.parquet"))
pt.preparar_amostra(Path("retreino/airflow/full.parquet"), Path("retreino/airflow/execucao"))
pt.dividir_dados(Path("retreino/airflow/execucao"))
r = pt.treinar_avaliar(Path("retreino/airflow/execucao"), "http://localhost:5000")
pt.promover(r, "http://localhost:5000", Path("mlflow/producao"))
```
