# MLflow

O servidor MLflow (`http://localhost:5000`) guarda os experimentos e o registro de modelos. Ele roda no
contêiner `mlflow`, com o **Postgres** como banco de metadados e um volume Docker para os **artefatos**
(o cliente envia os arquivos por HTTP, com `--serve-artifacts`, então o Airflow não precisa montar o volume).

## O que cada execução registra

Experimento: `fraude-cartao`. Cada execução da DAG cria uma execução (*run*) principal, com uma execução
aninhada por combinação testada na busca.

**Execução principal**

| Tipo | Itens |
|---|---|
| Parâmetros | `n_iter`, `fp_custo`, `semente`, `algoritmo`, linhas de ajuste, validação e teste, `melhor_*` (hiperparâmetros escolhidos) |
| Métricas de validação | `pr_auc_validacao`, `custo_validacao_us`, `limiar` |
| Métricas de teste | `teste_pr_auc`, `teste_roc_auc`, `teste_pr_auc_valor_sozinho`, `teste_fraudes_detectadas`, `teste_fraudes_perdidas`, `teste_falsos_alarmes`, `teste_recall`, `teste_precisao`, `teste_custo_total_us`, `teste_custo_sem_modelo_us`, `teste_reducao_custo` |
| Tags | `airflow_dag_run`, `periodo_teste_inicio` |
| Artefatos | `model/` (modelo scikit-learn), `pacote/modelo_final.joblib` (modelo + limiar + codificador, o formato que o app carrega) |

**Execuções aninhadas** (`tentativa-0`, `tentativa-1`, ...): hiperparâmetros da combinação, `pr_auc_validacao` e `tempo_s`.

`teste_pr_auc_valor_sozinho` é a referência "usar só o valor da compra como score". Um modelo que não a supera não
justifica a complexidade.

## Registro de modelos

- Nome: `fraude_cartao_random_forest`.
- Cada execução da DAG cria uma **versão** nova, com a tag `promovido` (`true` ou `false`).
- O alias **`champion`** aponta para a versão em produção. Ele só muda na tarefa `promover`.
- A versão campeã é carregável por `models:/fraude_cartao_random_forest@champion`.

## Consultas úteis

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")

# comparar execuções pelo custo no teste
runs = mlflow.search_runs(experiment_names=["fraude-cartao"], order_by=["metrics.teste_custo_total_us ASC"])
runs[["run_id", "metrics.teste_custo_total_us", "metrics.teste_pr_auc", "metrics.limiar"]]

# o campeão atual
modelo = mlflow.sklearn.load_model("models:/fraude_cartao_random_forest@champion")
```

Para voltar a uma versão anterior, na interface (`Models` → `fraude_cartao_random_forest`) mova o alias
`champion` para a versão desejada. Isso não republica o arquivo do app: para isso, baixe o artefato
`pacote/modelo_final.joblib` da execução daquela versão e copie para `mlflow/producao/modelo_final.joblib`.

## Formato do modelo

O modelo é salvo em `cloudpickle`, não no formato `skops` (padrão do MLflow 3): o `skops` recusa as árvores do
Random Forest (`sklearn.tree._tree.Tree`) como tipo não confiável. O MLflow avisa nos logs que pickle executa código
ao ser carregado. Por isso, **carregue apenas modelos do seu próprio servidor MLflow**, e o mesmo vale para o
`modelo_final.joblib` do app.

## Segurança

O servidor local não tem autenticação e só é exposto em `localhost`. Não publique a porta 5000 na internet
sem colocar um proxy com autenticação na frente. A opção `--allowed-hosts` no `docker-compose.yml` restringe
os nomes de host aceitos.
