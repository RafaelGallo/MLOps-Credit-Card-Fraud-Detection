# Modelo e dados

## A base

- Arquivo: `input/credit_card_fraud.csv` (~10,7 GB, 34.636.378 linhas, de dezembro de 2018 a dezembro de 2020).
- Taxa de fraude: 0,548%.
- **Dados sintéticos.** Há sinais claros disso (nenhuma fraude acima de ~US$ 1.250, titulares de 16 e 17 anos),
  então os resultados não são estimativas do mundo real.
- A base tem colunas pessoais (`ssn`, `cc_num`, nome, endereço). O pipeline e o app **nunca as leem**: só as 9
  variáveis do modelo, mais `unix_time` e `is_fraud`.
- Os rótulos de fraude só estão completos até 23/12/2020, por isso tudo depois de 24/12/2020 é descartado.

## As 9 variáveis do modelo

| Tipo | Colunas |
|---|---|
| Numéricas | `amt` (valor), `city_pop`, `lat`, `long` (titular), `merch_lat`, `merch_long` (estabelecimento) |
| Categóricas | `category`, `gender`, `state` (codificadas com `OrdinalEncoder`; categoria desconhecida vira `-1`) |

Não há variáveis criadas e nenhum identificador pessoal entra no modelo.

## Método (igual no notebook e no pipeline)

1. **Amostra de 20%**: uma a cada 5 transações, escolhida por hash, o que mantém a taxa real de fraude. Ordenada no tempo.
2. **Divisão cronológica**, sem embaralhar:
   - teste = últimos 20% da amostra;
   - do restante, os últimos 20% viram validação e os primeiros 80% são o ajuste.
3. **Subamostragem 1:10 só no ajuste** (10 legítimas para cada fraude). Validação e teste mantêm a taxa real.
4. **Busca aleatória** de hiperparâmetros do Random Forest (8 combinações), nota = PR-AUC na validação.
5. **Limiar de decisão** pelo menor custo na validação: custo = valor das fraudes que passaram + US$ 10 por falso alarme.
6. **Avaliação no teste**, que não participa de nenhuma escolha.

A métrica principal é a **PR-AUC**, porque a classe é muito rara: prever "não é fraude" para tudo já dá ~99,5% de acurácia.

## Resultados do modelo atual (`models/modelo_final.joblib`)

Vêm do retreino local do notebook de modelagem (22/10/2026): Random Forest com 200 árvores, profundidade 6,
mínimo de 20 linhas por folha, limiar **0,395**. Conjunto de teste: 1.358.891 transações, 7.042 fraudes (0,52%),
de 14/08 a 23/12/2020.

| Cenário no teste | Fraudes detectadas | Perdidas | Falsos alarmes | Custo (US$) |
|---|---|---|---|---|
| Sem modelo | 0 | 7.042 | 0 | 3.810.073 |
| Random Forest original, corte 0,5 | 4.985 | 2.057 | 120.011 | 1.446.207 |
| Ajustado, corte 0,5 | 4.843 | 2.199 | 102.030 | 1.290.984 |
| Ajustado, corte por custo (0,395) | 4.997 | 2.045 | 108.357 | 1.298.802 |

Em resumo: o modelo detecta cerca de 71% das fraudes, com precisão de ~4% (cerca de 22 falsos alarmes por fraude
detectada) e reduz o custo em ~66% frente a não ter modelo. No teste, o corte por custo ficou 0,6% acima do corte 0,5:
a diferença é pequena.

> Uma versão anterior do modelo (treinada no Colab, 300 árvores, limiar 0,85) tinha menos falsos alarmes: 31.314
> contra 108.357, com 5.096 fraudes detectadas. Ela continua no histórico do git (commit `b2df6f0`).
> A busca aleatória tem só 8 combinações e uma semente fixa, então modelos retreinados variam.

## Limitações

- **O valor da compra carrega quase todo o sinal.** Um modelo só com `amt` chega a 0,3366 de PR-AUC na validação,
  contra 0,3625 com as 9 colunas.
- **O score não é uma probabilidade calibrada**, porque o treino usou subamostragem. Ele serve para ordenar o risco.
- **US$ 10 por falso alarme é uma suposição.** Limiar e economia dependem desse número.
- **Uma única divisão dos dados e uma semente.** O teste foi consultado mais de uma vez na comparação dos modelos.
- **Fraudes mudam com o tempo**: o modelo precisa ser reavaliado periodicamente (é para isso que existe a DAG).

## Versões e o pickle

O modelo é salvo com `joblib` (pickle). Carregar com outra série do scikit-learn gera aviso e pode dar resultados
diferentes. Por isso `scikit-learn==1.5.2` está fixado no `requirements.txt` (app), em
`docker/requirements-airflow.txt` (DAG) e no `.venv` do projeto. Se você retreinar numa versão nova, atualize
os três.
