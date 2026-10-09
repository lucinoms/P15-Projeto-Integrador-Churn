
# P15 - Projeto Integrador UNIFAP
# Gerador do notebook final (.ipynb)

from pathlib import Path
import ast
import nbformat as nbf

BASE = Path(__file__).resolve().parent
DESTINO = BASE / "P15_Projeto_Integrador_Churn.ipynb"

nb = nbf.v4.new_notebook()
celulas = []


def texto(conteudo):
    celulas.append(nbf.v4.new_markdown_cell(conteudo))


def codigo(conteudo):
    ast.parse(conteudo)
    celulas.append(nbf.v4.new_code_cell(conteudo))


texto("""
# P15 — Projeto Integrador: Previsão de Churn

**Especialização em Inteligência Artificial — UNIFAP**

## Objetivo

Desenvolver um pipeline end-to-end de Engenharia de Dados
e Machine Learning, integrando duas fontes de dados,
armazenamento PostgreSQL, ETL Python, engenharia de
features, treinamento e avaliação de modelos.

**Problema:** classificação de clientes com possível
churn, definido neste experimento como recência de
compra superior a 90 dias.

**Natureza dos dados:** sintéticos, utilizados para
demonstração acadêmica.
""")

texto("""
## 1. Arquitetura do projeto

Fluxo implementado:

1. Fonte 1: histórico de vendas (`dados/vendas.csv`);
2. Fonte 2: cadastro de clientes (`dados/clientes.csv`);
3. PostgreSQL: banco `p15_churn_db`;
4. ETL: extração, validação e carga com Python;
5. Integração por `cliente_id`;
6. Engenharia de features;
7. Treinamento e validação cruzada;
8. Comparação de modelos e geração de artefatos.

O treinamento foi executado em Docker para garantir
um ambiente Linux reproduzível.
""")

codigo("""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

BASE = Path.cwd()
OUTPUT = BASE / "output"

print("Diretorio do projeto:", BASE)
""")

texto("""
## 2. Fontes de dados

As duas fontes são integradas pelo identificador
`cliente_id`.

A fonte de vendas contém informações transacionais.
A fonte de clientes contém informações cadastrais
e características comerciais.

Os dados são sintéticos e não contêm informações
reais de clientes.
""")

codigo("""
vendas = pd.read_csv(BASE / "dados" / "vendas.csv")
clientes = pd.read_csv(BASE / "dados" / "clientes.csv")

print("Vendas:", vendas.shape)
print("Clientes:", clientes.shape)

display(vendas.head())
display(clientes.head())
""")

texto("""
## 3. Validação das fontes

São verificadas a quantidade de registros,
a duplicidade de identificadores e a correspondência
entre as duas fontes.
""")

codigo("""
print("Vendas duplicadas por id_venda:",
      vendas["id_venda"].duplicated().sum())

print("Clientes duplicados por cliente_id:",
      clientes["cliente_id"].duplicated().sum())

ids_sem_cadastro = (
    set(vendas["cliente_id"])
    - set(clientes["cliente_id"])
)

print("Clientes referenciados sem cadastro:",
      len(ids_sem_cadastro))
""")

texto("""
## 4. ETL e armazenamento PostgreSQL

O script `etl_p15.py` executa a leitura e validação
das duas fontes e a carga no PostgreSQL.

O banco `p15_churn_db` contém as tabelas `clientes`
e `vendas`.

Na execução validada do projeto, foram registrados:

- 197 clientes;
- 787 vendas;
- 787 vendas associadas a clientes cadastrados.

A reprodução da carga depende de uma instância
PostgreSQL configurada conforme o README.
""")

codigo("""
from pathlib import Path

etl = BASE / "etl_p15.py"
print("Script ETL disponível:", etl.exists())

print("Arquivos de origem:")
print((BASE / "dados" / "clientes.csv").exists())
print((BASE / "dados" / "vendas.csv").exists())
""")

texto("""
## 5. Engenharia de features

O script `features_p15.py` consulta o PostgreSQL,
integra as tabelas e agrega o histórico de vendas
por cliente.

São produzidas variáveis de frequência, valor,
diversidade de produtos, características temporais
e atributos cadastrais.

O alvo `churn` foi definido como recência superior
a 90 dias na data de referência de 08/10/2026.
""")

codigo("""
df = pd.read_csv(
    OUTPUT / "clientes_features_p15.csv"
)

print("Clientes:", len(df))
print("Colunas totais:", len(df.columns))
print("Valores ausentes:", df.isna().sum().sum())
print("IDs duplicados:",
      df["cliente_id"].duplicated().sum())

display(df.head())
""")

texto("""
## 6. Análise exploratória — distribuição do alvo

A base apresenta desbalanceamento entre clientes
classificados com churn e sem churn.

Por isso, a avaliação não deve depender apenas
da acurácia. São utilizadas precisão, recall
e F1-score.
""")

codigo("""
distribuicao = df["churn"].value_counts().sort_index()

print(distribuicao)

ax = distribuicao.plot(
    kind="bar",
    figsize=(6, 4),
    title="Distribuicao do churn"
)

ax.set_xlabel("Classe")
ax.set_ylabel("Quantidade de clientes")
ax.set_xticklabels(["Sem churn", "Churn"], rotation=0)

plt.tight_layout()
plt.show()
""")

texto("""
## 7. Análise exploratória — comportamento de compra

São examinadas as distribuições de compras e
valores acumulados por cliente, sem interpretar
associações sintéticas como relações causais reais.
""")

codigo("""
fig, ax = plt.subplots(figsize=(8, 4))

df["total_compras"].hist(
    bins=15,
    ax=ax
)

ax.set_title("Distribuicao do total de compras")
ax.set_xlabel("Compras por cliente")
ax.set_ylabel("Frequencia")

plt.tight_layout()
plt.show()
""")

codigo("""
fig, ax = plt.subplots(figsize=(8, 4))

df.boxplot(
    column="valor_acumulado",
    by="churn",
    ax=ax
)

plt.suptitle("")
ax.set_title("Valor acumulado por classe")
ax.set_xlabel("Churn")
ax.set_ylabel("Valor acumulado")

plt.tight_layout()
plt.show()
""")

texto("""
## 8. Preparação para Machine Learning

O treinamento utiliza 27 features de entrada.

Foram excluídos identificadores, datas brutas
e a variável `recencia_dias`, que define diretamente
o alvo sintético.

O pré-processamento é realizado dentro do Pipeline
do scikit-learn, com imputação, padronização
e One-Hot Encoding.

A divisão é estratificada: 80% para treinamento
e 20% para teste.
""")

codigo("""
X_train = pd.read_csv(OUTPUT / "X_train_p15.csv")
X_test = pd.read_csv(OUTPUT / "X_test_p15.csv")
y_train = pd.read_csv(OUTPUT / "y_train_p15.csv")
y_test = pd.read_csv(OUTPUT / "y_test_p15.csv")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
""")

texto("""
## 9. Modelos avaliados

Foram comparados três classificadores:

- Regressão Logística;
- Random Forest;
- Gradient Boosting.

A seleção do modelo foi baseada no F1 médio
da validação cruzada estratificada com cinco folds,
calculada apenas no conjunto de treinamento.
""")

codigo("""
comparacao = pd.read_csv(
    OUTPUT / "comparacao_modelos_p15.csv"
)

display(comparacao)
""")

codigo("""
ax = comparacao.plot(
    x="modelo",
    y="f1_cv_medio",
    kind="bar",
    legend=False,
    figsize=(8, 4),
    title="F1 medio na validacao cruzada"
)

ax.set_xlabel("Modelo")
ax.set_ylabel("F1 medio")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
""")

texto("""
## 10. Seleção e resultados

O Random Forest apresentou o maior F1 médio
na validação cruzada e foi selecionado pelo
critério definido antes da avaliação no teste.

Resultados observados:

| Modelo | F1 CV médio | Acurácia teste | F1 teste |
|---|---:|---:|---:|
| Random Forest | 0,2497 | 0,8000 | 0,2000 |
| Gradient Boosting | 0,2467 | 0,8250 | 0,2222 |
| Regressão Logística | 0,2114 | 0,8000 | 0,4286 |

Embora a Regressão Logística tenha apresentado
maior F1 no teste, esse resultado não foi utilizado
para escolher o modelo, evitando seleção orientada
pelo conjunto reservado para avaliação.
""")

codigo("""
from IPython.display import Image, display

display(Image(
    filename=str(OUTPUT / "matriz_confusao_p15.png")
))
""")

texto("""
## 11. Interpretação da avaliação

O Random Forest obteve acurácia de 80% no teste,
mas apresentou recall de apenas 20% para churn.

Isso significa que identificou somente um dos
cinco clientes classificados com churn no conjunto
de teste.

A diferença entre acurácia e recall demonstra
a importância de avaliar métricas adequadas
para classes desbalanceadas.

O resultado não deve ser interpretado como
desempenho comprovado em produção.
""")

texto("""
## 12. Artefatos do modelo

O arquivo `modelo_churn_p15.pkl` contém o Pipeline
completo, incluindo pré-processamento e classificador.

O projeto também disponibiliza métricas em CSV,
gráficos, relatório de classificação, partições
de treinamento e teste e metadados do experimento.

A execução deve ocorrer em ambiente com dependências
compatíveis, preferencialmente utilizando Docker.
""")

codigo("""
artefatos = [
    "modelo_churn_p15.pkl",
    "comparacao_modelos_p15.csv",
    "matriz_confusao_p15.png",
    "metadados_p15.json",
    "relatorio_classificacao_p15.txt"
]

for nome in artefatos:
    print(nome, (OUTPUT / nome).exists())
""")

texto("""
## 13. Limitações e trabalhos futuros

Este estudo utiliza dados sintéticos, com
197 clientes e apenas 27 casos positivos de churn.

A variável-alvo foi construída a partir da
recência de compras, não de cancelamentos
efetivamente observados.

Embora a recência direta tenha sido excluída
das entradas, variáveis derivadas do histórico
podem apresentar relações indiretas com o alvo.

Para uma aplicação real, recomenda-se:

- Utilizar dados históricos reais e autorizados;
- Definir uma janela temporal de observação;
- Prever churn em uma janela futura;
- Evitar informações posteriores ao momento da previsão;
- Realizar validação temporal;
- Monitorar data drift, concept drift e desempenho;
- Avaliar estratégias para o desbalanceamento.

Assim, o projeto demonstra a integração técnica
do pipeline, sem reivindicar validação prospectiva.
""")

texto("""
## 14. Conclusão

O Projeto Integrador P15 implementou um fluxo
de Engenharia de Dados e Machine Learning,
integrando duas fontes sintéticas, PostgreSQL,
ETL em Python, engenharia de features,
treinamento e avaliação de três modelos.

O pipeline produziu artefatos auditáveis
e reproduzíveis com Docker.

A análise evidencia tanto a viabilidade técnica
da integração quanto as limitações do modelo
de churn desenvolvido para fins acadêmicos.
""")

nb.cells = celulas
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.12"
    }
}

nbf.validate(nb)
nbf.write(nb, DESTINO)

print("NOTEBOOK P15 CRIADO COM SUCESSO!")
print("Arquivo:", DESTINO.name)
print("Quantidade de celulas:", len(nb.cells))
print(
    "Celulas de codigo:",
    sum(c.cell_type == "code" for c in nb.cells)
)
