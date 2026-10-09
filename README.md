\# P15 — Projeto Integrador: Previsão de Churn



\*\*Especialização em Inteligência Artificial — Universidade Federal do Amapá (UNIFAP)\*\*



\*\*Disciplina:\*\* Banco de Dados e Engenharia de Dados para IA  

\*\*Projeto:\*\* Pipeline End-to-End de Engenharia de Dados e Machine Learning  

\*\*Linguagem:\*\* Python 3.12  

\*\*Banco de dados:\*\* PostgreSQL 17  

\*\*Ambiente de execução:\*\* Docker



\## 1. Apresentação



Este projeto desenvolve uma solução integrada de Engenharia de Dados e Machine Learning para classificação de clientes com possível churn (inatividade comercial).



O trabalho integra conhecimentos desenvolvidos nos módulos anteriores da disciplina, especialmente preparação de dados com Pandas, engenharia de features e treinamento de modelos supervisionados.



Os dados utilizados são sintéticos, destinados exclusivamente à demonstração acadêmica.



\## 2. Objetivo



Implementar um pipeline end-to-end que contemple:



\- Integração de duas fontes de dados;

\- Armazenamento relacional em PostgreSQL;

\- Processo ETL desenvolvido em Python;

\- Preparação e engenharia de features;

\- Treinamento de modelos de classificação;

\- Avaliação e comparação dos resultados;

\- Geração de artefatos reproduzíveis e documentação técnica.



\## 3. Arquitetura da solução



O fluxo implementado é:



\*\*Histórico de vendas + Cadastro de clientes → ETL Python → PostgreSQL → Engenharia de Features → Machine Learning → Avaliação → Artefatos\*\*



O identificador `cliente\_id` permite integrar as informações cadastrais com as transações comerciais.



O Docker é utilizado para executar o treinamento e o notebook em ambiente Linux, reduzindo problemas de compatibilidade entre sistemas operacionais.



\## 4. Fontes de dados



\### Fonte 1 — Histórico de vendas



Arquivo: `dados/vendas.csv`



Contém registros transacionais preparados na atividade P12.



\*\*Quantidade:\*\* 787 registros validados.



Inclui identificadores, datas, informações comerciais, produtos, quantidades e valores.



\### Fonte 2 — Cadastro de clientes



Arquivo: `dados/clientes.csv`



Contém informações cadastrais e comerciais geradas para integração no P15.



\*\*Quantidade:\*\* 197 clientes.



Inclui canal de aquisição, tipo de contrato, preferência de contato, aceitação de marketing e participação em programa de fidelidade.



\### Integração



As fontes são relacionadas por `cliente\_id`.



Na validação realizada no PostgreSQL, todas as 787 vendas estavam associadas a clientes cadastrados.



\## 5. Banco de dados PostgreSQL



\*\*Banco:\*\* `p15\_churn\_db`



\*\*Tabelas principais:\*\*



\- `clientes`: informações cadastrais;

\- `vendas`: histórico transacional.



O banco organiza os dados e permite consultas relacionais utilizadas no processo de engenharia de features.



Para reproduzir a solução, é necessário criar o banco e executar o script SQL de definição das tabelas antes da carga ETL.



\## 6. Processo ETL



\*\*Script:\*\* `etl\_p15.py`



O processo contempla:



1\. Extração dos arquivos CSV;

2\. Validação e preparação dos registros;

3\. Conexão ao PostgreSQL;

4\. Carga nas tabelas;

5\. Conferência dos quantitativos armazenados.



\*\*Resultado validado:\*\*



\- 197 clientes carregados;

\- 787 vendas carregadas;

\- 787 vendas vinculadas a clientes válidos.



A conexão com o PostgreSQL solicita a senha durante a execução, evitando sua inclusão diretamente no código-fonte.



\## 7. Engenharia de features



\*\*Script:\*\* `features\_p15.py`



Os dados são integrados e agregados por cliente para formar a base de classificação.



Foram desenvolvidas variáveis relacionadas a frequência de compras, valor acumulado, ticket médio, diversidade de produtos, características temporais e informações cadastrais.



\*\*Resultado:\*\*



\- 197 clientes;

\- 32 colunas no conjunto de features e informações auxiliares;

\- 27 features de entrada utilizadas no treinamento;

\- Nenhum valor ausente;

\- Nenhum identificador de cliente duplicado.



\### Variável-alvo



O churn foi definido, para este experimento, como recência de compra superior a 90 dias, considerando a data de referência de 08/10/2026.



\*\*Distribuição:\*\*



\- Classe 0 — Sem churn: 170 clientes;

\- Classe 1 — Churn: 27 clientes.



A variável `recencia\_dias`, diretamente relacionada à definição do alvo, não foi utilizada como preditora.



\## 8. Machine Learning



\*\*Script:\*\* `treinar\_modelo\_p15.py`



Foram utilizados três algoritmos:



\- Regressão Logística;

\- Random Forest;

\- Gradient Boosting.



A preparação das variáveis é realizada por meio de um Pipeline do scikit-learn, com tratamento de valores ausentes, padronização de variáveis numéricas e codificação de categorias.



A divisão dos dados foi estratificada:



\- Treinamento: 157 clientes;

\- Teste: 40 clientes.



A seleção do modelo foi realizada pelo F1-score médio obtido na validação cruzada do conjunto de treinamento.



\## 9. Resultados



| Modelo | F1 médio CV | Acurácia no teste | F1 no teste |

|---|---:|---:|---:|

| Random Forest | 0,2497 | 80,00% | 0,2000 |

| Gradient Boosting | 0,2467 | 82,50% | 0,2222 |

| Regressão Logística | 0,2114 | 80,00% | 0,4286 |



\*\*Modelo selecionado:\*\* Random Forest.



O Random Forest obteve o maior F1 médio na validação cruzada, embora a Regressão Logística tenha alcançado melhor F1 no conjunto de teste.



\### Interpretação



No conjunto de teste, o Random Forest apresentou:



\- Acurácia: 80%;

\- Precisão para churn: 20%;

\- Recall para churn: 20%;

\- F1-score para churn: 20%.



O modelo identificou apenas um dos cinco clientes classificados com churn no teste.



Esses resultados demonstram a necessidade de considerar métricas específicas para a classe minoritária, evitando interpretar a acurácia isoladamente como indicador de bom desempenho.



\## 10. Estrutura do projeto



```text

P15\_PROJETO\_INTEGRADOR/

│

├── dados/

│   ├── clientes.csv

│   └── vendas.csv

│

├── output/

│   ├── clientes\_features\_p15.csv

│   ├── comparacao\_modelos\_p15.csv

│   ├── comparacao\_modelos\_p15.png

│   ├── matriz\_confusao\_p15.png

│   ├── modelo\_churn\_p15.pkl

│   ├── metadados\_p15.json

│   ├── relatorio\_classificacao\_p15.txt

│   ├── X\_train\_p15.csv

│   ├── X\_test\_p15.csv

│   ├── y\_train\_p15.csv

│   └── y\_test\_p15.csv

│

├── gerar\_clientes\_p15.py

├── etl\_p15.py

├── features\_p15.py

├── treinar\_modelo\_p15.py

├── gerar\_notebook\_p15.py

├── P15\_Projeto\_Integrador\_Churn.ipynb

├── P15\_Projeto\_Integrador\_Churn\_Executado.ipynb

├── requirements.txt

├── Dockerfile

└── README.md

```



O script SQL de criação das tabelas deverá ser incluído no repositório para permitir a reprodução integral da etapa PostgreSQL.



\## 11. Como executar



\### Pré-requisitos



\- Docker Desktop com suporte a contêineres Linux;

\- PostgreSQL;

\- Banco `p15\_churn\_db` criado;

\- Arquivos CSV disponíveis na pasta `dados`;

\- Tabelas do banco criadas conforme o script SQL.



\### Etapa 1 — Preparar o banco



Criar o banco `p15\_churn\_db` no PostgreSQL e executar o script SQL de criação das tabelas.



\### Etapa 2 — Construir a imagem Docker



No terminal, dentro da pasta do projeto:



```cmd

docker build -t p15-churn-ml .

```



\### Etapa 3 — Executar o ETL



O ETL requer acesso ao PostgreSQL. A conexão deve ser configurada conforme o ambiente onde o banco estiver instalado.



Em um terminal com Python e as dependências necessárias:



```cmd

python etl\_p15.py

```



\### Etapa 4 — Gerar as features



Após a carga e a validação das tabelas:



```cmd

python features\_p15.py

```



\### Etapa 5 — Treinar os modelos



Com o arquivo `output/clientes\_features\_p15.csv` disponível:



```cmd

docker run --rm -v "%cd%:/app" -w /app p15-churn-ml

```



Esse comando executa `treinar\_modelo\_p15.py` e grava os resultados na pasta `output`.



\### Etapa 6 — Executar o notebook



```cmd

docker run --rm -v "%cd%:/app" -w /app p15-churn-ml python -m jupyter nbconvert --to notebook --execute P15\_Projeto\_Integrador\_Churn.ipynb --output P15\_Projeto\_Integrador\_Churn\_Executado.ipynb --ExecutePreprocessor.timeout=300

```



O comando executa o notebook e salva uma versão com os resultados incorporados.



\## 12. Validação do notebook



O notebook final foi gerado com a biblioteca `nbformat` e executado no Docker utilizando `nbconvert`.



\*\*Resultado da validação:\*\*



\- 28 células no notebook;

\- 13 células de código;

\- 13 células executadas;

\- 0 erros registrados.



\## 13. Limitações



A base utilizada é sintética, e o alvo foi definido por uma regra de inatividade, não por cancelamentos efetivamente observados.



A amostra possui apenas 27 casos positivos de churn, o que limita a estabilidade das métricas.



Para uma aplicação real, seria necessário utilizar dados históricos adequados, estabelecer uma janela futura de previsão, realizar validação temporal e monitorar alterações nas distribuições dos dados e no comportamento dos clientes.



O objetivo principal deste projeto é demonstrar a integração e a reprodutibilidade das etapas de Engenharia de Dados e Machine Learning.



\## 14. Conclusão



O Projeto Integrador P15 implementou um pipeline completo de Engenharia de Dados e Machine Learning, integrando duas fontes sintéticas, armazenamento PostgreSQL, ETL Python, engenharia de features, treinamento e comparação de três classificadores.



A solução gerou arquivos de dados, métricas, gráficos, modelo serializado e notebook executado, permitindo analisar tanto o funcionamento técnico do pipeline quanto as limitações de desempenho do modelo.



\*\*Universidade Federal do Amapá — UNIFAP\*\*  

\*\*Especialização em Inteligência Artificial\*\*  

\*\*Projeto Integrador P15\*\*



