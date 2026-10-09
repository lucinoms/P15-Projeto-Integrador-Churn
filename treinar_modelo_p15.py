
# P15 - Projeto Integrador UNIFAP
# Treinamento e avaliacao de modelos de churn

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "output"
OUTPUT.mkdir(exist_ok=True)

ARQUIVO = OUTPUT / "clientes_features_p15.csv"

print("=== P15 - MACHINE LEARNING ===")

# 1. Carregar o dataset
df = pd.read_csv(ARQUIVO)

if df["churn"].nunique() != 2:
    raise ValueError("O alvo precisa ter duas classes.")

# 2. Selecionar as features
# Excluir identificadores, datas e variaveis que
# definem diretamente o alvo sintetico de churn.

features_numericas = [
    "idade",
    "total_compras",
    "valor_acumulado",
    "ticket_medio",
    "quantidade_total",
    "quantidade_media",
    "diversidade_produtos",
    "maior_compra",
    "menor_compra",
    "desvio_valor",
    "tempo_cadastro_dias",
    "mes_cadastro",
    "ano_cadastro",
    "media_mensal_compras",
    "valor_por_produto",
    "amplitude_compras",
    "produtos_por_compra",
    "diversidade_por_compra",
    "coef_variacao_valor",
    "aceita_marketing",
    "possui_programa_fidelidade"
]

features_categoricas = [
    "cidade",
    "segmento",
    "faixa_etaria",
    "canal_aquisicao",
    "tipo_contrato",
    "preferencia_contato"
]

features = features_numericas + features_categoricas

X = df[features].copy()
y = df["churn"].astype(int)

# 3. Divisao treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Clientes:", len(df))
print("Features de entrada:", len(features))
print("Treinamento:", len(X_train))
print("Teste:", len(X_test))

# 4. Pre-processamento
# Ajustado exclusivamente nos folds de treino
# durante a validacao cruzada.

preprocessador = ColumnTransformer(
    transformers=[
        (
            "numericas",
            Pipeline([
                ("imputacao", SimpleImputer(
                    strategy="median"
                )),
                ("escala", StandardScaler())
            ]),
            features_numericas
        ),
        (
            "categoricas",
            Pipeline([
                ("imputacao", SimpleImputer(
                    strategy="most_frequent"
                )),
                ("encoding", OneHotEncoder(
                    handle_unknown="ignore"
                ))
            ]),
            features_categoricas
        )
    ]
)

# 5. Modelos para comparacao
modelos = {
    "Regressao Logistica": LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

validacao = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

resultados = []
pipelines = {}

# 6. Validacao cruzada e teste
for nome, modelo in modelos.items():

    pipeline = Pipeline([
        ("preprocessamento", preprocessador),
        ("modelo", modelo)
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=validacao,
        scoring="f1"
    )

    pipeline.fit(X_train, y_train)
    previsoes = pipeline.predict(X_test)

    resultado = {
        "modelo": nome,
        "f1_cv_medio": float(scores.mean()),
        "f1_cv_desvio": float(scores.std()),
        "acuracia_teste": accuracy_score(
            y_test, previsoes
        ),
        "precisao_teste": precision_score(
            y_test, previsoes, zero_division=0
        ),
        "recall_teste": recall_score(
            y_test, previsoes, zero_division=0
        ),
        "f1_teste": f1_score(
            y_test, previsoes, zero_division=0
        )
    }

    resultados.append(resultado)
    pipelines[nome] = pipeline

    print("\nModelo:", nome)
    print("F1 CV medio:", round(scores.mean(), 4))
    print(
        "F1 teste:",
        round(resultado["f1_teste"], 4)
    )

# 7. Selecionar pelo F1 da validacao cruzada
comparacao = pd.DataFrame(resultados)
comparacao = comparacao.sort_values(
    "f1_cv_medio",
    ascending=False
).reset_index(drop=True)

comparacao.to_csv(
    OUTPUT / "comparacao_modelos_p15.csv",
    index=False
)

melhor_nome = comparacao.iloc[0]["modelo"]
melhor_pipeline = pipelines[melhor_nome]

previsoes_finais = melhor_pipeline.predict(X_test)

print("\n=== COMPARACAO DOS MODELOS ===")
print(comparacao.to_string(index=False))

print("\n=== MODELO SELECIONADO ===")
print(melhor_nome)

print("\nRelatorio de classificacao:")
relatorio = classification_report(
    y_test,
    previsoes_finais,
    zero_division=0
)
print(relatorio)

# 8. Salvar o modelo completo
# Inclui imputacao, escala, encoding e classificador.
joblib.dump(
    melhor_pipeline,
    OUTPUT / "modelo_churn_p15.pkl"
)

# 9. Salvar relatorio
(OUTPUT / "relatorio_classificacao_p15.txt").write_text(
    relatorio,
    encoding="utf-8"
)

# 10. Metadados
metadados = {
    "projeto": "P15 - Projeto Integrador UNIFAP",
    "modelo_selecionado": melhor_nome,
    "criterio_selecao": "F1 medio em validacao cruzada",
    "total_clientes": int(len(df)),
    "amostras_treino": int(len(X_train)),
    "amostras_teste": int(len(X_test)),
    "features": features,
    "alvo": "churn",
    "definicao_alvo": (
        "Recencia superior a 90 dias em 2026-10-08"
    ),
    "observacao": (
        "Experimento academico com dados sinteticos; "
        "nao representa validacao prospectiva real."
    )
}

with open(
    OUTPUT / "metadados_p15.json",
    "w",
    encoding="utf-8"
) as arquivo:
    json.dump(
        metadados,
        arquivo,
        indent=4,
        ensure_ascii=False
    )

# 11. Grafico de comparacao
plt.figure(figsize=(9, 5))
plt.bar(
    comparacao["modelo"],
    comparacao["f1_cv_medio"]
)
plt.ylabel("F1 medio - Validacao cruzada")
plt.title("P15 - Comparacao dos modelos")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig(
    OUTPUT / "comparacao_modelos_p15.png",
    dpi=150
)
plt.close()

# 12. Matriz de confusao
matriz = confusion_matrix(
    y_test,
    previsoes_finais,
    labels=[0, 1]
)

ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=["Sem churn", "Churn"]
).plot(values_format="d")

plt.title("P15 - Matriz de confusao")
plt.tight_layout()
plt.savefig(
    OUTPUT / "matriz_confusao_p15.png",
    dpi=150
)
plt.close()

# 13. Exportar particoes para auditoria
X_train.to_csv(
    OUTPUT / "X_train_p15.csv",
    index=False
)
X_test.to_csv(
    OUTPUT / "X_test_p15.csv",
    index=False
)
y_train.to_csv(
    OUTPUT / "y_train_p15.csv",
    index=False
)
y_test.to_csv(
    OUTPUT / "y_test_p15.csv",
    index=False
)

print("\nArquivos gerados em:", OUTPUT)
print("\nTREINAMENTO P15 CONCLUIDO COM SUCESSO!")
