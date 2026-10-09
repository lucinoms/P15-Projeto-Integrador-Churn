
# P15 - Projeto Integrador UNIFAP
# Engenharia de Features - PostgreSQL + Python

import os
import getpass
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "output"
OUTPUT.mkdir(exist_ok=True)

DATA_REFERENCIA = pd.Timestamp("2026-10-08")
LIMITE_CHURN = 90


def carregar_dados():
    senha = os.getenv("P15_DB_PASSWORD")

    if not senha:
        senha = getpass.getpass(
            "Senha do PostgreSQL: "
        )

    conexao = psycopg2.connect(
        dbname="p15_churn_db",
        user="postgres",
        password=senha,
        host="localhost",
        port=5432
    )

    try:
        sql = """
            SELECT
                v.id_venda,
                v.cliente_id,
                v.data_cadastro,
                v.ultima_compra,
                v.cidade,
                v.segmento,
                v.produto_id,
                v.idade,
                v.quantidade,
                v.valor,
                c.canal_aquisicao,
                c.tipo_contrato,
                c.preferencia_contato,
                c.aceita_marketing,
                c.possui_programa_fidelidade
            FROM vendas v
            INNER JOIN clientes c
                ON v.cliente_id = c.cliente_id
        """

        with conexao.cursor() as cursor:
            cursor.execute(sql)
            colunas = [
                coluna[0] for coluna in cursor.description
            ]
            registros = cursor.fetchall()

        df = pd.DataFrame(registros, columns=colunas)

    finally:
        conexao.close()

    print("[EXTRACT] Registros integrados:", len(df))
    return df


def construir_features(df):
    df = df.copy()

    df["data_cadastro"] = pd.to_datetime(
        df["data_cadastro"]
    )
    df["ultima_compra"] = pd.to_datetime(
        df["ultima_compra"]
    )

    df["valor"] = pd.to_numeric(df["valor"])
    df["quantidade"] = pd.to_numeric(df["quantidade"])

    df = df[
        (df["data_cadastro"] <= DATA_REFERENCIA)
        & (df["ultima_compra"] <= DATA_REFERENCIA)
    ].copy()

    df["valor_total"] = df["valor"] * df["quantidade"]

    clientes = df.groupby("cliente_id").agg(
        data_cadastro=("data_cadastro", "min"),
        ultima_compra=("ultima_compra", "max"),
        cidade=("cidade", "first"),
        segmento=("segmento", "first"),
        idade=("idade", "median"),
        canal_aquisicao=("canal_aquisicao", "first"),
        tipo_contrato=("tipo_contrato", "first"),
        preferencia_contato=("preferencia_contato", "first"),
        aceita_marketing=("aceita_marketing", "first"),
        possui_programa_fidelidade=(
            "possui_programa_fidelidade", "first"
        ),
        total_compras=("id_venda", "count"),
        valor_acumulado=("valor_total", "sum"),
        ticket_medio=("valor_total", "mean"),
        quantidade_total=("quantidade", "sum"),
        quantidade_media=("quantidade", "mean"),
        diversidade_produtos=("produto_id", "nunique"),
        maior_compra=("valor_total", "max"),
        menor_compra=("valor_total", "min"),
        desvio_valor=("valor_total", "std")
    ).reset_index()

    clientes["desvio_valor"] = (
        clientes["desvio_valor"].fillna(0)
    )

    clientes["tempo_cadastro_dias"] = (
        DATA_REFERENCIA - clientes["data_cadastro"]
    ).dt.days

    clientes["recencia_dias"] = (
        DATA_REFERENCIA - clientes["ultima_compra"]
    ).dt.days

    clientes["mes_cadastro"] = (
        clientes["data_cadastro"].dt.month
    )

    clientes["ano_cadastro"] = (
        clientes["data_cadastro"].dt.year
    )

    clientes["media_mensal_compras"] = (
        clientes["total_compras"]
        / (
            clientes["tempo_cadastro_dias"].clip(lower=1)
            / 30.44
        )
    )

    clientes["valor_por_produto"] = (
        clientes["valor_acumulado"]
        / clientes["quantidade_total"].clip(lower=1)
    )

    clientes["amplitude_compras"] = (
        clientes["maior_compra"] - clientes["menor_compra"]
    )

    clientes["produtos_por_compra"] = (
        clientes["quantidade_total"]
        / clientes["total_compras"].clip(lower=1)
    )

    clientes["diversidade_por_compra"] = (
        clientes["diversidade_produtos"]
        / clientes["total_compras"].clip(lower=1)
    )

    clientes["coef_variacao_valor"] = (
        clientes["desvio_valor"]
        / clientes["ticket_medio"].replace(0, np.nan)
    ).fillna(0)

    clientes["faixa_etaria"] = pd.cut(
        clientes["idade"],
        bins=[0, 25, 40, 60, np.inf],
        labels=[
            "ate_25", "26_40", "41_60", "acima_60"
        ]
    ).astype(str)

    # Rotulo sintetico para demonstracao academica.
    clientes["churn"] = (
        clientes["recencia_dias"] > LIMITE_CHURN
    ).astype(int)

    return clientes


def main():
    print("=== P15 - ENGENHARIA DE FEATURES ===")

    dados = carregar_dados()
    clientes = construir_features(dados)

    arquivo = OUTPUT / "clientes_features_p15.csv"
    clientes.to_csv(arquivo, index=False)

    print("\n=== RESULTADOS ===")
    print("Clientes:", len(clientes))
    print("Colunas:", len(clientes.columns))
    print("Valores ausentes:",
          int(clientes.isna().sum().sum()))
    print("IDs duplicados:",
          int(clientes["cliente_id"].duplicated().sum()))

    print("\nDistribuicao do churn:")
    print(clientes["churn"].value_counts().sort_index())

    print("\nArquivo gerado:", arquivo)
    print("\nFEATURES P15 GERADAS COM SUCESSO!")


if __name__ == "__main__":
    main()
