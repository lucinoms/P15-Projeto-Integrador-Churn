
import os
import getpass
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# P15 - Projeto Integrador UNIFAP
# ETL: CSV -> Validacao -> PostgreSQL

BASE = Path(__file__).resolve().parent
ARQUIVO_CLIENTES = BASE / "dados" / "clientes.csv"
ARQUIVO_VENDAS = BASE / "dados" / "vendas.csv"

BANCO = "p15_churn_db"
USUARIO = "postgres"
HOST = "localhost"
PORTA = 5432

COLUNAS_CLIENTES = [
    "cliente_id",
    "canal_aquisicao",
    "tipo_contrato",
    "preferencia_contato",
    "aceita_marketing",
    "possui_programa_fidelidade"
]

COLUNAS_VENDAS = [
    "id_venda",
    "cliente_id",
    "data_cadastro",
    "ultima_compra",
    "cidade",
    "segmento",
    "produto_id",
    "idade",
    "quantidade",
    "valor"
]


def validar_colunas(df, colunas, nome):
    faltantes = set(colunas) - set(df.columns)
    if faltantes:
        raise ValueError(
            f"{nome}: colunas ausentes: {sorted(faltantes)}"
        )


def preparar_clientes():
    print("\n[EXTRACT] Lendo cadastro de clientes...")
    df = pd.read_csv(ARQUIVO_CLIENTES)

    validar_colunas(df, COLUNAS_CLIENTES, "clientes")
    df = df[COLUNAS_CLIENTES].copy()

    if df.isna().any().any():
        raise ValueError("Cadastro contem valores ausentes.")

    for coluna in [
        "cliente_id",
        "aceita_marketing",
        "possui_programa_fidelidade"
    ]:
        df[coluna] = pd.to_numeric(
            df[coluna], errors="raise"
        ).astype(int)

    if df["cliente_id"].duplicated().any():
        raise ValueError("Existem clientes duplicados.")

    for coluna in [
        "aceita_marketing",
        "possui_programa_fidelidade"
    ]:
        if not df[coluna].isin([0, 1]).all():
            raise ValueError(
                f"Valores invalidos na coluna {coluna}."
            )

    print("[TRANSFORM] Clientes validados:", len(df))
    return df


def preparar_vendas(ids_clientes):
    print("\n[EXTRACT] Lendo historico de vendas...")
    df = pd.read_csv(ARQUIVO_VENDAS)

    validar_colunas(df, COLUNAS_VENDAS, "vendas")
    df = df[COLUNAS_VENDAS].copy()

    if df.isna().any().any():
        raise ValueError("Vendas contem valores ausentes.")

    for coluna in [
        "id_venda", "cliente_id", "idade", "quantidade"
    ]:
        df[coluna] = pd.to_numeric(
            df[coluna], errors="raise"
        ).astype(int)

    df["valor"] = pd.to_numeric(
        df["valor"], errors="raise"
    ).round(2)

    for coluna in ["data_cadastro", "ultima_compra"]:
        df[coluna] = pd.to_datetime(
            df[coluna], errors="raise"
        ).dt.date

    if df["id_venda"].duplicated().any():
        raise ValueError("Existem vendas duplicadas.")

    if not df["cliente_id"].isin(ids_clientes).all():
        raise ValueError(
            "Existem vendas com clientes nao cadastrados."
        )

    if (df["quantidade"] <= 0).any():
        raise ValueError("Quantidade invalida.")

    if (df["valor"] <= 0).any():
        raise ValueError("Valor de venda invalido.")

    if (df["idade"] <= 0).any():
        raise ValueError("Idade invalida.")

    print("[TRANSFORM] Vendas validadas:", len(df))
    return df


def carregar_postgresql(clientes, vendas, senha):
    print("\n[LOAD] Conectando ao PostgreSQL...")

    conexao = psycopg2.connect(
        dbname=BANCO,
        user=USUARIO,
        password=senha,
        host=HOST,
        port=PORTA
    )

    try:
        with conexao:
            with conexao.cursor() as cursor:

                # Inserir ou atualizar clientes
                sql_clientes = """
                    INSERT INTO clientes (
                        cliente_id,
                        canal_aquisicao,
                        tipo_contrato,
                        preferencia_contato,
                        aceita_marketing,
                        possui_programa_fidelidade
                    ) VALUES %s
                    ON CONFLICT (cliente_id)
                    DO UPDATE SET
                        canal_aquisicao =
                            EXCLUDED.canal_aquisicao,
                        tipo_contrato =
                            EXCLUDED.tipo_contrato,
                        preferencia_contato =
                            EXCLUDED.preferencia_contato,
                        aceita_marketing =
                            EXCLUDED.aceita_marketing,
                        possui_programa_fidelidade =
                            EXCLUDED.possui_programa_fidelidade
                """

                execute_values(
                    cursor,
                    sql_clientes,
                    list(
                        clientes[COLUNAS_CLIENTES]
                        .itertuples(index=False, name=None)
                    )
                )

                print("[LOAD] Clientes processados.")

                # Inserir ou atualizar vendas
                sql_vendas = """
                    INSERT INTO vendas (
                        id_venda,
                        cliente_id,
                        data_cadastro,
                        ultima_compra,
                        cidade,
                        segmento,
                        produto_id,
                        idade,
                        quantidade,
                        valor
                    ) VALUES %s
                    ON CONFLICT (id_venda)
                    DO UPDATE SET
                        cliente_id = EXCLUDED.cliente_id,
                        data_cadastro = EXCLUDED.data_cadastro,
                        ultima_compra = EXCLUDED.ultima_compra,
                        cidade = EXCLUDED.cidade,
                        segmento = EXCLUDED.segmento,
                        produto_id = EXCLUDED.produto_id,
                        idade = EXCLUDED.idade,
                        quantidade = EXCLUDED.quantidade,
                        valor = EXCLUDED.valor
                """

                execute_values(
                    cursor,
                    sql_vendas,
                    list(
                        vendas[COLUNAS_VENDAS]
                        .itertuples(index=False, name=None)
                    )
                )

                print("[LOAD] Vendas processadas.")

                # Validar a carga
                cursor.execute("SELECT COUNT(*) FROM clientes")
                total_clientes = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM vendas")
                total_vendas = cursor.fetchone()[0]

                print("\n=== VALIDACAO POSTGRESQL ===")
                print("Clientes no banco:", total_clientes)
                print("Vendas no banco:", total_vendas)

                if total_clientes != len(clientes):
                    raise ValueError(
                        "Contagem de clientes divergente."
                    )

                if total_vendas != len(vendas):
                    raise ValueError(
                        "Contagem de vendas divergente."
                    )

        print("\nETL P15 EXECUTADO COM SUCESSO!")

    finally:
        conexao.close()


def main():
    print("=== P15 - ETL PYTHON + POSTGRESQL ===")

    if not ARQUIVO_CLIENTES.exists():
        raise FileNotFoundError(ARQUIVO_CLIENTES)

    if not ARQUIVO_VENDAS.exists():
        raise FileNotFoundError(ARQUIVO_VENDAS)

    clientes = preparar_clientes()
    vendas = preparar_vendas(
        set(clientes["cliente_id"])
    )

    senha = os.getenv("P15_DB_PASSWORD")

    if not senha:
        senha = getpass.getpass(
            "\nSenha do PostgreSQL (nao aparece ao digitar): "
        )

    carregar_postgresql(clientes, vendas, senha)


if __name__ == "__main__":
    main()
