
import os
import numpy as np
import pandas as pd

# P15 - Projeto Integrador UNIFAP
# Segunda fonte: cadastro sintetico de clientes

os.makedirs("dados", exist_ok=True)

# 1. Carregar a fonte de vendas
vendas = pd.read_csv("dados/vendas.csv")

if "cliente_id" not in vendas.columns:
    raise ValueError("Coluna cliente_id nao encontrada.")

# 2. Identificar os clientes existentes
clientes = (
    vendas[["cliente_id"]]
    .drop_duplicates()
    .sort_values("cliente_id")
    .reset_index(drop=True)
)

# 3. Gerar atributos cadastrais sinteticos
rng = np.random.default_rng(42)

n = len(clientes)

clientes["canal_aquisicao"] = rng.choice(
    ["Online", "Loja fisica", "Indicacao", "Campanha"],
    size=n,
    p=[0.40, 0.30, 0.20, 0.10]
)

clientes["tipo_contrato"] = rng.choice(
    ["Mensal", "Trimestral", "Anual"],
    size=n,
    p=[0.50, 0.30, 0.20]
)

clientes["preferencia_contato"] = rng.choice(
    ["Email", "WhatsApp", "Telefone"],
    size=n,
    p=[0.40, 0.45, 0.15]
)

clientes["aceita_marketing"] = rng.choice(
    [0, 1],
    size=n,
    p=[0.35, 0.65]
)

clientes["possui_programa_fidelidade"] = rng.choice(
    [0, 1],
    size=n,
    p=[0.60, 0.40]
)

# 4. Validar a integridade dos identificadores
assert clientes["cliente_id"].is_unique
assert clientes["cliente_id"].notna().all()

# 5. Salvar a segunda fonte
arquivo = "dados/clientes.csv"
clientes.to_csv(arquivo, index=False, encoding="utf-8")

# 6. Apresentar os resultados
print("=== P15 - CADASTRO DE CLIENTES ===")
print("Clientes cadastrados:", len(clientes))
print("Colunas:", len(clientes.columns))
print("IDs duplicados:", clientes["cliente_id"].duplicated().sum())
print("Arquivo gerado:", arquivo)

print("\nAmostra dos dados:")
print(clientes.head().to_string(index=False))

print("\nSEGUNDA FONTE GERADA COM SUCESSO!")
