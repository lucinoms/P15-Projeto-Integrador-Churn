
-- =====================================================
-- P15 - PROJETO INTEGRADOR UNIFAP
-- Pipeline de Engenharia de Dados e Machine Learning
-- Banco: p15_churn_db
-- SGBD: PostgreSQL 17
-- =====================================================

-- Execute este script conectado ao banco p15_churn_db.

-- 1. Cadastro de clientes
CREATE TABLE IF NOT EXISTS clientes (
    cliente_id INTEGER PRIMARY KEY,
    canal_aquisicao VARCHAR(50),
    tipo_contrato VARCHAR(30),
    preferencia_contato VARCHAR(30),
    aceita_marketing INTEGER,
    possui_programa_fidelidade INTEGER
);

-- 2. Historico de vendas
CREATE TABLE IF NOT EXISTS vendas (
    id_venda INTEGER PRIMARY KEY,
    cliente_id INTEGER NOT NULL,
    data_cadastro DATE,
    ultima_compra DATE,
    cidade VARCHAR(100),
    segmento VARCHAR(50),
    produto_id VARCHAR(100),
    idade INTEGER,
    quantidade INTEGER,
    valor NUMERIC(12, 2),

    CONSTRAINT fk_vendas_clientes
        FOREIGN KEY (cliente_id)
        REFERENCES clientes(cliente_id)
);

-- 3. Indice para otimizar a integracao
CREATE INDEX IF NOT EXISTS idx_vendas_cliente_id
    ON vendas(cliente_id);

-- 4. Consulta de validacao
SELECT
    (SELECT COUNT(*) FROM clientes) AS total_clientes,
    (SELECT COUNT(*) FROM vendas) AS total_vendas;

-- 5. Validacao de integridade referencial
SELECT
    COUNT(*) AS vendas_sem_cliente
FROM vendas v
LEFT JOIN clientes c
    ON v.cliente_id = c.cliente_id
WHERE c.cliente_id IS NULL;

-- =====================================================
-- RESULTADO ESPERADO APOS A EXECUCAO DO ETL
-- Clientes: 197
-- Vendas: 787
-- Vendas sem cliente: 0
-- =====================================================
