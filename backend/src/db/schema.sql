-- backend/src/db/schema.sql (versión mejorada)

CREATE TABLE IF NOT EXISTS queries (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    brand TEXT,
    topic TEXT,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    language TEXT DEFAULT 'en'
);

CREATE TABLE IF NOT EXISTS mentions (
    id SERIAL PRIMARY KEY,
    query_id INTEGER REFERENCES queries(id) ON DELETE CASCADE,
    engine TEXT NOT NULL,
    source TEXT,
    response TEXT NOT NULL,
    sentiment FLOAT,
    emotion TEXT,
    confidence_score REAL,
    source_url TEXT,
    source_title TEXT,
    language TEXT DEFAULT 'unknown',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- CAMPOS ENRIQUECIDOS PARA UN FRONTEND SUPERIOR --
    summary TEXT,                         -- Resumen generado por IA (1-2 frases).
    key_topics JSONB,                     -- Lista JSON con los temas/marcas clave.
    generated_insight_id INTEGER,        -- Enlace al insight detallado (si se generó).

    -- NUEVOS CAMPOS PARA GESTIÓN Y OBSERVABILIDAD --
    status TEXT DEFAULT 'active' CHECK (status IN ('active','archived','ignored','flagged')),
    is_bot BOOLEAN DEFAULT FALSE,
    spam_score REAL DEFAULT 0,
    duplicate_group_id TEXT,
    alert_triggered BOOLEAN DEFAULT FALSE,
    alert_reason TEXT,
    engine_latency_ms INTEGER,
    error TEXT,

    -- v3 Observabilidad extendida y enriquecimiento de fuentes
    model_name TEXT,
    api_status_code INTEGER,
    engine_request_id TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    price_usd NUMERIC(10,4),
    analysis_latency_ms INTEGER,
    total_pipeline_ms INTEGER,
    error_category TEXT,
    source_domain TEXT,
    source_rank INTEGER,
    query_text TEXT,
    query_topic TEXT
);

CREATE TABLE IF NOT EXISTS insights (
    id SERIAL PRIMARY KEY,
    query_id INTEGER REFERENCES queries(id) ON DELETE CASCADE,
    payload JSONB NOT NULL,
    -- ... (el resto de tu tabla insights se mantiene igual)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Índices para optimizar búsquedas
CREATE INDEX IF NOT EXISTS idx_mentions_query_id ON mentions(query_id);
CREATE INDEX IF NOT EXISTS idx_mentions_created_at ON mentions(created_at);
CREATE INDEX IF NOT EXISTS idx_mentions_engine ON mentions(engine);
CREATE INDEX IF NOT EXISTS idx_mentions_source ON mentions(source);
CREATE INDEX IF NOT EXISTS idx_mentions_sentiment ON mentions(sentiment);
CREATE INDEX IF NOT EXISTS idx_mentions_status ON mentions(status);
CREATE INDEX IF NOT EXISTS idx_mentions_is_bot ON mentions(is_bot);
CREATE INDEX IF NOT EXISTS idx_mentions_key_topics_gin ON mentions USING GIN (key_topics);
CREATE INDEX IF NOT EXISTS idx_mentions_model_name ON mentions(model_name);
CREATE INDEX IF NOT EXISTS idx_mentions_source_domain ON mentions(source_domain);
CREATE INDEX IF NOT EXISTS idx_mentions_query_topic ON mentions(query_topic);
CREATE INDEX IF NOT EXISTS idx_insights_query_id ON insights(query_id);

-- FK explícita para generated_insight_id (compatible sin IF NOT EXISTS)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'fk_mentions_generated_insight'
          AND table_name = 'mentions'
    ) THEN
        ALTER TABLE mentions
            ADD CONSTRAINT fk_mentions_generated_insight
            FOREIGN KEY (generated_insight_id)
            REFERENCES insights(id)
            ON DELETE SET NULL;
    END IF;
END $$;