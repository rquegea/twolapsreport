import os
import sys
import math
import time
import logging
from typing import List, Tuple

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Reutilizamos el cliente OpenAI ya configurado en src/engines/openai_engine.py
from src.engines.openai_engine import client as openai_client


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


DB_CFG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5433)),
    "database": os.getenv("POSTGRES_DB", "ai_visibility"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
}

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 100))
REQUEST_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "25"))
MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", 3))


def fetch_missing_mentions(cur, limit: int) -> List[Tuple[int, str]]:
    cur.execute(
        """
        SELECT id, summary
        FROM mentions
        WHERE embedding IS NULL
        ORDER BY id ASC
        LIMIT %s
        """,
        (limit,),
    )
    return cur.fetchall()


def get_embeddings(texts: List[str]) -> List[List[float]]:
    # OpenAI embeddings API requiere inputs como lista de strings
    attempt = 0
    while True:
        try:
            res = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts, timeout=REQUEST_TIMEOUT)
            vectors = [item.embedding for item in res.data]
            return vectors
        except Exception as exc:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise
            sleep_s = min(2 ** attempt, 10)
            logging.warning("⚠️ Error en embeddings (%s). Reintentando en %ss (intento %s/%s)", exc, sleep_s, attempt, MAX_RETRIES)
            time.sleep(sleep_s)


def update_batch_embeddings(cur, ids: List[int], vectors: List[List[float]]):
    # Convertimos a literal de vector de pgvector: "[v1,v2,...]"
    def to_vector_literal(vec: List[float]) -> str:
        return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"

    rows = [(idx, to_vector_literal(vec)) for idx, vec in zip(ids, vectors)]

    execute_values(
        cur,
        """
        WITH data(id, emb_txt) AS (
            VALUES %s
        )
        UPDATE mentions AS m
        SET embedding = d.emb_txt::vector
        FROM data AS d
        WHERE m.id = d.id
        """,
        rows,
        template="(%s, %s)",
    )


def to_vector_sql_function(cur):
    # Ya no es necesario; mantenemos stub para compatibilidad
    return


def main():
    logging.info("🚀 Iniciando backfill de embeddings (modelo=%s, batch=%s)", EMBEDDING_MODEL, BATCH_SIZE)
    with psycopg2.connect(**DB_CFG) as conn:
        with conn.cursor() as cur:
            # No se requiere función auxiliar; usamos literales ::vector

            total_processed = 0
            batch_index = 0
            while True:
                records = fetch_missing_mentions(cur, BATCH_SIZE)
                if not records:
                    break
                batch_index += 1
                ids = [r[0] for r in records]
                texts = [(r[1] or "") for r in records]
                # Evitar None o strings vacíos: si vacío, usamos un placeholder mínimo
                texts = [t if isinstance(t, str) and t.strip() != "" else "(sin resumen)" for t in texts]

                vectors = get_embeddings(texts)
                if not vectors or len(vectors) != len(ids):
                    raise RuntimeError("La longitud de embeddings no coincide con ids")

                update_batch_embeddings(cur, ids, vectors)
                conn.commit()

                total_processed += len(ids)
                logging.info("Procesado lote %s | filas=%s | total=%s", batch_index, len(ids), total_processed)

    logging.info("✅ Backfill completado. Total de menciones actualizadas: %s", total_processed)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Interrumpido por el usuario")
        sys.exit(130)
    except Exception as exc:
        logging.exception("❌ Error en backfill: %s", exc)
        sys.exit(1)


