import os
import sys
import logging

import psycopg2
from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")


DB_CFG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5433)),
    "database": os.getenv("POSTGRES_DB", "ai_visibility"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
}


def check_vector_extension(cur) -> bool:
    cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
    return cur.fetchone() is not None


def check_mentions_embedding_column(cur) -> tuple[bool, str | None]:
    cur.execute(
        """
        SELECT data_type, udt_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'mentions' AND column_name = 'embedding'
        """
    )
    row = cur.fetchone()
    if not row:
        return False, None
    _, udt_name = row
    # En pgvector, information_schema suele mostrar data_type='USER-DEFINED' y udt_name='vector'
    return udt_name == "vector", udt_name


def get_vector_dimension(cur) -> int | None:
    # Método robusto: usa format_type() y parsea "vector(XXXX)"
    try:
        cur.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            WHERE c.relname = 'mentions' AND a.attname = 'embedding'
            """
        )
        row = cur.fetchone()
        if not row or not row[0]:
            return None
        type_str = str(row[0])
        if type_str.startswith("vector(") and type_str.endswith(")"):
            inside = type_str[len("vector("):-1]
            try:
                return int(inside)
            except Exception:
                return None
    except Exception:
        return None
    return None


def get_counts(cur) -> tuple[int, int, int]:
    cur.execute("SELECT COUNT(*) FROM mentions")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM mentions WHERE embedding IS NOT NULL")
    with_embedding = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM mentions WHERE embedding IS NULL")
    pending = cur.fetchone()[0]
    return total, with_embedding, pending


def main() -> int:
    print("🔎 Verificación de base de datos para embeddings (pgvector)")
    try:
        with psycopg2.connect(**DB_CFG) as conn:
            with conn.cursor() as cur:
                print(f"• Base de datos: {DB_CFG['database']}@{DB_CFG['host']}:{DB_CFG['port']}")

                has_vector = check_vector_extension(cur)
                print(f"• Extensión vector activa: {'sí' if has_vector else 'no'}")

                has_col, _ = check_mentions_embedding_column(cur)
                print(f"• Columna mentions.embedding existe: {'sí' if has_col else 'no'}")
                if has_col:
                    dim = get_vector_dimension(cur)
                    dim_text = f"{dim}" if dim is not None else "(desconocida)"
                    print(f"  - Tipo: vector, dimensión: {dim_text}")

                total, with_emb, pending = get_counts(cur)
                print(f"• Total menciones: {total}")
                print(f"• Con embedding: {with_emb}")
                print(f"• Pendientes (embedding IS NULL): {pending}")

                # Resultado final
                ok = has_vector and has_col and (get_vector_dimension(cur) in (1536, None))
                # Permitimos None en dimensión si el sistema no pudo consultarla; no bloquea
                print("✅ Listo para backfill" if ok else "⚠️ Faltan requisitos para backfill")
                return 0 if ok else 2
    except Exception as exc:
        print(f"❌ Error de verificación: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


