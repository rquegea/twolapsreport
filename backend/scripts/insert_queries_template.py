import os
import json
import psycopg2
import argparse
from typing import List, Tuple, Optional
from dotenv import load_dotenv


# Cargar variables de entorno (útil para DB y defaults de IDs)
load_dotenv()


# Estructura esperada de cada entrada en QUERIES:
# (query_text, brand_name, category, language)
# - query_text: str (obligatorio)
# - brand_name: str (se puede sobreescribir para todas con --brand-override)
# - category: str (ej. "Análisis de Mercado", "Competencia", etc.)
# - language: str (ej. "es", "en")
QUERIES: List[Tuple[str, str, str, str]] = [
    # Ejemplos (elimina y añade los tuyos):
    # ("¿Qué universidades privadas lideran en Ingeniería Informática en España?", "Acme University", "Análisis de Mercado", "es"),
    # ("Comparativa de posgrados en Data Science en Madrid", "Acme University", "Análisis de Competencia", "es"),
]


def parse_int(value: Optional[str]) -> Optional[int]:
    try:
        if value is None:
            return None
        value = str(value).strip()
        return int(value) if value else None
    except Exception:
        return None


def ensure_columns(cur) -> None:
    """Asegura columnas necesarias en modo no destructivo."""
    try:
        cur.execute("ALTER TABLE queries ADD COLUMN IF NOT EXISTS language TEXT DEFAULT 'en';")
    except Exception:
        pass
    try:
        cur.execute("ALTER TABLE queries ADD COLUMN IF NOT EXISTS category TEXT;")
    except Exception:
        pass
    try:
        cur.execute("ALTER TABLE queries ADD COLUMN IF NOT EXISTS client_id INTEGER;")
    except Exception:
        pass
    try:
        cur.execute("ALTER TABLE queries ADD COLUMN IF NOT EXISTS brand_id INTEGER;")
    except Exception:
        pass
    try:
        cur.execute("ALTER TABLE queries ADD COLUMN IF NOT EXISTS project_id INTEGER;")
    except Exception:
        pass


def load_queries_from_json(path: str) -> List[Tuple[str, str, str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Permitir formato: [ {"query":..., "brand":..., "category":..., "language":...}, ... ]
    if isinstance(data, list) and data and isinstance(data[0], dict):
        out = []
        for row in data:
            out.append((
                row.get("query", "").strip(),
                row.get("brand", "").strip(),
                row.get("category", "").strip(),
                row.get("language", "es").strip() or "es",
            ))
        return out
    # O directamente lista de tuplas
    if isinstance(data, list):
        return [tuple(item) for item in data]  # type: ignore
    raise ValueError("Formato JSON no soportado para QUERIES")


def insert_queries(
    queries: List[Tuple[str, str, str, str]],
    *,
    client_id: Optional[int] = None,
    brand_id: Optional[int] = None,
    project_id: Optional[int] = None,
    brand_override: Optional[str] = None,
) -> None:
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5433)),
        database=os.getenv("POSTGRES_DB", "ai_visibility"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    )
    cur = conn.cursor()

    print("ℹ️ Modo no destructivo: no se eliminarán datos existentes.")
    ensure_columns(cur)
    conn.commit()

    print(f"🎯 Insertando/actualizando {len(queries)} queries...")
    inserted = 0
    updated = 0
    for i, (query_text, brand_name, category, lang) in enumerate(queries, 1):
        brand = (brand_override or brand_name or "").strip() or None
        cur.execute("SELECT 1 FROM queries WHERE query = %s", (query_text,))
        exists = cur.fetchone() is not None
        if not exists:
            cur.execute(
                """
                INSERT INTO queries (query, brand, topic, category, language, enabled, client_id, brand_id, project_id)
                VALUES (%s, %s, %s, %s, %s, TRUE, %s, %s, %s)
                """,
                (query_text, brand, category, category, lang, client_id, brand_id, project_id),
            )
            inserted += 1
            print(f"   {i:2d}. {query_text[:80]}... (+)")
        else:
            cur.execute(
                """
                UPDATE queries
                SET category = COALESCE(category, %s),
                    language = COALESCE(language, %s),
                    client_id = COALESCE(client_id, %s),
                    brand_id = COALESCE(brand_id, %s),
                    project_id = COALESCE(project_id, %s),
                    brand = COALESCE(brand, %s),
                    topic = COALESCE(topic, %s)
                WHERE query = %s
                """,
                (category, lang, client_id, brand_id, project_id, brand, category, query_text),
            )
            updated += 1
            print(f"   {i:2d}. {query_text[:80]}... (upd)")

    conn.commit()
    print(f"✅ Insert/Update completado. Nuevas: {inserted} | Actualizadas: {updated}")

    # Mostrar un resumen de queries activas (opcional)
    cur.execute(
        """
        SELECT id, COALESCE(brand,'-') AS brand, COALESCE(category, topic) AS category, LEFT(query, 80)
        FROM queries
        WHERE enabled = TRUE
        ORDER BY id DESC
        LIMIT 20
        """
    )
    rows = cur.fetchall()
    if rows:
        print("\n📌 Últimas queries ACTIVAS:")
        for r in rows:
            print(f"   ID {r[0]:2d}: [{r[1]}] {r[2]} - {r[3]}...")

    cur.close()
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inserta queries para un cliente/mercado")
    parser.add_argument("--client-id", dest="client_id", default=os.getenv("DEFAULT_CLIENT_ID"))
    parser.add_argument("--brand-id", dest="brand_id", default=os.getenv("DEFAULT_BRAND_ID"))
    parser.add_argument("--project-id", dest="project_id", default=os.getenv("DEFAULT_PROJECT_ID"))
    parser.add_argument("--brand-override", dest="brand_override", default=os.getenv("DEFAULT_BRAND"))
    parser.add_argument("--queries-json", dest="queries_json", help="Ruta a JSON con queries (opcional)")
    args = parser.parse_args()

    client_id = parse_int(args.client_id)
    brand_id = parse_int(args.brand_id)
    project_id = parse_int(args.project_id)
    brand_override = args.brand_override.strip() if args.brand_override else None

    queries_local = QUERIES
    if args.queries_json:
        queries_local = load_queries_from_json(args.queries_json)

    if not queries_local:
        print("⚠️ La lista QUERIES está vacía. Añade entradas en el archivo o pasa --queries-json.")
        return

    insert_queries(
        queries_local,
        client_id=client_id,
        brand_id=brand_id,
        project_id=project_id,
        brand_override=brand_override,
    )


if __name__ == "__main__":
    main()


