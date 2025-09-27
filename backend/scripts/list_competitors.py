#!/usr/bin/env python3
import os
import psycopg2
from typing import Dict, Any

DB_CFG = dict(
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=int(os.getenv("POSTGRES_PORT", 5433)),
    database=os.getenv("POSTGRES_DB", "ai_visibility"),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD", "postgres"),
)


def list_competitors_for_brand(brand_name: str) -> None:
    sql = """
        WITH pid AS (
          SELECT COALESCE(project_id, id) AS project_id
          FROM queries
          WHERE COALESCE(brand, topic, 'Unknown') = %(b)s
          ORDER BY id ASC LIMIT 1
        )
        SELECT DISTINCT jsonb_array_elements_text(COALESCE(m.competitors, '[]'::jsonb)) AS competitor
        FROM mentions m
        JOIN queries q ON q.id = m.query_id
        WHERE COALESCE(q.project_id, q.id) = (SELECT project_id FROM pid)
          AND jsonb_array_length(COALESCE(m.competitors, '[]'::jsonb)) > 0
        ORDER BY 1;
    """
    with psycopg2.connect(**DB_CFG) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"b": brand_name})
            rows = cur.fetchall()
            for (name,) in rows:
                print(name)


if __name__ == "__main__":
    import sys
    brand = sys.argv[1] if len(sys.argv) > 1 else None
    if not brand:
        print("Uso: list_competitors.py 'The Core School'")
        raise SystemExit(1)
    list_competitors_for_brand(brand)


