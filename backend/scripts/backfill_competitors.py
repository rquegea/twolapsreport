#!/usr/bin/env python3
import os
import json
import argparse
import psycopg2
from typing import List, Dict, Any, Optional
from psycopg2.extras import Json

# Reutilizamos el diccionario de sinónimos de marcas
from src.reports.aggregator import BRAND_SYNONYMS

DB_CFG = dict(
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=int(os.getenv("POSTGRES_PORT", 5433)),
    database=os.getenv("POSTGRES_DB", "ai_visibility"),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD", "postgres"),
)


def detect_competitors(text: str,
                       title: Optional[str],
                       key_topics: Optional[List[str]],
                       insights_payload: Optional[Dict[str, Any]],
                       client_brand: Optional[str]) -> List[str]:
    try:
        resp = (text or "").lower()
        ttl = (title or "").lower()
        topics = [str(t).strip().lower() for t in (key_topics or []) if t]
        payload_brands: List[str] = []
        try:
            if isinstance(insights_payload, dict):
                raw_b = insights_payload.get("brands")
                if isinstance(raw_b, list):
                    for b in raw_b:
                        if isinstance(b, dict):
                            name = (b.get("name") or "").strip().lower()
                            if name:
                                payload_brands.append(name)
                        elif isinstance(b, str):
                            payload_brands.append(b.strip().lower())
        except Exception:
            pass

        client_norm = (client_brand or "").strip().lower()
        norm_syns = {canon: [canon.lower(), *[s.lower() for s in alts]] for canon, alts in (BRAND_SYNONYMS or {}).items()}

        detected: List[str] = []
        for canon, syns in norm_syns.items():
            hit = False
            for s in syns:
                if not s:
                    continue
                if s in topics:
                    hit = True; break
                if s in resp or s in ttl:
                    hit = True; break
                if s in payload_brands:
                    hit = True; break
            if hit and canon.strip().lower() != client_norm:
                detected.append(canon)
        return sorted(set(detected))
    except Exception:
        return []


def ensure_schema(cur) -> None:
    cur.execute("ALTER TABLE mentions ADD COLUMN IF NOT EXISTS competitors JSONB")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mentions_competitors_gin ON mentions USING GIN (competitors)")
    cur.execute("ALTER TABLE mentions ADD COLUMN IF NOT EXISTS brands JSONB")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mentions_brands_gin ON mentions USING GIN (brands)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-id", type=int, help="Limitar al mercado (COALESCE(project_id,id))")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--max-rows", type=int, default=1_000_000)
    args = ap.parse_args()

    with psycopg2.connect(**DB_CFG) as conn:
        conn.autocommit = False
        with conn.cursor() as cur:
            ensure_schema(cur)
            processed = 0
            while processed < args.max_rows:
                where = ["(m.competitors IS NULL OR m.brands IS NULL)"]
                params: Dict[str, Any] = {}
                if args.project_id is not None:
                    where.append("COALESCE(q.project_id, q.id) = %(pid)s")
                    params["pid"] = int(args.project_id)
                sql = f"""
                    SELECT
                      m.id, m.response, m.source_title, m.key_topics,
                      i.payload, COALESCE(q.brand, q.topic, 'Unknown') AS client_brand
                    FROM mentions m
                    JOIN queries q ON q.id = m.query_id
                    LEFT JOIN insights i ON i.id = m.generated_insight_id
                    WHERE {' AND '.join(where)}
                    ORDER BY m.id
                    LIMIT %(lim)s
                """
                params["lim"] = int(args.batch_size)
                cur.execute(sql, params)
                rows = cur.fetchall()
                if not rows:
                    break

                for mid, response, title, key_topics, payload, client_brand in rows:
                    try:
                        payload_json = payload if isinstance(payload, dict) else json.loads(payload) if payload else None
                    except Exception:
                        payload_json = None
                    comps = detect_competitors(
                        text=response or "",
                        title=title,
                        key_topics=key_topics or [],
                        insights_payload=payload_json,
                        client_brand=client_brand,
                    )
                    # construir brands = cliente + competidores detectados
                    brands = sorted(set(([client_brand] if client_brand else []) + comps))
                    cur.execute("UPDATE mentions SET competitors = %s, brands = %s WHERE id = %s", (Json(comps), Json(brands), mid))
                    processed += 1

                conn.commit()
                print(f"✓ Procesadas {processed} menciones...")

    print("✅ Backfill de competitors completado.")


if __name__ == "__main__":
    main()


