import os
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import re

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from ..insight_analysis import summarize_agent_insights
from math import sqrt
from typing import TypedDict
import numpy as np

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover
    KMeans = None  # type: ignore
from src.engines.openai_engine import fetch_response

# Mapeo de sinónimos de marcas (alineado con backend/app.py)
BRAND_SYNONYMS: Dict[str, List[str]] = {
    "The Core School": ["the core", "the core school", "thecore"],
    "U-TAD": ["u-tad", "utad"],
    "ECAM": ["ecam"],
    "TAI": ["tai"],
    "CES": ["ces"],
    "CEV": ["cev"],
    "FX Barcelona Film School": ["fx barcelona", "fx barcelona film school", "fx animation"],
    "Septima Ars": ["septima ars", "séptima ars"],
}


def _db_url() -> str:
    host = os.getenv("POSTGRES_HOST", os.getenv("DB_HOST", "localhost"))
    port = int(os.getenv("POSTGRES_PORT", os.getenv("DB_PORT", 5433)))
    db = os.getenv("POSTGRES_DB", os.getenv("DB_NAME", "ai_visibility"))
    user = os.getenv("POSTGRES_USER", os.getenv("DB_USER", "postgres"))
    pwd = os.getenv("POSTGRES_PASSWORD", os.getenv("DB_PASSWORD", "postgres"))
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"


_ENGINE = None
_SessionLocal = None


def _engine():
    global _ENGINE, _SessionLocal
    if _ENGINE is None:
        _ENGINE = create_engine(_db_url(), pool_pre_ping=True, future=True)
        _SessionLocal = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False, future=True)
    return _ENGINE


def get_session() -> Session:
    _engine()
    return _SessionLocal()  # type: ignore[call-arg]


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _format_date(date_obj: datetime) -> str:
    return date_obj.strftime("%Y-%m-%d")


def _compute_previous_period(start_date: str, end_date: str) -> Tuple[str, str]:
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    duration_days = (end_dt - start_dt).days + 1
    prev_end = start_dt - timedelta(days=1)
    prev_start = prev_end - timedelta(days=duration_days - 1)
    return _format_date(prev_start), _format_date(prev_end)


def _compile_brand_patterns(brands: List[str]) -> List[Tuple[str, re.Pattern]]:
    patterns: List[Tuple[str, re.Pattern]] = []
    for brand in brands:
        if not brand:
            continue
        escaped = re.escape(brand.strip())
        patterns.append((brand, re.compile(rf"(?i)(?<!\w){escaped}(?!\w)")))
    return patterns


def _detect_brands(text: Optional[str], brands: List[str]) -> List[str]:
    if not text:
        return []
    found = set()
    for original, pattern in _compile_brand_patterns(brands):
        if pattern.search(text or ""):
            found.add(original)
    return list(found)


def get_kpi_summary(
    session: Optional[Session],
    project_id: int,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    client_brand: Optional[str] = None,
) -> Dict:
    # Aislamiento por mercado: agrupar por COALESCE(project_id, id)
    where: List[str] = [
        "COALESCE(q.project_id, q.id) = COALESCE((SELECT project_id FROM queries WHERE id = :project_id), :project_id)"
    ]
    params: Dict[str, Any] = {"project_id": project_id}
    if start_date:
        where.append("m.created_at >= CAST(:start_date AS date)")
        params["start_date"] = start_date
    if end_date:
        where.append("m.created_at < (CAST(:end_date AS date) + INTERVAL '1 day')")
        params["end_date"] = end_date

    sql = text(
        f"""
        SELECT COUNT(*) AS total_mentions,
               AVG(m.sentiment) AS sentiment_avg
        FROM mentions m
        JOIN queries q ON q.id = m.query_id
        WHERE {' AND '.join(where)}
        """
    )
    sov_sql = text(
        """
        WITH scoped AS (
            SELECT m.id, COALESCE(q.brand, q.topic, 'Unknown') AS brand
            FROM mentions m
            JOIN queries q ON q.id = m.query_id
            WHERE COALESCE(q.project_id, q.id) = COALESCE((SELECT project_id FROM queries WHERE id = :project_id), :project_id)
        ), brand_counts AS (
            SELECT brand, COUNT(*) AS cnt
            FROM scoped
            GROUP BY brand
        )
        SELECT brand, cnt FROM brand_counts ORDER BY cnt DESC
        """
    )
    own_session = False
    if session is None:
        session = get_session()
        own_session = True
    try:
        row = session.execute(sql, params).mappings().first()
        totals = dict(row) if row else {"total_mentions": 0, "sentiment_avg": None}
        sov_rows = session.execute(sov_sql, params).mappings().all()
    finally:
        if own_session:
            session.close()
    total_all = sum(r["cnt"] for r in sov_rows) or 1
    proj_brand_sql = text("SELECT COALESCE(brand, topic, 'Unknown') AS b FROM queries WHERE id=:pid")
    s2 = get_session()
    try:
        brow = s2.execute(proj_brand_sql, {"pid": project_id}).first()
    finally:
        s2.close()
    brand_name = (brow[0] if brow else "Unknown")
    if client_brand and str(client_brand).strip():
        brand_name = str(client_brand).strip()
    brand_cnt = next((r["cnt"] for r in sov_rows if r["brand"] == brand_name), 0)
    sov_pct = round(100.0 * brand_cnt / total_all, 1)

    # Si se solicita filtrar KPIs por cliente, recalcular total y sentimiento para esa marca
    if client_brand and str(client_brand).strip():
        session2 = get_session()
        try:
            # Preparar sinónimos del cliente
            syns = [brand_name.lower()] + [s.lower() for s in BRAND_SYNONYMS.get(brand_name, [])]
            likes = [f"%{s}%" for s in syns]
            brand_sql = text(
                f"""
                WITH rows AS (
                    SELECT 
                        COALESCE(m.sentiment, 0) AS sent,
                        (
                            EXISTS (
                                SELECT 1 FROM jsonb_array_elements_text(COALESCE(to_jsonb(m.key_topics),'[]'::jsonb)) kt
                                WHERE LOWER(TRIM(kt)) = ANY(:syns)
                            )
                            OR LOWER(COALESCE(m.response,'')) LIKE ANY(:likes)
                        OR LOWER(COALESCE(m.source_title,'')) LIKE ANY(:likes)
                            OR EXISTS (
                                SELECT 1 FROM jsonb_array_elements(COALESCE(i.payload->'brands','[]'::jsonb)) b
                                WHERE LOWER(TRIM(CASE WHEN jsonb_typeof(b)='object' THEN COALESCE(b->>'name','') ELSE TRIM(BOTH '"' FROM b::text) END)) = ANY(:syns)
                            )
                        ) AS is_brand
                    FROM mentions m
                    JOIN queries q ON q.id = m.query_id
                    LEFT JOIN insights i ON i.id = m.generated_insight_id
                    WHERE {' AND '.join(where)}
                )
                SELECT 
                    SUM(CASE WHEN is_brand THEN 1 ELSE 0 END) AS brand_cnt,
                    AVG(CASE WHEN is_brand THEN sent ELSE NULL END) AS brand_avg
                FROM rows
                """
            )
            b_row = session2.execute(brand_sql, {**params, "syns": syns, "likes": likes}).first()
            if b_row is not None:
                totals["total_mentions"] = int(b_row[0] or 0)
                totals["sentiment_avg"] = float(b_row[1] or 0.0)
        finally:
            session2.close()
    return {
        "total_mentions": int(totals.get("total_mentions") or 0),
        "sentiment_avg": float(totals.get("sentiment_avg") or 0.0),
        "sov": sov_pct,
        "brand_name": brand_name,
        "sov_table": [(r["brand"], r["cnt"]) for r in sov_rows],
    }


def get_sentiment_evolution(
    session: Optional[Session],
    project_id: int,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    client_brand: Optional[str] = None,
) -> List[Tuple[str, float]]:
    # Si llega client_brand, calcular la media diaria SOLO de menciones de esa marca
    if client_brand and str(client_brand).strip():
        params: Dict[str, Any] = {"project_id": project_id}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        # Resolver sinónimos
        base = str(client_brand).strip()
        syns = [base.lower()] + [s.lower() for s in BRAND_SYNONYMS.get(base, [])]
        likes = [f"%{s}%" for s in syns]
        sql = text(
            """
            WITH rows AS (
                SELECT DATE_TRUNC('day', m.created_at)::date AS d,
                       COALESCE(m.sentiment, 0) AS sent,
                       (
                         EXISTS (
                           SELECT 1 FROM jsonb_array_elements_text(COALESCE(to_jsonb(m.key_topics),'[]'::jsonb)) kt
                           WHERE LOWER(TRIM(kt)) = ANY(:syns)
                         )
                         OR LOWER(COALESCE(m.response,'')) LIKE ANY(:likes)
                          OR LOWER(COALESCE(m.source_title,'')) LIKE ANY(:likes)
                          OR LOWER(TRIM(COALESCE(q.brand, q.topic, ''))) = ANY(:syns)
                         OR EXISTS (
                           SELECT 1 FROM jsonb_array_elements(COALESCE(i.payload->'brands','[]'::jsonb)) b
                           WHERE LOWER(TRIM(CASE WHEN jsonb_typeof(b)='object' THEN COALESCE(b->>'name','') ELSE TRIM(BOTH '"' FROM b::text) END)) = ANY(:syns)
                         )
                       ) AS is_brand
                FROM mentions m
                JOIN queries q ON q.id = m.query_id
                LEFT JOIN insights i ON i.id = m.generated_insight_id
                WHERE COALESCE(q.project_id, q.id) = COALESCE((SELECT project_id FROM queries WHERE id = :project_id), :project_id)
                  AND m.created_at >= CAST(COALESCE(:start_date, '1970-01-01') AS date)
                  AND m.created_at < (CAST(COALESCE(:end_date, '2999-12-31') AS date) + INTERVAL '1 day')
            )
            SELECT d, AVG(CASE WHEN is_brand THEN sent ELSE NULL END) AS avg_s
            FROM rows
            GROUP BY d
            ORDER BY d
            """
        )
        rows = session.execute(sql, {"project_id": project_id, "syns": syns, "likes": likes, "start_date": start_date, "end_date": end_date}).all()
        return [(r[0].strftime("%Y-%m-%d"), float(r[1] or 0.0)) for r in rows]
    # Sin client_brand: media diaria del mercado completo
    where: List[str] = [
        "COALESCE(q.project_id, q.id) = COALESCE((SELECT project_id FROM queries WHERE id = :project_id), :project_id)"
    ]
    params: Dict[str, Any] = {"project_id": project_id}
    if start_date:
        where.append("m.created_at >= CAST(:start_date AS date)")
        params["start_date"] = start_date
    if end_date:
        where.append("m.created_at < (CAST(:end_date AS date) + INTERVAL '1 day')")
        params["end_date"] = end_date
    sql = text(
        f"""
        SELECT DATE_TRUNC('day', m.created_at)::date AS d,
               AVG(m.sentiment) AS avg_s
        FROM mentions m
        JOIN queries q ON q.id = m.query_id
        WHERE {' AND '.join(where)}
        GROUP BY 1
        ORDER BY 1
        """
    )
    own_session = False
    if session is None:
        session = get_session()
        own_session = True
    try:
        rows = session.execute(sql, params).all()
    finally:
        if own_session:
            session.close()
    return [(r[0].strftime("%Y-%m-%d"), float(r[1] or 0.0)) for r in rows]


def get_sentiment_by_category(
    session: Optional[Session],
    project_id: int,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    client_brand: Optional[str] = None,
) -> Dict[str, float]:
    if client_brand and str(client_brand).strip():
        base = str(client_brand).strip()
        syns = [base.lower()] + [s.lower() for s in BRAND_SYNONYMS.get(base, [])]
        likes = [f"%{s}%" for s in syns]
        sql = text(
            """
            WITH rows AS (
                SELECT COALESCE((i.payload->>'category'), COALESCE(q.category, q.topic, 'Desconocida')) AS cat,
                       COALESCE(m.sentiment, 0) AS sent,
                       (
                         EXISTS (
                           SELECT 1 FROM jsonb_array_elements_text(COALESCE(to_jsonb(m.key_topics),'[]'::jsonb)) kt
                           WHERE LOWER(TRIM(kt)) = ANY(:syns)
                         )
                         OR LOWER(COALESCE(m.response,'')) LIKE ANY(:likes)
                         OR LOWER(COALESCE(m.source_title,'')) LIKE ANY(:likes)
                         OR EXISTS (
                           SELECT 1 FROM jsonb_array_elements(COALESCE(i.payload->'brands','[]'::jsonb)) b
                           WHERE LOWER(TRIM(CASE WHEN jsonb_typeof(b)='object' THEN COALESCE(b->>'name','') ELSE TRIM(BOTH '"' FROM b::text) END)) = ANY(:syns)
                         )
                       ) AS is_brand
                FROM mentions m
                JOIN queries q ON q.id = m.query_id
                LEFT JOIN insights i ON i.id = m.generated_insight_id
                WHERE COALESCE(q.project_id, q.id) = COALESCE((SELECT project_id FROM queries WHERE id = :project_id), :project_id)
                  AND m.created_at >= CAST(COALESCE(:start_date, '1970-01-01') AS date)
                  AND m.created_at < (CAST(COALESCE(:end_date, '2999-12-31') AS date) + INTERVAL '1 day')
            )
            SELECT cat, AVG(CASE WHEN is_brand THEN sent ELSE NULL END) AS avg_s
            FROM rows
            GROUP BY cat
            ORDER BY cat
            """
        )
        own_session = False
        if session is None:
            session = get_session()
            own_session = True
        try:
            rows = session.execute(sql, {"project_id": project_id, "start_date": start_date, "end_date": end_date, "syns": syns, "likes": likes}).all()
        finally:
            if own_session:
                session.close()
        return {str(r[0]): float(r[1] or 0.0) for r in rows}
    # Sin client_brand: promedio por categoría del mercado
    where: List[str] = [
        "COALESCE(q.project_id, q.id) = COALESCE((SELECT project_id FROM queries WHERE id = :project_id), :project_id)"
    ]
    params: Dict[str, Any] = {"project_id": project_id}
    if start_date:
        where.append("m.created_at >= CAST(:start_date AS date)")
        params["start_date"] = start_date
    if end_date:
        where.append("m.created_at < (CAST(:end_date AS date) + INTERVAL '1 day')")
        params["end_date"] = end_date
    sql = text(
        f"""
        SELECT
          COALESCE((i.payload->>'category'), COALESCE(q.category, q.topic, 'Desconocida')) AS cat,
          AVG(m.sentiment) AS avg_s
        FROM mentions m
        JOIN queries q ON q.id = m.query_id
        LEFT JOIN insights i ON i.id = m.generated_insight_id
        WHERE {' AND '.join(where)}
        GROUP BY 1
        ORDER BY 1
        """
    )
    own_session = False
    if session is None:
        session = get_session()
        own_session = True
    try:
        rows = session.execute(sql, params).all()
    finally:
        if own_session:
            session.close()
    return {str(r[0]): float(r[1] or 0.0) for r in rows}


def get_topics_by_sentiment(
    session: Optional[Session],
    project_id: int,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    client_brand: Optional[str] = None,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    if client_brand and str(client_brand).strip():
        base = str(client_brand).strip()
        syns = [base.lower()] + [s.lower() for s in BRAND_SYNONYMS.get(base, [])]
        likes = [f"%{s}%" for s in syns]
        sql = text(
            """
            WITH rows AS (
                SELECT jsonb_array_elements_text(COALESCE(m.key_topics, '[]'::jsonb)) AS topic,
                       COALESCE(m.sentiment, 0) AS sent,
                       (
                         EXISTS (
                           SELECT 1 FROM jsonb_array_elements_text(COALESCE(to_jsonb(m.key_topics),'[]'::jsonb)) kt
                           WHERE LOWER(TRIM(kt)) = ANY(:syns)
                         )
                         OR LOWER(COALESCE(m.response,'')) LIKE ANY(:likes)
                         OR LOWER(COALESCE(m.source_title,'')) LIKE ANY(:likes)
                         OR EXISTS (
                           SELECT 1 FROM jsonb_array_elements(COALESCE(i.payload->'brands','[]'::jsonb)) b
                           WHERE LOWER(TRIM(CASE WHEN jsonb_typeof(b)='object' THEN COALESCE(b->>'name','') ELSE TRIM(BOTH '"' FROM b::text) END)) = ANY(:syns)
                         )
                       ) AS is_brand
                FROM mentions m
                JOIN queries q ON q.id = m.query_id
                LEFT JOIN insights i ON i.id = m.generated_insight_id
                WHERE COALESCE(q.project_id, q.id) = COALESCE((SELECT project_id FROM queries WHERE id = :project_id), :project_id)
                  AND m.created_at >= CAST(COALESCE(:start_date, '1970-01-01') AS date)
                  AND m.created_at < (CAST(COALESCE(:end_date, '2999-12-31') AS date) + INTERVAL '1 day')
            )
            SELECT topic, AVG(CASE WHEN is_brand THEN sent ELSE NULL END) AS avg_s, COUNT(*) AS cnt
            FROM rows
            GROUP BY topic
            HAVING COUNT(*) >= 3
            """
        )
        own_session = False
        if session is None:
            session = get_session()
            own_session = True
        try:
            rows = session.execute(sql, {"project_id": project_id, "start_date": start_date, "end_date": end_date, "syns": syns, "likes": likes}).all()
        finally:
            if own_session:
                session.close()
        arr = [(str(r[0]), float(r[1] or 0.0)) for r in rows]
        arr.sort(key=lambda x: x[1])
        bottom5 = arr[:5]
        top5 = arr[-5:][::-1]
        return top5, bottom5
    where: List[str] = [
        "COALESCE(q.project_id, q.id) = COALESCE((SELECT project_id FROM queries WHERE id = :project_id), :project_id)"
    ]
    params: Dict[str, Any] = {"project_id": project_id}
    if start_date:
        where.append("m.created_at >= CAST(:start_date AS date)")
        params["start_date"] = start_date
    if end_date:
        where.append("m.created_at < (CAST(:end_date AS date) + INTERVAL '1 day')")
        params["end_date"] = end_date
    sql = text(
        f"""
        SELECT jsonb_array_elements_text(COALESCE(m.key_topics, '[]'::jsonb)) AS topic,
               AVG(m.sentiment) AS avg_s,
               COUNT(*) AS cnt
        FROM mentions m
        JOIN queries q ON q.id = m.query_id
        WHERE {' AND '.join(where)}
        GROUP BY 1
        HAVING COUNT(*) >= 3
        """
    )
    own_session = False
    if session is None:
        session = get_session()
        own_session = True
    try:
        rows = session.execute(sql, params).all()
    finally:
        if own_session:
            session.close()
    arr = [(str(r[0]), float(r[1] or 0.0)) for r in rows]
    arr.sort(key=lambda x: x[1])
    bottom5 = arr[:5]
    top5 = arr[-5:][::-1]
    return top5, bottom5


def get_share_of_voice_and_trends(
    session: Optional[Session],
    project_id: int,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    client_brand: Optional[str] = None,
) -> Dict:
    """
    Calcula SOV total y por categoría con detección de marcas en resúmenes, y deltas entre
    periodo actual y anterior. Si no se proporcionan fechas, usa todo el histórico.
    """
    own_session = False
    if session is None:
        session = get_session()
        own_session = True
    try:
        periods: List[Tuple[str, str]] = []
        if start_date and end_date:
            prev_start, prev_end = _compute_previous_period(start_date, end_date)
            periods = [(start_date, end_date), (prev_start, prev_end)]
        else:
            periods = [("1970-01-01", "2999-12-31")]

        if client_brand is None:
            brow = session.execute(text("SELECT COALESCE(brand, topic, 'Unknown') AS b FROM queries WHERE id=:pid"), {"pid": project_id}).first()
            client_brand = (brow[0] if brow else "Unknown")

        sql = text(
            """
            SELECT
              COALESCE((i.payload->>'category'), COALESCE(q.category, q.topic, 'Desconocida')) AS category,
              m.key_topics,
              LOWER(COALESCE(m.response,'')) AS resp,
              LOWER(COALESCE(m.source_title,'')) AS title,
              i.payload,
              m.summary,
              m.sentiment,
              DATE_TRUNC('day', m.created_at)::date AS d
            FROM mentions m
            JOIN queries q ON q.id = m.query_id
            LEFT JOIN insights i ON i.id = m.generated_insight_id
            WHERE COALESCE(q.project_id, q.id) = COALESCE((SELECT project_id FROM queries WHERE id = :project_id), :project_id)
              AND m.created_at >= CAST(:start_date AS date)
              AND m.created_at < (CAST(:end_date AS date) + INTERVAL '1 day')
            """
        )

        brand_counts_by_period: List[Dict] = []
        competitors_global: Counter = Counter()
        # Preparar sinónimos en minúsculas para detección robusta
        norm_synonyms: Dict[str, List[str]] = {
            canon: [canon.lower()] + [s.lower() for s in alts]
            for canon, alts in BRAND_SYNONYMS.items()
        }

        for (p_start, p_end) in periods:
            rows = session.execute(sql, {"project_id": project_id, "start_date": p_start, "end_date": p_end}).mappings().all()
            per_category = defaultdict(lambda: {"client": 0, "total": 0, "competitors": Counter()})
            comp_mentions = Counter()
            sentiment_sum = 0.0
            sentiment_count = 0

            for r in rows:
                category = str(r["category"]) if r["category"] is not None else "Desconocida"
                key_topics = r.get("key_topics") or []
                resp = (r.get("resp") or "").lower()
                title = (r.get("title") or "").lower()
                payload = r.get("payload") or {}
                sent = float(r.get("sentiment") or 0.0)

                detected: List[str] = []
                try:
                    # key_topics: coincidencia exacta con sinónimos normalizados
                    try_topics = [str(t).strip().lower() for t in (key_topics or [])]
                    # payload.brands: puede ser lista de objetos o strings
                    payload_brands: List[str] = []
                    try:
                        raw_b = payload.get('brands') if isinstance(payload, dict) else []
                        if isinstance(raw_b, list):
                            for b in raw_b:
                                if isinstance(b, dict):
                                    name = (b.get('name') or '').strip().lower()
                                    if name:
                                        payload_brands.append(name)
                                elif isinstance(b, str):
                                    payload_brands.append(b.strip().lower())
                    except Exception:
                        pass

                    for canon, syns in norm_synonyms.items():
                        hit = False
                        for s in syns:
                            if s in try_topics:
                                hit = True; break
                            if s and (s in resp or s in title):
                                hit = True; break
                            if s in payload_brands:
                                hit = True; break
                        if hit:
                            detected.append(canon)
                except Exception:
                    detected = []

                if detected:
                    seen = set()
                    for b in detected:
                        if b in seen:
                            continue
                        seen.add(b)
                        if b == client_brand:
                            per_category[category]["client"] += 1
                        else:
                            comp_mentions[b] += 1
                            per_category[category]["competitors"][b] += 1
                        per_category[category]["total"] += 1

                sentiment_sum += sent
                sentiment_count += 1

            total_client = sum(v["client"] for v in per_category.values())
            total_comp = sum(comp_mentions.values())
            total_with_brands = float(total_client + total_comp)
            sov_total = (total_client / total_with_brands * 100.0) if total_with_brands > 0 else 0.0
            avg_sent = (sentiment_sum / float(sentiment_count)) if sentiment_count > 0 else 0.0

            brand_counts_by_period.append({
                "period": {"start": p_start, "end": p_end},
                "sov_by_category": {k: {"client": int(v["client"]), "total": int(v["total"]), "competitors": dict(v["competitors"]) } for k, v in per_category.items()},
                "competitor_mentions": dict(comp_mentions),
                "share_of_voice": sov_total,
                "average_sentiment": avg_sent,
            })
            competitors_global.update(comp_mentions)

        current = brand_counts_by_period[0]
        trends: Dict = {}
        if len(brand_counts_by_period) == 2:
            prev = brand_counts_by_period[1]
            def pct(entry: Dict) -> float:
                total_local = float(entry.get("total", 0))
                client_local = float(entry.get("client", 0))
                return (client_local / total_local * 100.0) if total_local > 0 else 0.0

            sov_delta_by_category: Dict[str, float] = {}
            for cat, entry in current.get("sov_by_category", {}).items():
                sov_delta_by_category[cat] = pct(entry) - pct(prev.get("sov_by_category", {}).get(cat, {}))

            competitor_mentions_delta = {b: int(current.get("competitor_mentions", {}).get(b, 0)) - int(prev.get("competitor_mentions", {}).get(b, 0)) for b in set(list(current.get("competitor_mentions", {}).keys()) + list(prev.get("competitor_mentions", {}).keys()))}

            trends = {
                "sentiment_delta_total": float(current.get("average_sentiment", 0.0)) - float(prev.get("average_sentiment", 0.0)),
                "sov_delta_by_category": sov_delta_by_category,
                "competitor_mentions_delta": competitor_mentions_delta,
            }

        return {
            "client_brand": client_brand,
            "current": current,
            "trends": trends,
            "competitors_seen": dict(competitors_global),
        }
    finally:
        if own_session:
            session.close()


def get_visibility_series(
    session: Optional[Session],
    project_id: int,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    client_brand: Optional[str] = None,
) -> tuple[list[str], list[float]]:
    """
    Serie diaria EXACTA de visibilidad (%) replicando /api/visibility (granularity=day).
    Devuelve (dates[], values[]) donde values son porcentajes 0–100 por día.
    """
    own_session = False
    if session is None:
        session = get_session()
        own_session = True
    try:
        # Resolver marca principal
        if client_brand is None or not str(client_brand).strip():
            brow = session.execute(text("SELECT COALESCE(brand, topic, 'Unknown') AS b FROM queries WHERE id=:pid"), {"pid": int(project_id)}).first()
            client_brand = (brow[0] if brow else "Unknown")

        # Ventana temporal
        if not start_date or not end_date:
            # Por defecto: últimos 30 días ventana cerrada
            from datetime import datetime as _dt, timedelta as _td
            end_dt = _dt.utcnow().date()
            start_dt = (end_dt - _td(days=29))
            start_date = start_dt.strftime("%Y-%m-%d")
            end_date = end_dt.strftime("%Y-%m-%d")

        # Preparar sinónimos + patrones LIKE
        syns = [str(client_brand or "Unknown").lower()] + [s.lower() for s in BRAND_SYNONYMS.get(str(client_brand or "Unknown"), [])]
        likes = [f"%{s}%" for s in syns]

        # Cálculo correcto: % de queries del día donde la marca aparece al menos 1 vez
        # Paso 1: marcar por mención si hay marca; Paso 2: colapsar a (día, query_id) con ANY marca
        # Paso 3: por día, contar queries con marca / total de queries
        sql = text(
            """
            WITH rows AS (
                SELECT 
                    DATE(m.created_at) AS d,
                    m.query_id AS qid,
                    (
                        EXISTS (
                            SELECT 1 FROM jsonb_array_elements_text(COALESCE(to_jsonb(m.key_topics),'[]'::jsonb)) kt
                            WHERE LOWER(TRIM(kt)) = ANY(:syns)
                        )
                        OR LOWER(COALESCE(m.response,'')) LIKE ANY(:likes)
                        OR LOWER(COALESCE(m.source_title,'')) LIKE ANY(:likes)
                        OR LOWER(TRIM(COALESCE(q.brand, q.topic, ''))) = ANY(:syns)
                        OR EXISTS (
                            SELECT 1 FROM jsonb_array_elements(COALESCE(i.payload->'brands','[]'::jsonb)) b
                            WHERE LOWER(TRIM(CASE WHEN jsonb_typeof(b)='object' THEN COALESCE(b->>'name','') ELSE TRIM(BOTH '"' FROM b::text) END)) = ANY(:syns)
                        )
                    ) AS is_brand
                FROM mentions m
                JOIN queries q ON q.id = m.query_id
                LEFT JOIN insights i ON i.id = m.generated_insight_id
                WHERE COALESCE(q.project_id, q.id) = COALESCE((SELECT project_id FROM queries WHERE id = :pid), :pid)
                  AND m.created_at >= CAST(:start_date AS date)
                  AND m.created_at < (CAST(:end_date AS date) + INTERVAL '1 day')
            ), per_query AS (
                SELECT d, qid, MAX(CASE WHEN is_brand THEN 1 ELSE 0 END) AS any_brand
                FROM rows
                GROUP BY d, qid
            )
            SELECT d,
                   SUM(any_brand) AS brand_cnt,
                   COUNT(*) AS total_cnt
            FROM per_query
            GROUP BY d
            ORDER BY d
            """
        )

        rows = session.execute(sql, {"pid": int(project_id), "start_date": start_date, "end_date": end_date, "syns": syns, "likes": likes}).all()

        # Mapear resultados por día
        by_day: Dict[str, tuple[int, int]] = {str(d): (int(b or 0), int(t or 0)) for d, b, t in rows}

        # Rellenar todos los días en el rango
        from datetime import datetime as _dt
        from datetime import timedelta as _td
        start_dt = _dt.strptime(start_date, "%Y-%m-%d").date()
        end_dt = _dt.strptime(end_date, "%Y-%m-%d").date()
        dates: list[str] = []
        values: list[float] = []
        day = start_dt
        while day <= end_dt:
            key = day.strftime("%Y-%m-%d")
            brand_cnt, total_cnt = by_day.get(key, (0, 0))
            pct = (float(brand_cnt) / float(max(total_cnt, 1))) * 100.0
            dates.append(key)
            values.append(round(pct, 1))
            day += _td(days=1)
        return dates, values
    finally:
        if own_session:
            session.close()


def get_industry_sov_ranking(
    session: Optional[Session],
    project_id: Optional[int] = None,
    *,
    start_date: Optional[str],
    end_date: Optional[str],
) -> list[tuple[str, float]]:
    """Ranking SOV global: porcentaje por marca sobre total de menciones con marca detectada."""
    own_session = False
    if session is None:
        session = get_session()
        own_session = True
    try:
        # Traer también key_topics para replicar la detección del endpoint
        where = [
            "m.created_at >= CAST(:start_date AS date)",
            "m.created_at < (CAST(:end_date AS date) + INTERVAL '1 day')",
        ]
        params: Dict[str, Any] = {"start_date": start_date or "1970-01-01", "end_date": end_date or "2999-12-31"}
        if project_id is not None:
            where.append("COALESCE(q.project_id, q.id) = COALESCE((SELECT project_id FROM queries WHERE id = :pid), :pid)")
            params["pid"] = int(project_id)
        sql = text(
            f"""
            SELECT
              m.key_topics,
              LOWER(COALESCE(m.response,'')) AS resp,
              LOWER(COALESCE(m.source_title,'')) AS title,
              i.payload
            FROM mentions m
            JOIN queries q ON q.id = m.query_id
            LEFT JOIN insights i ON i.id = m.generated_insight_id
            WHERE {' AND '.join(where)}
            """
        )
        rows = session.execute(sql, params).mappings().all()
        from collections import Counter
        counts: Counter[str] = Counter()
        # Sinónimos normalizados (canónico -> lista de variantes en minúsculas)
        norm_synonyms: Dict[str, list[str]] = {c: [c.lower()] + [s.lower() for s in alts] for c, alts in BRAND_SYNONYMS.items()}
        for r in rows:
            payload = r.get("payload") or {}
            resp = (r.get("resp") or "").lower()
            title = (r.get("title") or "").lower()
            # key_topics puede ser TEXT[] o JSONB; normalizamos a lista en minúsculas
            try_topics: list[str] = []
            try:
                if isinstance(r.get("key_topics"), (list, tuple)):
                    try_topics = [str(t).strip().lower() for t in (r.get("key_topics") or [])]
            except Exception:
                try_topics = []
            payload_brands: list[str] = []
            if isinstance(payload, dict):
                raw_b = payload.get("brands")
                if isinstance(raw_b, list):
                    for b in raw_b:
                        if isinstance(b, dict):
                            name = (b.get("name") or "").strip().lower()
                            if name:
                                payload_brands.append(name)
                        elif isinstance(b, str):
                            payload_brands.append(b.strip().lower())
            seen = set()
            for canon, syns in norm_synonyms.items():
                if canon in seen:
                    continue
                for s in syns:
                    if s in try_topics or (s and (s in resp or s in title or s in payload_brands)):
                        counts[canon] += 1
                        seen.add(canon)
                        break
        total = sum(counts.values()) or 1
        pairs = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [(name, round(100.0 * cnt / total, 1)) for name, cnt in pairs]
    finally:
        if own_session:
            session.close()


def get_visibility_ranking(
    session: Optional[Session],
    *,
    project_id: Optional[int] = None,
    start_date: Optional[str],
    end_date: Optional[str],
) -> list[tuple[str, float]]:
    """Ranking de visibilidad: porcentaje de apariciones por marca sobre total de respuestas, acotado al mercado si se indica project_id."""
    own_session = False
    if session is None:
        session = get_session()
        own_session = True
    try:
        # Traer también key_topics para replicar la detección del endpoint
        where = [
            "m.created_at >= CAST(:start_date AS date)",
            "m.created_at < (CAST(:end_date AS date) + INTERVAL '1 day')",
        ]
        params: Dict[str, Any] = {"start_date": start_date or "1970-01-01", "end_date": end_date or "2999-12-31"}
        if project_id is not None:
            where.append("COALESCE(q.project_id, q.id) = COALESCE((SELECT project_id FROM queries WHERE id = :pid), :pid)")
            params["pid"] = int(project_id)
        sql = text(
            f"""
            SELECT
              m.key_topics,
              LOWER(COALESCE(m.response,'')) AS resp,
              LOWER(COALESCE(m.source_title,'')) AS title,
              i.payload
            FROM mentions m
            JOIN queries q ON q.id = m.query_id
            LEFT JOIN insights i ON i.id = m.generated_insight_id
            WHERE {' AND '.join(where)}
            """
        )
        rows = session.execute(sql, params).mappings().all()
        from collections import Counter
        total_by_brand: Counter[str] = Counter()
        # Sinónimos normalizados
        norm_synonyms: Dict[str, list[str]] = {c: [c.lower()] + [s.lower() for s in alts] for c, alts in BRAND_SYNONYMS.items()}
        for r in rows:
            payload = r.get("payload") or {}
            resp = (r.get("resp") or "").lower()
            title = (r.get("title") or "").lower()
            try_topics: list[str] = []
            try:
                if isinstance(r.get("key_topics"), (list, tuple)):
                    try_topics = [str(t).strip().lower() for t in (r.get("key_topics") or [])]
            except Exception:
                try_topics = []
            payload_brands: list[str] = []
            if isinstance(payload, dict):
                raw_b = payload.get("brands")
                if isinstance(raw_b, list):
                    for b in raw_b:
                        if isinstance(b, dict):
                            name = (b.get("name") or "").strip().lower()
                            if name:
                                payload_brands.append(name)
                        elif isinstance(b, str):
                            payload_brands.append(b.strip().lower())
            detected: list[str] = []
            for canon, syns in norm_synonyms.items():
                for s in syns:
                    if s in try_topics or (s and (s in resp or s in title or s in payload_brands)):
                        detected.append(canon)
                        break
            for canon in set(detected):
                total_by_brand[canon] += 1
        total_responses = len(rows) or 1
        pairs = sorted(total_by_brand.items(), key=lambda x: x[1], reverse=True)
        return [(name, round(100.0 * cnt / total_responses, 1)) for name, cnt in pairs]
    finally:
        if own_session:
            session.close()


def get_sentiment_positive_series(
    session: Optional[Session],
    project_id: int,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    client_brand: Optional[str] = None,
) -> list[tuple[str, float]]:
    """Serie diaria del porcentaje de menciones POSITIVAS de la marca principal (0-100%)."""
    own_session = False
    if session is None:
        session = get_session()
        own_session = True
    try:
        # Resolver marca
        if client_brand is None or not str(client_brand).strip():
            brow = session.execute(text("SELECT COALESCE(brand, topic, 'Unknown') AS b FROM queries WHERE id=:pid"), {"pid": int(project_id)}).first()
            client_brand = (brow[0] if brow else "Unknown")
        syns = [str(client_brand or "Unknown").lower()]
        syns.extend([s.lower() for s in BRAND_SYNONYMS.get(str(client_brand or "Unknown"), [])])

        if not start_date:
            start_date = "1970-01-01"
        if not end_date:
            end_date = "2999-12-31"

        sql = text(
            """
            WITH rows AS (
                SELECT 
                    DATE(m.created_at) AS d,
                    COALESCE(m.sentiment, 0) AS sent,
                    (
                        EXISTS (
                            SELECT 1 FROM jsonb_array_elements_text(COALESCE(to_jsonb(m.key_topics),'[]'::jsonb)) kt
                            WHERE LOWER(TRIM(kt)) = ANY(:syns)
                        )
                        OR LOWER(COALESCE(m.response,'')) LIKE ANY(:likes)
                        OR LOWER(COALESCE(m.source_title,'')) LIKE ANY(:likes)
                        OR EXISTS (
                            SELECT 1 FROM jsonb_array_elements(COALESCE(i.payload->'brands','[]'::jsonb)) b
                            WHERE LOWER(TRIM(CASE WHEN jsonb_typeof(b)='object' THEN COALESCE(b->>'name','') ELSE TRIM(BOTH '"' FROM b::text) END)) = ANY(:syns)
                        )
                    ) AS is_brand
                FROM mentions m
                JOIN queries q ON q.id = m.query_id
                LEFT JOIN insights i ON i.id = m.generated_insight_id
                WHERE COALESCE(q.project_id, q.id) = COALESCE((SELECT project_id FROM queries WHERE id = :pid), :pid)
                  AND m.created_at >= CAST(:start_date AS date)
                  AND m.created_at < (CAST(:end_date AS date) + INTERVAL '1 day')
            )
            SELECT d,
                   SUM(CASE WHEN is_brand AND sent > 0.3 THEN 1 ELSE 0 END) AS pos,
                   SUM(CASE WHEN is_brand THEN 1 ELSE 0 END) AS tot
            FROM rows
            GROUP BY d
            ORDER BY d
            """
        )
        likes = [f"%{s}%" for s in syns]
        rows = session.execute(sql, {"pid": int(project_id), "syns": syns, "likes": likes, "start_date": start_date, "end_date": end_date}).all()
        series: list[tuple[str, float]] = []
        for d, pos, tot in rows:
            pct = (float(pos or 0) / float(max(tot or 0, 1))) * 100.0
            series.append((str(d), round(pct, 1)))
        return series
    finally:
        if own_session:
            session.close()

def get_agent_insights_data(session: Optional[Session], project_id: int | None, limit: int = 200) -> Dict[str, Any]:
    """
    Recupera payloads de la tabla insights asociados al proyecto (query_id) y
    devuelve un resumen normalizado para prompts y PDF.
    """
    own_session = False
    if session is None:
        session = get_session()
        own_session = True
    try:
        if project_id is None or int(project_id) <= 0:
            sql = text(
                """
                SELECT i.payload
                FROM insights i
                ORDER BY i.created_at DESC
                LIMIT :lim
                """
            )
            params = {"lim": int(limit)}
        else:
            sql = text(
                """
                SELECT i.payload
                FROM insights i
                WHERE i.query_id = :pid
                ORDER BY i.created_at DESC
                LIMIT :lim
                """
            )
            params = {"pid": int(project_id), "lim": int(limit)}
        rows = session.execute(sql, params).mappings().all()
        payload_rows = [{"payload": r.get("payload")} for r in rows]
        return summarize_agent_insights(payload_rows)
    finally:
        if own_session:
            session.close()


def get_agent_insight_payloads(
    session: Optional[Session],
    project_id: int,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Devuelve filas crudas de la tabla insights (payload y created_at) para un proyecto
    y rango temporal opcional. Útil para poblar la UI con TODAS las entradas sin
    recortes Top-N.
    """
    own_session = False
    if session is None:
        session = get_session()
        own_session = True
    try:
        where = ["i.query_id = :pid"]
        params: Dict[str, Any] = {"pid": int(project_id), "lim": int(max(1, min(limit, 5000)))}
        if start_date:
            where.append("i.created_at >= CAST(:start AS date)")
            params["start"] = start_date
        if end_date:
            where.append("i.created_at < (CAST(:end AS date) + INTERVAL '1 day')")
            params["end"] = end_date

        sql = text(
            f"""
            SELECT i.payload, i.created_at
            FROM insights i
            WHERE {' AND '.join(where)}
            ORDER BY i.created_at DESC
            LIMIT :lim
            """
        )
        rows = session.execute(sql, params).mappings().all()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({"payload": r.get("payload"), "created_at": r.get("created_at")})
        return out
    finally:
        if own_session:
            session.close()


def get_raw_mentions_for_topic(topic: str, limit: int = 25) -> List[str]:
    """
    Devuelve texto literal de menciones asociadas a un tema dado.
    Coincide por:
      - que el topic exista en m.key_topics (array JSON)
      - o que q.topic o q.category coincida con el tema
    """
    session = get_session()
    try:
        sql = text(
            """
            SELECT m.response
            FROM mentions m
            JOIN queries q ON q.id = m.query_id
            WHERE (
                EXISTS (
                    SELECT 1
                    FROM jsonb_array_elements_text(COALESCE(m.key_topics, '[]'::jsonb)) AS t(val)
                    WHERE lower(t.val) = lower(:topic)
                )
                OR lower(COALESCE(q.category, '')) = lower(:topic)
                OR lower(COALESCE(q.topic, '')) = lower(:topic)
            )
            ORDER BY m.created_at DESC
            LIMIT :lim
            """
        )
        rows = session.execute(sql, {"topic": topic, "lim": int(max(1, limit))}).all()
        out: List[str] = []
        for (resp,) in rows:
            if isinstance(resp, str) and resp.strip():
                out.append(resp.strip())
        return out
    finally:
        session.close()


def get_all_mentions_for_period(
    limit: int = 100,
    *,
    start_date: Optional[str] | None = None,
    end_date: Optional[str] | None = None,
    client_id: Optional[int] | None = None,
    brand_id: Optional[int] | None = None,
    project_id: Optional[int] | None = None,
    client_brand: Optional[str] | None = None,
) -> List[str]:
    """
    Devuelve un corpus global de menciones (texto crudo) sin filtrar por tema, dentro de un
    rango de fechas. Recoge menciones recientes y representativas de TODAS las queries.

    - Usa un límite entre 100 y 150 para equilibrio coste/calidad.
    - Descarta textos vacíos y muy cortos.
    - Ordena por fecha descendente para priorizar actualidad.
    """
    from datetime import datetime, timedelta
    # Normalizar fechas
    def _to_dt(val, default):
        try:
            if isinstance(val, datetime):
                return val
            if isinstance(val, str) and val:
                return datetime.strptime(val[:10], "%Y-%m-%d")
            return default
        except Exception:
            return default

    end_dt = _to_dt(end_date, datetime.utcnow())
    start_dt = _to_dt(start_date, end_dt - timedelta(days=30))

    session = get_session()
    try:
        where = [
            "m.created_at >= :start::date",
            "m.created_at < (:end::date + INTERVAL '1 day')",
        ]
        params: Dict[str, Any] = {
            "start": start_dt.strftime("%Y-%m-%d"),
            "end": end_dt.strftime("%Y-%m-%d"),
            "lim": int(max(1, min(150, limit))),
        }
        if client_id is not None:
            where.append("q.client_id = :client_id")
            params["client_id"] = int(client_id)
        if brand_id is not None:
            where.append("q.brand_id = :brand_id")
            params["brand_id"] = int(brand_id)
        if project_id is not None:
            where.append("COALESCE(q.project_id, q.id) = COALESCE((SELECT project_id FROM queries WHERE id = :pid), :pid)")
            params["pid"] = int(project_id)

        # Usar CAST(:param AS date) para evitar problemas de parseo de SQLAlchemy/psycopg2
        where_sql = ' AND '.join(where).replace(":start::date", "CAST(:start AS date)").replace(":end::date", "CAST(:end AS date)")
        sql = text(
            f"""
            SELECT m.response
            FROM mentions m
            JOIN queries q ON q.id = m.query_id
            WHERE {where_sql}
            ORDER BY m.created_at DESC
            LIMIT :lim
            """
        )

        rows = session.execute(sql, params).all()
        corpus: List[str] = []
        syns: List[str] = []
        if client_brand and str(client_brand).strip():
            base = str(client_brand).strip()
            syns = [base.lower()] + [s.lower() for s in BRAND_SYNONYMS.get(base, [])]
        for (resp,) in rows:
            if not isinstance(resp, str):
                continue
            txt = resp.strip()
            if len(txt) < 40:
                continue
            if syns:
                low = txt.lower()
                if not any(s in low for s in syns):
                    continue
            corpus.append(txt)
        return corpus
    finally:
        session.close()


class ClusterMention(TypedDict):
    id: int
    summary: str
    sentiment: float
    source: str | None
    domain: str | None
    created_at: str


class ClusterResult(TypedDict):
    cluster_id: int
    centroid: list[float]
    count: int
    avg_sentiment: float
    top_sources: list[tuple[str, int]]
    example_mentions: list[ClusterMention]


def _parse_vector_text(vec_text: str) -> list[float]:
    # Espera formato "[v1,v2,...]"; robustez ante espacios
    s = vec_text.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    if not s:
        return []
    return [float(x) for x in s.replace(' ', '').split(',') if x]


def aggregate_clusters_for_report(
    session: Optional[Session],
    project_id: int,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_rows: int = 5000,
) -> list[ClusterResult]:
    """
    Recupera menciones con embedding no nulo en el periodo, ejecuta clustering (KMeans) y
    devuelve clusters con metadatos y ejemplos representativos.
    """
    own_session = False
    if session is None:
        session = get_session()
        own_session = True
    try:
        where = [
            "COALESCE(q.project_id, q.id) = COALESCE((SELECT project_id FROM queries WHERE id = :project_id), :project_id)",
            "m.embedding IS NOT NULL",
        ]
        params: Dict[str, Any] = {"project_id": int(project_id), "lim": int(max_rows)}
        if start_date:
            where.append("m.created_at >= CAST(:start AS date)")
            params["start"] = start_date
        if end_date:
            where.append("m.created_at < (CAST(:end AS date) + INTERVAL '1 day')")
            params["end"] = end_date

        sql = text(
            f"""
            SELECT m.id,
                   m.summary,
                   m.sentiment,
                   m.source,
                   m.source_domain,
                   m.created_at,
                   m.embedding::text AS emb
            FROM mentions m
            JOIN queries q ON q.id = m.query_id
            WHERE {' AND '.join(where)}
            ORDER BY m.created_at DESC
            LIMIT :lim
            """
        )
        rows = session.execute(sql, params).all()
        if not rows:
            return []

        mentions: list[ClusterMention] = []
        vectors: list[list[float]] = []
        for rid, summary, sent, source, domain, created_at, emb_txt in rows:
            if not isinstance(emb_txt, str):
                continue
            vec = _parse_vector_text(emb_txt)
            if not vec:
                continue
            mentions.append({
                "id": int(rid),
                "summary": str(summary or ""),
                "sentiment": float(sent or 0.0),
                "source": str(source) if source is not None else None,
                "domain": str(domain) if domain is not None else None,
                "created_at": str(created_at),
            })
            vectors.append(vec)

        if not mentions or not vectors:
            return []

        X = np.array(vectors, dtype=np.float32)
        # Normalizar para usar similitud coseno de forma eficiente con dot product
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        Xn = X / norms

        n = Xn.shape[0]
        k_default = int(max(2, min(12, round(sqrt(n / 2)))))
        if KMeans is None:
            # Fallback: un único cluster si no hay sklearn
            labels = np.zeros(n, dtype=int)
            centroids = np.mean(Xn, axis=0, keepdims=True)
        else:
            km = KMeans(n_clusters=k_default, n_init=10, max_iter=300, random_state=42)
            labels = km.fit_predict(Xn)
            centroids = km.cluster_centers_

        results: list[ClusterResult] = []
        for cid in sorted(set(int(l) for l in labels.tolist())):
            idx = np.where(labels == cid)[0]
            if idx.size == 0:
                continue
            cluster_vectors = Xn[idx]
            centroid = np.mean(cluster_vectors, axis=0)
            # cercanía por coseno = dot(centroid_norm, vec)
            cent_norm = np.linalg.norm(centroid) + 1e-12
            centroid_n = centroid / cent_norm
            scores = cluster_vectors @ centroid_n
            order = np.argsort(-scores)[:20]
            sel_idx = idx[order]
            selected: list[ClusterMention] = [mentions[i] for i in sel_idx]

            avg_sent = float(np.mean([mentions[i]["sentiment"] for i in idx]))
            from collections import Counter
            src_counter = Counter([mentions[i]["domain"] or mentions[i]["source"] or "unknown" for i in idx])
            top_sources = [(k or "unknown", int(v)) for k, v in src_counter.most_common(5)]

            results.append({
                "cluster_id": int(cid),
                "centroid": [float(x) for x in centroid.tolist()],
                "count": int(idx.size),
                "avg_sentiment": avg_sent,
                "top_sources": top_sources,
                "example_mentions": selected,
            })
        # Ordenar por tamaño desc
        results.sort(key=lambda c: c["count"], reverse=True)
        return results
    finally:
        if own_session:
            session.close()


def get_full_report_data(
    project_id: int,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_rows: int = 5000,
    client_brand: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Devuelve un objeto híbrido con KPIs + series + SOV + clusters (con ejemplos) listo
    para ser consumido por el generador de informes.
    """
    session = get_session()
    try:
        # KPIs y métricas (aplicar rango temporal si viene informado)
        kpis = get_kpi_summary(session, project_id, start_date=start_date, end_date=end_date, client_brand=client_brand)
        evo = get_sentiment_evolution(session, project_id, start_date=start_date, end_date=end_date, client_brand=client_brand)
        by_cat = get_sentiment_by_category(session, project_id, start_date=start_date, end_date=end_date, client_brand=client_brand)
        top5, bottom5 = get_topics_by_sentiment(session, project_id, start_date=start_date, end_date=end_date, client_brand=client_brand)
        sov_trends = get_share_of_voice_and_trends(session, project_id, start_date=start_date, end_date=end_date, client_brand=client_brand)

        # Clusters
        clusters = aggregate_clusters_for_report(session, project_id, start_date=start_date, end_date=end_date, max_rows=max_rows)

        # Oportunidades competitivas (tabla)
        def _resolve_main_brand(sess: Session, pid: int) -> str:
            row = sess.execute(text("SELECT COALESCE(brand, topic, 'Unknown') FROM queries WHERE id=:pid"), {"pid": pid}).first()
            return str(row[0]) if row and row[0] is not None else "Unknown"

        main_brand = _resolve_main_brand(session, project_id)
        competitive_opps = get_competitive_opportunities(
            session,
            project_id,
            start_date=start_date or "1970-01-01",
            end_date=end_date or "2999-12-31",
            main_brand=main_brand,
            top_n=10,
        )

        # Serie global de visibilidad para el dashboard
        visibility_series = get_visibility_series(session, project_id, start_date=start_date, end_date=end_date, client_brand=client_brand)

        return {
            "project_id": project_id,
            "kpis": kpis,
            "time_series": {
                "sentiment_per_day": evo,
            },
            "visibility_timeseries": visibility_series,
            "sentiment_by_category": by_cat,
            "topics_top5": top5,
            "topics_bottom5": bottom5,
            "sov": sov_trends,
            "clusters": clusters,
            "competitive_opportunities": competitive_opps,
        }
    finally:
        session.close()


class Aggregator:
    """
    Interfaz orientada a objetos para orquestar la obtención de datos del informe.
    Mantiene compatibilidad con el enfoque SQL actual sin requerir DataFrames.
    """
    def get_full_report_data(
        self,
        project_id: int,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_rows: int = 5000,
    ) -> Dict[str, Any]:
        return get_full_report_data(
            project_id,
            start_date=start_date,
            end_date=end_date,
            max_rows=max_rows,
        )

def get_competitive_opportunities(
    session: Session,
    project_id: int,
    *,
    start_date: str,
    end_date: str,
    main_brand: str,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Detecta oportunidades competitivas: temas donde los competidores (no la marca principal)
    tienen presencia significativa, y genera una acción de contenido para capitalizarlas.
    """
    sql = text(
        """
        SELECT
          COALESCE(q.brand, q.topic, 'Unknown') AS brand_name,
          COALESCE((i.payload->>'category'), COALESCE(q.category, q.topic, 'Desconocida')) AS topic,
          COUNT(*) AS mention_count
        FROM mentions m
        JOIN queries q ON q.id = m.query_id
        LEFT JOIN insights i ON i.id = m.generated_insight_id
        WHERE COALESCE(q.project_id, q.id) = COALESCE((SELECT project_id FROM queries WHERE id = :project_id), :project_id)
          AND m.created_at >= CAST(:start_date AS date)
          AND m.created_at < (CAST(:end_date AS date) + INTERVAL '1 day')
        GROUP BY 1, 2
        ORDER BY mention_count DESC
        LIMIT 50
        """
    )
    rows = session.execute(sql, {
        "project_id": int(project_id),
        "start_date": start_date,
        "end_date": end_date,
    }).mappings().all()

    candidates: List[Dict[str, Any]] = []
    for r in rows:
        brand_name = str(r.get("brand_name") or "Unknown")
        if brand_name == (main_brand or "Unknown"):
            continue
        topic = str(r.get("topic") or "Desconocida")
        cnt = int(r.get("mention_count") or 0)
        candidates.append({"Competidor": brand_name, "Debilidad Detectada": topic, "#": cnt})

    # Top N por conteo
    candidates.sort(key=lambda x: x.get("#", 0), reverse=True)
    selected = candidates[:max(1, min(top_n, len(candidates)))]

    # Generar acción de contenido con IA
    enriched: List[Dict[str, Any]] = []
    for item in selected:
        competitor = item["Competidor"]
        weakness_topic = item["Debilidad Detectada"]
        prompt = (
            f"Eres un estratega de contenidos. Un competidor llamado '{competitor}' tiene fuerte presencia en el tema "
            f"'{weakness_topic}'. Propón una acción de contenido concreta para que '{main_brand}' capitalice esta situación. "
            f"Devuelve una única idea en una frase, clara y accionable."
        )
        try:
            idea = fetch_response(prompt, model="gpt-4o-mini", temperature=0.3, max_tokens=80)
        except Exception:
            idea = ""  # Fallback silencioso
        enriched.append({
            "Competidor": competitor,
            "Debilidad Detectada": weakness_topic,
            "#": item["#"],
            "Acción de Contenido": (idea or "Producir una pieza diferenciadora enfocado en el tema.").strip(),
        })

    return enriched
