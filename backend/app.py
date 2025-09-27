# backend/app.py

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import psycopg2
import os
from datetime import datetime, timedelta
import pytz
# Importamos una librería más robusta para parsear fechas
from dateutil.parser import parse as parse_date
import json
from dotenv import load_dotenv
import re
import unicodedata
from difflib import SequenceMatcher
from src.engines.openai_engine import fetch_response, fetch_response_with_metadata
from src.engines.report_prompts import (
    get_executive_summary_prompt,
    get_deep_dive_analysis_prompt,
    get_competitive_analysis_prompt,
    get_recommendations_prompt,
    get_methodology_prompt,
    get_correlation_anomalies_prompt,
)
from time import time
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle, Image
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.lib import colors
import matplotlib
matplotlib.use('Agg')  # Evita backends GUI (NSWindow) en macOS/entornos no interactivos
import matplotlib.pyplot as plt
from reportlab.lib.utils import ImageReader
from src.reports.generator import generate_report as generate_report_v2

load_dotenv()

app = Flask(__name__)
CORS(app)

# --- CONFIGURACIÓN Y HELPERS ---

def _env(*names: str, default: str | int | None = None):
    for n in names:
        v = os.getenv(n)
        if v is not None and v != "":
            return v
    return default

def _clean_model_json(text: str) -> dict:
    try:
        raw = (text or '').strip()
        if raw.startswith('```json') and raw.endswith('```'):
            raw = raw[7:-3].strip()
        if raw.startswith('```') and raw.endswith('```'):
            raw = raw[3:-3].strip()
        return json.loads(raw)
    except Exception:
        # Fallback seguro
        return {}

def _log_ai_call(kind: str, prompt: str, raw: str, meta: dict) -> None:
    try:
        logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, 'ai_calls.log')
        entry = {
            'ts': datetime.utcnow().isoformat(),
            'kind': kind,
            'model': meta.get('model_name'),
            'meta': {k: meta.get(k) for k in ['input_tokens','output_tokens','price_usd','engine_request_id']},
            'prompt': prompt,
            'raw': raw,
        }
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception:
        pass

# Caché simple en memoria para resultados de agrupación por IA (TTL configurable)
_GROUPS_CACHE: dict[str, tuple[float, list[dict]]] = {}
GROUPS_CACHE_TTL_SECONDS = int(os.getenv("TOPICS_GROUPS_TTL", "60"))

def _groups_cache_get(key: str) -> list[dict] | None:
    try:
        ts, data = _GROUPS_CACHE.get(key, (0.0, None))  # type: ignore
        if not data:
            return None
        if (time() - ts) <= GROUPS_CACHE_TTL_SECONDS:
            return data
        # Expirado
        _GROUPS_CACHE.pop(key, None)
        return None
    except Exception:
        return None

def _groups_cache_set(key: str, data: list[dict]) -> None:
    try:
        _GROUPS_CACHE[key] = (time(), data)
    except Exception:
        # En caso de error, no bloquear el flujo
        pass

# --- Nueva Función de Categorización con IA ---
def _normalize_text_for_match(text: str) -> list[str]:
    s = (text or "").lower()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^a-z0-9áéíóúñü\s]", " ", s)
    tokens = [t for t in s.split() if len(t) >= 2]
    return tokens


def _best_topic_by_similarity(candidate: str, topics: list[str]) -> tuple[str, float]:
    """Devuelve (mejor_topic, score) usando difflib + similitud por tokens."""
    if not topics:
        return candidate, 0.0
    cand_norm = " ".join(_normalize_text_for_match(candidate))
    best = None
    best_score = -1.0
    for t in topics:
        s1 = SequenceMatcher(None, cand_norm, t.lower()).ratio()
        # Jaccard simple de conjuntos de tokens
        a = set(_normalize_text_for_match(candidate))
        b = set(_normalize_text_for_match(t))
        j = (len(a & b) / max(1, len(a | b))) if (a or b) else 0.0
        score = max(s1, j)
        if score > best_score:
            best = t
            best_score = score
    return best or candidate, float(best_score)


def categorize_prompt_with_ai(query_text: str, topics: list[str]) -> str:
    """Usa un LLM para clasificar una query; mapea a categorías existentes si son parecidas."""
    # Preparamos los topics como una lista numerada para la IA
    topics_list_str = "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(topics)])

    prompt = f"""
Eres un asistente experto en organización de contenido para "The Core School". Tu tarea es clasificar la siguiente "Consulta del Usuario" en una de las "Categorías Disponibles".

Instrucciones Clave:

Si la consulta es clara y relevante para la escuela (cine, audiovisual, software, videojuegos, marketing, etc.), clasifícala en una categoría existente o crea una nueva.

Si la consulta es demasiado vaga, ambigua, sin sentido, o completamente irrelevante (ej. "dónde comprar pan", "el cielo es azul"), responde únicamente con la palabra: Inclasificable.

Categorías Disponibles:
{topics_list_str}

Consulta del Usuario:
"{query_text}"

Responde solo con el nombre de la categoría o con la palabra "Inclasificable".
"""
    try:
        # Usamos el motor de OpenAI para obtener la respuesta
        best_category = fetch_response(prompt, model="gpt-4o-mini", temperature=0.0)

        # Limpiamos respuesta (a veces IA añade números o frases)
        cleaned_category = (best_category or "").strip()
        if "\n" in cleaned_category:
            cleaned_category = cleaned_category.split("\n", 1)[0]
        if ":" in cleaned_category and cleaned_category.lower().startswith("categoria"):
            cleaned_category = cleaned_category.split(":", 1)[-1]
        cleaned_category = cleaned_category.strip().split('.')[-1].strip()

        if cleaned_category.lower() == "inclasificable":
            return "Inclasificable"

        # Si coincide exactamente con una categoría conocida
        if cleaned_category in topics:
            return cleaned_category

        # Si no coincide, mapear por similitud a categoría existente cuando sea alta
        mapped, score = _best_topic_by_similarity(cleaned_category, topics)
        if score >= 0.72:
            return mapped

        # Como respaldo: si la query contiene términos del dominio, preferir el mejor match aunque no supere umbral alto
        DOMAIN_HINTS = [
            "cine", "film", "rodaje", "guion", "guión", "fotografia", "fotografía",
            "animacion", "animación", "vfx", "edicion", "edición", "postproduccion", "postproducción",
            "videojuego", "videojuegos", "programacion", "programación", "software", "ingenieria", "ingeniería",
            "marketing", "comunicacion", "comunicación", "master", "máster", "grado", "curso", "beca", "becas",
            "escuela", "universidad"
        ]
        qtoks = set(_normalize_text_for_match(query_text))
        if any(h in qtoks for h in DOMAIN_HINTS) and score >= 0.6:
            return mapped

        # En caso contrario, devolver tal cual (puede ser nueva categoría)
        return cleaned_category or (mapped if mapped else "Inclasificable")

    except Exception as e:
        print(f"Error en la categorización con IA: {e}")
        # Si todo falla, volvemos a la lógica antigua como respaldo
        return categorize_prompt(None, query_text)

DB_CONFIG = {
    # Preferimos variables con prefijo POSTGRES_ (docker compose),
    # si no existen caemos a DB_* y finalmente a un valor por defecto seguro.
    "host": _env("POSTGRES_HOST", "DB_HOST", default="localhost"),
    "port": int(_env("POSTGRES_PORT", "DB_PORT", default=5433)),
    "database": _env("POSTGRES_DB", "DB_NAME", default="ai_visibility"),
    "user": _env("POSTGRES_USER", "DB_USER", default="postgres"),
    "password": _env("POSTGRES_PASSWORD", "DB_PASSWORD", default="postgres"),
}

def get_db_connection():
    try:
        # Debug: mostrar configuración (sin password)
        dbg = {k: ("***" if k == "password" else v) for k, v in DB_CONFIG.items()}
        print(f"DB_CONFIG: {dbg}")
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.OperationalError as e:
        print(f"ERROR: No se pudo conectar a la base de datos: {e}")
        raise e


# ───────────────────── Motor de informes (helpers) ─────────────────────
def _aggregate_data_for_report_mock(filters):
    """Recopila y resume todos los datos necesarios para el informe desde la BD."""
    conn = get_db_connection()
    cur = conn.cursor()

    # Esta función debe ser implementada con consultas SQL reales a tu base de datos
    # A continuación, se muestra una estructura de datos simulada como ejemplo.
    aggregated_data = {
        "start_date": (filters or {}).get('start_date', 'N/A')[:10],
        "end_date": (filters or {}).get('end_date', 'N/A')[:10],
        "kpis": {
            "visibility_score": 34.5, "visibility_delta": -2.1,
            "sentiment_avg": 0.61, "sov_score": 28.9,
            "total_mentions": 1240
        },
        "time_series": {
            "dates": [f"Día {i+1}" for i in range(7)],
            "visibility": [10, 12, 15, 11, 14, 18, 15],
            "sentiment": [0.4, 0.5, 0.6, 0.45, 0.55, 0.7, 0.6]
        },
        "competitor_ranking": [
            ["1", "ECAM", "45.1%", "Becas, Precio"],
            ["2", "The Core School", "28.9%", "VFX, Prácticas"],
            ["3", "U-tad", "15.3%", "Videojuegos, Ingeniería"]
        ],
        "top_positive_themes": [
            ["VFX", "0.85", "150 menciones"],
            ["Salidas Profesionales", "0.78", "120 menciones"]
        ],
        "top_negative_themes": [
            ["Precio del Máster", "-0.65", "80 menciones"],
            ["Proceso de Admisión", "-0.40", "65 menciones"]
        ],
        "emerging_trends": ["Interés en producción virtual", "Demanda de perfiles híbridos (arte y tecnología)"],
        "top_opportunities": ["Capitalizar el interés creciente en VFX.", "Lanzar campaña sobre el éxito laboral de alumni."],
        "top_risks": ["Competencia agresiva en precios y becas.", "Percepción de que la formación es solo para cine."],
        "key_quotes": [
            "Me encantaría estudiar en The Core, pero el máster es demasiado caro para mí.",
            "Las prácticas que ofrecen en empresas top son el principal motivo por el que los elijo."
        ]
    }

    cur.close()
    conn.close()
    return aggregated_data


def _aggregate_data_for_report(filters):
    """Agrega datos reales desde la base de datos (menciones, insights, KPIs)."""
    return _aggregate_data_for_report_db(filters)


def _aggregate_data_for_report_db(filters):
    """Agrega datos reales desde la base de datos (menciones, insights, KPIs)."""
    # Normaliza filtros
    brand = os.getenv('DEFAULT_BRAND', 'The Core School')
    start_s = (filters or {}).get('start_date')
    end_s = (filters or {}).get('end_date')
    model = (filters or {}).get('model')
    source = (filters or {}).get('source')
    topic = (filters or {}).get('topic')
    # Multi-cliente y categorías
    client_id = (filters or {}).get('client_id')
    brand_id = (filters or {}).get('brand_id')
    category = (filters or {}).get('category') or (filters or {}).get('prompt_category')

    # Acepta tanto strings ISO como objetos datetime
    def _to_dt(val, default):
        try:
            from datetime import datetime as _dt
            if isinstance(val, _dt):
                return val
            if isinstance(val, str):
                return parse_date(val)
            return default
        except Exception:
            return default
    start_dt = _to_dt(start_s, datetime.utcnow() - timedelta(days=30))
    end_dt = _to_dt(end_s, datetime.utcnow())

    conn = get_db_connection(); cur = conn.cursor()

    # WHERE base
    where = ["m.created_at >= %s AND m.created_at < %s"]
    params = [start_dt, end_dt]
    if model and model != 'all':
        where.append("m.engine = %s"); params.append(model)
    if source and source != 'all':
        where.append("m.source = %s"); params.append(source)
    if topic and topic != 'all':
        where.append("COALESCE(q.category, q.topic) = %s"); params.append(topic)
    if category and category != 'all':
        where.append("q.category = %s"); params.append(category)
    if client_id:
        where.append("q.client_id = %s"); params.append(client_id)
    if brand_id:
        where.append("q.brand_id = %s"); params.append(brand_id)
    where_sql = " AND ".join(where)

    # Traer filas necesarias para cómputos (detección de marcas + sentimiento)
    cur.execute(f"""
        SELECT m.id, m.key_topics, LOWER(COALESCE(m.response,'')) AS resp,
               LOWER(COALESCE(m.source_title,'')) AS title,
               i.payload, m.sentiment, m.created_at
        FROM mentions m
        JOIN queries q ON m.query_id = q.id
        LEFT JOIN insights i ON i.id = m.generated_insight_id
        WHERE {where_sql}
    """, tuple(params))
    rows = cur.fetchall()

    total_responses = len(rows)

    # Conteos por marca (SOV y visibilidad)
    from collections import defaultdict, Counter
    brand_counts = defaultdict(int)
    brand_sentiments = defaultdict(list)
    competitor_daily_counts = defaultdict(lambda: defaultdict(int))  # brand -> day -> count

    by_day_sentiments = defaultdict(list)
    topic_counts = defaultdict(int)
    topic_sent_sum = defaultdict(float)
    for _id, key_topics, resp, title, payload, senti, created_at in rows:
        detected = _detect_brands(key_topics, resp, title, payload)
        seen = set()
        for bn in detected:
            if bn in seen:
                continue
            seen.add(bn)
            brand_counts[bn] += 1
            if isinstance(senti, (int, float)):
                brand_sentiments[bn].append(float(senti))
            # Conteo diario por competidor
            d_day = created_at.date() if created_at else start_dt.date()
            if bn != brand:
                competitor_daily_counts[bn][d_day] += 1
        # Sentimiento promedio por día (global del periodo)
        d_sent = created_at.date() if created_at else start_dt.date()
        if isinstance(senti, (int, float)):
            by_day_sentiments[d_sent].append(float(senti))
        # Agregado de temas a partir de key_topics
        try:
            if isinstance(key_topics, (list, tuple)):
                for t in key_topics:
                    if not t:
                        continue
                    topic = str(t).strip().lower()
                    topic_counts[topic] += 1
                    if isinstance(senti, (int, float)):
                        topic_sent_sum[topic] += float(senti)
        except Exception:
            pass

    def avg(xs):
        return (sum(xs) / max(len(xs), 1)) if xs else 0.0

    primary_brand = brand
    visibility_score = round((brand_counts.get(primary_brand, 0) / max(total_responses, 1)) * 100.0, 1)
    sentiment_avg = round(avg(brand_sentiments.get(primary_brand, [])), 2)

    # Serie por día para delta de visibilidad
    by_day_total = defaultdict(int)
    by_day_brand = defaultdict(int)
    for _id, key_topics, resp, title, payload, senti, created_at in rows:
        d = created_at.date() if created_at else start_dt.date()
        by_day_total[d] += 1
        if primary_brand in _detect_brands(key_topics, resp, title, payload):
            by_day_brand[d] += 1
    days_sorted = sorted(by_day_total.keys())
    series_visibility = [ (by_day_brand[d] / max(by_day_total[d], 1)) * 100.0 for d in days_sorted ]
    series_sentiment = [ (sum(by_day_sentiments.get(d, [])) / max(len(by_day_sentiments.get(d, [])), 1)) if by_day_sentiments.get(d) else 0.0 for d in days_sorted ]
    if len(series_visibility) >= 2:
        mid = len(series_visibility) // 2
        delta = round((sum(series_visibility[mid:]) / max(len(series_visibility[mid:]), 1)) - (sum(series_visibility[:mid]) / max(len(series_visibility[:mid]), 1)), 1)
    else:
        delta = 0.0

    # Ranking SOV de los principales (según diccionario de marcas)
    sov_list = []
    for canon in BRAND_SYNONYMS.keys():
        cnt = brand_counts.get(canon, 0)
        sov = round((cnt / max(total_responses, 1)) * 100.0, 1)
        if cnt > 0:
            sov_list.append({"name": canon, "sov": sov})
    sov_list.sort(key=lambda x: x["sov"], reverse=True)
    competitor_ranking = [
        {
            "rank": i+1,
            "name": it["name"],
            "sov": f"{it['sov']:.1f}%",
            "mentions": brand_counts.get(it["name"], 0),
        }
        for i, it in enumerate(sov_list[:10])
    ]

    # GAP frente al líder de SOV
    top_sov = sov_list[0]["sov"] if sov_list else 0.0
    own_sov = next((it["sov"] for it in sov_list if it["name"] == primary_brand), 0.0)
    sov_gap_vs_leader = round(max(top_sov - own_sov, 0.0), 1)
    sov_leader_name = sov_list[0]["name"] if sov_list else None
    # Líder competitivo distinto de la marca propia (para serie temporal)
    competitor_leader_name = next((it["name"] for it in sov_list if it["name"] != primary_brand), None)

    # Oportunidades / Riesgos / Tendencias / Citas desde insights
    # Filtros equivalentes para insights (sin campos de mentions como engine/source)
    where_i = ["i.created_at >= %s AND i.created_at < %s"]
    params_i = [start_dt, end_dt]
    if topic and topic != 'all':
        where_i.append("COALESCE(q.category, q.topic) = %s"); params_i.append(topic)
    if category and category != 'all':
        where_i.append("q.category = %s"); params_i.append(category)
    if client_id:
        where_i.append("q.client_id = %s"); params_i.append(client_id)
    if brand_id:
        where_i.append("q.brand_id = %s"); params_i.append(brand_id)
    where_sql_i = " AND ".join(where_i)
    cur.execute(f"""
        SELECT i.payload
        FROM insights i
        JOIN queries q ON i.query_id = q.id
        WHERE {where_sql_i}
    """, tuple(params_i))
    insights_rows = cur.fetchall()
    opp, risk, trend, quotes = Counter(), Counter(), Counter(), []
    for (payload,) in insights_rows:
        if not isinstance(payload, dict):
            continue
        for t in (payload.get('opportunities') or []):
            opp[t] += 1
        for t in (payload.get('risks') or []):
            risk[t] += 1
        for t in (payload.get('trends') or []):
            trend[t] += 1
        for q in (payload.get('quotes') or []):
            if isinstance(q, str):
                quotes.append(q)

    aggregated_data = {
        "start_date": start_dt.strftime('%Y-%m-%d'),
        "end_date": end_dt.strftime('%Y-%m-%d'),
        "brand": primary_brand,
        "kpis": {
            "visibility_score": visibility_score,
            "visibility_delta": delta,
            "sentiment_avg": sentiment_avg,
            "sov_score": (sov_list[0]["sov"] if sov_list else 0.0),
            "sov_leader_name": sov_leader_name,
            "sov_gap_vs_leader": sov_gap_vs_leader,
            "total_mentions": total_responses,
        },
        "competitor_ranking": competitor_ranking,
        "competitor_themes": {},
        # Temas top por sentimiento
        "top_positive_themes": [],
        "top_negative_themes": [],
        # Materia prima para gráficos de temas
        "topic_counts": {k: int(v) for k, v in topic_counts.items()},
        "topic_sent_sum": {k: float(v) for k, v in topic_sent_sum.items()},
        "emerging_trends": [t for t, _ in trend.most_common(5)],
        "top_opportunities": [t for t, _ in opp.most_common(5)],
        "top_risks": [t for t, _ in risk.most_common(5)],
        "key_quotes": quotes[:5],
        "sov_chart_data": sov_list[:5],
        # Series temporales para gráficos
        "time_series": {
            "dates": [d.strftime('%Y-%m-%d') for d in days_sorted],
            "visibility": [round(v, 1) for v in series_visibility],
            "sentiment": [round(v, 2) for v in series_sentiment],
            "mentions": [int(by_day_total[d]) for d in days_sorted],
        },
        # Serie temporal de menciones del competidor líder (si no somos nosotros)
        "competitor_series": {
            "name": competitor_leader_name,
            "counts": [
                (competitor_daily_counts.get(competitor_leader_name, {}) or {}).get(d, 0)
                for d in days_sorted
            ] if competitor_leader_name else [],
        },
        # Correlaciones tema→sentimiento (coeficiente de Pearson simplificado si aplica)
        "topic_sentiment_correlations": _topic_sentiment_correlation(topic_counts, topic_sent_sum),
        # Días outlier por sentimiento (threshold dinámico por desviación estándar)
        "outlier_days": _detect_outlier_days(days_sorted, series_sentiment),
    }

    # Calcular Top 5 temas positivos/negativos a partir de key_topics
    try:
        topic_avg = []
        for t, cnt in topic_counts.items():
            if cnt < 3:
                continue
            avg_s = topic_sent_sum[t] / max(cnt, 1)
            topic_avg.append((t, avg_s, cnt))
        topic_avg.sort(key=lambda x: x[1], reverse=True)
        aggregated_data["top_positive_themes"] = [
            [name, f"{score:.2f}", f"{cnt} menciones"] for name, score, cnt in topic_avg[:5]
        ]
        topic_avg.sort(key=lambda x: x[1])
        aggregated_data["top_negative_themes"] = [
            [name, f"{score:.2f}", f"{cnt} menciones"] for name, score, cnt in topic_avg[:5]
        ]
    except Exception:
        pass

    # Correlaciones simples: caídas fuertes de sentimiento y pico de un competidor el mismo día
    try:
        correlation_events = []
        # Detecta caídas > 0.2 respecto al día anterior
        for idx in range(1, len(days_sorted)):
            d = days_sorted[idx]
            prev = days_sorted[idx - 1]
            s_today = series_sentiment[idx]
            s_prev = series_sentiment[idx - 1]
            if (s_prev - s_today) >= 0.2:
                # busca competidor con más menciones ese día
                best_comp = None
                best_cnt = 0
                for comp, by_day in competitor_daily_counts.items():
                    c = by_day.get(d, 0)
                    if c > best_cnt:
                        best_cnt = c
                        best_comp = comp
                if best_comp and best_cnt > 0:
                    correlation_events.append({
                        "date": d.strftime('%Y-%m-%d'),
                        "competitor": best_comp,
                        "competitor_mentions": best_cnt,
                        "sentiment_drop": round(s_prev - s_today, 2),
                    })
        aggregated_data["correlation_events"] = correlation_events[:5]
    except Exception:
        aggregated_data["correlation_events"] = []

    # Solapamientos: días con riesgos citados y picos de un competidor
    try:
        # Construye mapa de riesgos por día desde insights_rows
        risk_by_day = defaultdict(lambda: defaultdict(int))
        # Reutiliza la consulta previa de insights agregados pero con fechas
        cur = get_db_connection().cursor()
        cur.execute(
            f"""
            SELECT i.payload, i.created_at
            FROM insights i
            JOIN queries q ON i.query_id = q.id
            WHERE {where_sql_i}
            """,
            tuple(params_i)
        )
        for payload, created_at in cur.fetchall():
            if not isinstance(payload, dict):
                continue
            d_day = (created_at.date() if created_at else start_dt.date())
            for rtxt in (payload.get('risks') or []):
                if not rtxt:
                    continue
                risk_by_day[d_day][rtxt] += 1
        cur.close()

        overlap_events = []
        for comp, by_day in competitor_daily_counts.items():
            for d, cnt in by_day.items():
                if cnt <= 0:
                    continue
                risks_today = risk_by_day.get(d, {})
                if not risks_today:
                    continue
                # Top 1-2 riesgos del día
                top_risks = sorted(risks_today.items(), key=lambda x: x[1], reverse=True)[:2]
                overlap_events.append({
                    "date": d.strftime('%Y-%m-%d'),
                    "competitor": comp,
                    "competitor_mentions": cnt,
                    "risks": [t for t, _ in top_risks],
                })
        # Limita y ordena por menciones del competidor desc
        overlap_events.sort(key=lambda x: x.get('competitor_mentions', 0), reverse=True)
        aggregated_data["overlap_events"] = overlap_events[:8]
    except Exception:
        aggregated_data["overlap_events"] = []

    # ── SOV por modelo de IA y por tema ─────────────────────────────────
    try:
        cur = get_db_connection().cursor()
        cur.execute(
            f"""
            SELECT m.engine, q.topic,
                   m.key_topics,
                   LOWER(COALESCE(m.response,'')) AS resp,
                   LOWER(COALESCE(m.source_title,'')) AS title
            FROM mentions m
            JOIN queries q ON m.query_id = q.id
            WHERE {where_sql}
            """,
            tuple(params),
        )
        rows_et = cur.fetchall()
        cur.close()

        counts_by_engine_brand = defaultdict(lambda: defaultdict(int))
        counts_by_topic_brand = defaultdict(lambda: defaultdict(int))
        total_by_engine = defaultdict(int)
        total_by_topic = defaultdict(int)

        for engine, topic, key_topics, resp, title in rows_et:
            detected = _detect_brands(key_topics, resp, title, None)
            if not detected:
                continue
            seen = set()
            for bn in detected:
                if bn in seen:
                    continue
                seen.add(bn)
                counts_by_engine_brand[engine][bn] += 1
                counts_by_topic_brand[topic][bn] += 1
                total_by_engine[engine] += 1
                total_by_topic[topic] += 1

        sov_by_model = []
        for engine, brand_map in counts_by_engine_brand.items():
            tot = max(total_by_engine.get(engine, 0), 1)
            items = sorted(brand_map.items(), key=lambda x: x[1], reverse=True)[:6]
            for bn, cnt in items:
                sov_by_model.append([str(engine or 'unknown'), bn, f"{(cnt/tot)*100:.1f}%"]) 

        sov_by_topic = []
        for topic, brand_map in counts_by_topic_brand.items():
            tot = max(total_by_topic.get(topic, 0), 1)
            items = sorted(brand_map.items(), key=lambda x: x[1], reverse=True)[:6]
            for bn, cnt in items:
                sov_by_topic.append([str(topic or 'unknown'), bn, f"{(cnt/tot)*100:.1f}%"]) 

        aggregated_data["sov_by_model"] = sov_by_model[:30]
        aggregated_data["sov_by_topic"] = sov_by_topic[:30]
    except Exception:
        aggregated_data["sov_by_model"] = []
        aggregated_data["sov_by_topic"] = []

    # Distribuciones densas en datos para el Analista de Correlaciones y Anomalías
    try:
        aggregated_data["distributions"] = {
            "sentiment_hist": _histogram([s for vals in brand_sentiments.values() for s in vals], bins=10, range_min=-1.0, range_max=1.0),
            "mentions_by_engine": _dict_counts(total_by_engine),
            "mentions_by_topic": _dict_counts(total_by_topic),
        }
    except Exception:
        aggregated_data["distributions"] = {"sentiment_hist": [], "mentions_by_engine": [], "mentions_by_topic": []}

    # Comparativa de sentimiento por marca
    try:
        sentiment_comp = [
            [bn, f"{(sum(vals)/max(len(vals),1)):.2f}"]
            for bn, vals in brand_sentiments.items() if vals
        ]
        sentiment_comp.sort(key=lambda x: float(x[1]), reverse=True)
        aggregated_data["sentiment_comparison"] = sentiment_comp[:10]
    except Exception:
        aggregated_data["sentiment_comparison"] = []

    # ── Desglose por Categoría (7 áreas) ────────────────────────────────
    try:
        base_no_cat = ["m.created_at >= %s AND m.created_at < %s"]
        params_no_cat = [start_dt, end_dt]
        if model and model != 'all':
            base_no_cat.append("m.engine = %s"); params_no_cat.append(model)
        if source and source != 'all':
            base_no_cat.append("m.source = %s"); params_no_cat.append(source)
        if client_id:
            base_no_cat.append("q.client_id = %s"); params_no_cat.append(client_id)
        if brand_id:
            base_no_cat.append("q.brand_id = %s"); params_no_cat.append(brand_id)
        where_no_cat_sql = " AND ".join(base_no_cat)

        cur = get_db_connection().cursor()
        cur.execute(
            f"""
            SELECT COALESCE(q.category, q.topic) AS cat,
                   m.sentiment, m.summary, i.payload
            FROM mentions m
            JOIN queries q ON m.query_id = q.id
            LEFT JOIN insights i ON i.id = m.generated_insight_id
            WHERE {where_no_cat_sql}
            """,
            tuple(params_no_cat)
        )
        cat_rows = cur.fetchall()
        cur.close()

        from collections import defaultdict
        cat_sent = defaultdict(list)
        cat_summaries = defaultdict(list)
        cat_opps = defaultdict(lambda: defaultdict(int))
        cat_risks = defaultdict(lambda: defaultdict(int))
        cat_quotes = defaultdict(list)
        for cat, s, summ, payload in cat_rows:
            cname = (cat or 'Sin categoría')
            try:
                if isinstance(s, (int, float)):
                    cat_sent[cname].append(float(s))
            except Exception:
                pass
            if isinstance(summ, str) and summ.strip():
                if len(cat_summaries[cname]) < 6:
                    cat_summaries[cname].append(summ.strip())
            if isinstance(payload, dict):
                for t in (payload.get('opportunities') or []):
                    if isinstance(t, str):
                        cat_opps[cname][t] += 1
                for t in (payload.get('risks') or []):
                    if isinstance(t, str):
                        cat_risks[cname][t] += 1
                for q in (payload.get('quotes') or []):
                    if isinstance(q, str) and len(cat_quotes[cname]) < 4:
                        cat_quotes[cname].append(q)

        by_category = []
        for cname in sorted(cat_sent.keys() or {"Sin categoría": []}):
            opp_sorted = sorted(cat_opps[cname].items(), key=lambda x: x[1], reverse=True)[:5]
            risk_sorted = sorted(cat_risks[cname].items(), key=lambda x: x[1], reverse=True)[:5]
            by_category.append({
                "category": cname,
                "sentiment_avg": round((sum(cat_sent[cname]) / max(len(cat_sent[cname]), 1)), 2) if cat_sent[cname] else 0.0,
                "top_opportunities": [t for t, _ in opp_sorted],
                "top_risks": [t for t, _ in risk_sorted],
                "examples": cat_summaries[cname][:6],
                "quotes": cat_quotes[cname][:4],
            })
        aggregated_data["by_category"] = by_category
    except Exception:
        aggregated_data["by_category"] = []

    # ── Corpus textual por categoría para prosa del informe ─────────────
    try:
        cur = get_db_connection().cursor()
        cur.execute(
            f"""
            SELECT COALESCE(q.category, q.topic) AS cat,
                   m.response, m.sentiment, m.source_title, m.source_url
            FROM mentions m
            JOIN queries q ON m.query_id = q.id
            WHERE {where_sql}
            ORDER BY m.created_at DESC
            """,
            tuple(params)
        )
        rows_corpus = cur.fetchall()
        cur.close()

        from collections import defaultdict
        bucket = defaultdict(lambda: {"pos": [], "neu": [], "neg": []})

        def _shorten(txt: str, limit: int = 700) -> str:
            try:
                t = (txt or '').strip().replace('\n', ' ')
                if len(t) <= limit:
                    return t
                return t[:limit].rsplit(' ', 1)[0] + '…'
            except Exception:
                return (txt or '')[:limit]

        for cat, text, s, title, url in rows_corpus:
            cname = (cat or 'Sin categoría')
            sentiment_band = 'neu'
            try:
                if isinstance(s, (int, float)):
                    if s > 0.15:
                        sentiment_band = 'pos'
                    elif s < -0.15:
                        sentiment_band = 'neg'
            except Exception:
                sentiment_band = 'neu'
            item = {"excerpt": _shorten(text or ''), "title": title or '', "url": url or '', "sentiment": float(s) if isinstance(s, (int, float)) else 0.0}
            lst = bucket[cname][sentiment_band]
            if len(lst) < 10:
                lst.append(item)

        corpus_by_category = []
        corpus_for_llm = {}
        for cname, bands in bucket.items():
            texts = bands['pos'] + bands['neu'] + bands['neg']
            corpus_by_category.append({"category": cname, "texts": texts})
            sample = texts[:6]
            corpus_for_llm[cname] = [t["excerpt"] for t in sample]

        aggregated_data["corpus_by_category"] = corpus_by_category
        aggregated_data["corpus_for_llm"] = corpus_for_llm
    except Exception:
        aggregated_data["corpus_by_category"] = []
        aggregated_data["corpus_for_llm"] = {}

    # Intención del usuario (informacional, comparativa, transaccional)
    try:
        cur = get_db_connection().cursor()
        cur.execute(f"""
            SELECT COALESCE(m.query_text, q.query) AS qtext
            FROM mentions m
            JOIN queries q ON m.query_id = q.id
            WHERE {where_sql}
        """, tuple(params))
        intent_counts = {"informacional": 0, "comparativa": 0, "transaccional": 0}
        examples = {"informacional": [], "comparativa": [], "transaccional": []}
        import re as _re
        def _classify_intent(txt: str) -> str:
            t = (txt or '').lower()
            if _re.search(r"\b(vs|versus|comparar|comparaci[oó]n|mejor\s+(escuela|universidad|centro))\b", t):
                return "comparativa"
            if _re.search(r"\b(precio|coste|costo|beca|becas|financiaci[oó]n|matr[ií]cula|inscripci[oó]n|admis[ií]n|solicitar|aplicar|apply|comprar)\b", t):
                return "transaccional"
            return "informacional"
        for (qtext,) in cur.fetchall():
            intent = _classify_intent(qtext)
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            if len(examples[intent]) < 3 and qtext:
                examples[intent].append(qtext[:140])
        cur.close()
        total_i = max(sum(intent_counts.values()), 1)
        aggregated_data["user_intent"] = {
            "distribution": [
                {"intent": k, "count": v, "pct": round((v/total_i)*100.0, 1)}
                for k, v in intent_counts.items()
            ],
            "examples": examples,
        }
    except Exception:
        aggregated_data["user_intent"] = {"distribution": [], "examples": {}}

    # Serie temporal de SOV por temas clave (para la marca propia)
    try:
        # Selección de top temas por volumen total en el periodo
        top_topics = [t for t, _ in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True) if t][:5]
        # Limitar a 3 para legibilidad del gráfico
        top_topics = top_topics[:3]
        from collections import defaultdict as _dd
        topic_day_total = _dd(lambda: _dd(int))
        topic_day_own = _dd(lambda: _dd(int))
        # Reiterar filas ya cargadas para construir contadores por día y tema
        for _id, key_topics, resp, title, payload, senti, created_at in rows:
            d_day = created_at.date() if created_at else start_dt.date()
            detected = _detect_brands(key_topics, resp, title, payload)
            topics_iter = []
            try:
                if isinstance(key_topics, (list, tuple)):
                    topics_iter = [str(t).strip().lower() for t in key_topics if t]
            except Exception:
                topics_iter = []
            for t in topics_iter:
                if t not in top_topics:
                    continue
                topic_day_total[t][d_day] += 1
                if primary_brand in detected:
                    topic_day_own[t][d_day] += 1
        topic_sov_timeseries = []
        for t in top_topics:
            sov_series = []
            for d in days_sorted:
                tot = max(topic_day_total[t].get(d, 0), 1)
                own = topic_day_own[t].get(d, 0)
                sov_series.append(round((own / tot) * 100.0, 1))
            topic_sov_timeseries.append({
                "topic": t,
                "dates": [d.strftime('%Y-%m-%d') for d in days_sorted],
                "values": sov_series,
            })
        aggregated_data["topic_sov_timeseries"] = topic_sov_timeseries
    except Exception:
        aggregated_data["topic_sov_timeseries"] = []

    # Resumen de alertas clave
    try:
        alerts = []
        # Alerta: sentimiento muy negativo en un tema relevante
        for name, score_str, cnt_str in (aggregated_data.get("top_negative_themes") or [])[:5]:
            try:
                score = float(score_str)
                cnt = int((cnt_str or '0').split()[0])
            except Exception:
                score = 0.0; cnt = 0
            if score <= -0.5 and cnt >= 10:
                alerts.append(f"Alerta: El sentimiento sobre '{name}' cayó por debajo de -0.5 (promedio {score:.2f}, {cnt} menciones)")
        # Alerta: picos competitivos vinculados a caídas de sentimiento
        if aggregated_data.get("correlation_events"):
            ev = aggregated_data["correlation_events"][0]
            alerts.append(
                f"Alerta: Caída de sentimiento el {ev.get('date')} coincide con pico de '{ev.get('competitor')}' ({ev.get('competitor_mentions')} menciones)"
            )
        # Alerta: nuevo competidor en top 5 vs periodo previo
        try:
            prev_start = start_dt - (end_dt - start_dt)
            prev_end = start_dt
            c2 = get_db_connection().cursor()
            c2.execute("""
                SELECT m.id, m.key_topics, LOWER(COALESCE(m.response,'')) AS resp,
                       LOWER(COALESCE(m.source_title,'')) AS title, m.sentiment, m.created_at
                FROM mentions m
                JOIN queries q ON m.query_id = q.id
                WHERE m.created_at >= %s AND m.created_at < %s
            """, (prev_start, prev_end))
            prev_rows = c2.fetchall(); c2.close()
            from collections import defaultdict as __dd
            prev_counts = __dd(int)
            for _id, key_topics, resp, title, senti, created_at in prev_rows:
                for bn in _detect_brands(key_topics, resp, title, None):
                    prev_counts[bn] += 1
            prev_total = max(sum(prev_counts.values()), 1)
            prev_sov = sorted([
                {"name": bn, "sov": (cnt/prev_total)*100.0} for bn, cnt in prev_counts.items()
            ], key=lambda x: x["sov"], reverse=True)
            curr_top5 = {r["name"] for r in aggregated_data.get("competitor_ranking", [])[:5] if r.get("name")}
            prev_top5 = {it["name"] for it in prev_sov[:5]}
            newcomers = [n for n in curr_top5 if (n not in prev_top5 and n != primary_brand)]
            for n in newcomers:
                alerts.append(f"Alerta: Nuevo competidor '{n}' ha entrado en el top 5 de SOV")
        except Exception:
            pass
        aggregated_data["alerts_summary"] = alerts[:6]
    except Exception:
        aggregated_data["alerts_summary"] = []

    # Insight comparativo destacado (dos métricas y breve análisis)
    try:
        k = aggregated_data.get("kpis", {})
        leader = aggregated_data.get("kpis", {}).get("sov_leader_name")
        own_vis = float(k.get("visibility_score", 0.0))
        leader_sov = float(k.get("sov_score", 0.0))
        gap = float(k.get("sov_gap_vs_leader", 0.0))
        analysis = (
            f"La visibilidad de {primary_brand} es {own_vis:.1f}% mientras el SOV de {leader or 'el líder'} es {leader_sov:.1f}%. "
            + (f"Brecha de {gap:.1f} pts a cerrar con acciones de contenido en los temas ganadores." if leader else "")
        )
        aggregated_data["comparative_highlight"] = {
            "title": "Análisis Comparativo Destacado",
            "metric_labels": [f"Visibilidad {primary_brand}", f"SOV {leader or 'Líder'}"],
            "metric_values": [round(own_vis,1), round(leader_sov,1)],
            "analysis_text": analysis,
        }
    except Exception:
        aggregated_data["comparative_highlight"] = None

    # Benchmarking de contenido competitivo (debilidades de competidores y acciones)
    try:
        weakness_map = {}
        for _id, key_topics, resp, title, payload, senti, created_at in rows:
            try:
                s_val = float(senti) if isinstance(senti, (int, float)) else 0.0
            except Exception:
                s_val = 0.0
            if s_val >= -0.2:
                continue
            detected = _detect_brands(key_topics, resp, title, payload)
            topics_iter = []
            try:
                if isinstance(key_topics, (list, tuple)):
                    topics_iter = [str(t).strip() for t in key_topics if t]
            except Exception:
                topics_iter = []
            for bn in detected:
                if bn == primary_brand:
                    continue
                for t in topics_iter:
                    key = (bn, t)
                    weakness_map[key] = weakness_map.get(key, 0) + 1
        # Ordenar por conteo desc y limitar
        ordered = sorted(weakness_map.items(), key=lambda x: x[1], reverse=True)[:8]
        suggestions = []
        for (bn, t), cnt in ordered:
            title_sug = f"Así son las {t} en {primary_brand}: de la teoría a la práctica"
            if "precio" in t.lower() or "beca" in t.lower():
                title_sug = f"Transparencia en {primary_brand}: precios, becas y financiación explicados"
            suggestions.append({
                "competitor": bn, "theme": t, "count": cnt,
                "suggestion_title": title_sug,
                "suggestion_text": f"Se detectaron {cnt} menciones negativas sobre '{t}' en {bn}. Publicar contenido que muestre fortalezas reales de {primary_brand} en este tema."
            })
        aggregated_data["content_competitive"] = suggestions
    except Exception:
        aggregated_data["content_competitive"] = []

    return aggregated_data
def _generate_sentiment_by_topic_chart(positive_themes, negative_themes):
    """Genera un gráfico de barras horizontales para el sentimiento por tema."""
    try:
        # Combinar y limitar temas
        themes = positive_themes[:5] + negative_themes[:5]
        if not themes:
            return None

        theme_names = [t[0] for t in themes]
        sentiments = [float(t[1]) for t in themes]
        colors_list = ['#2ca02c' if s > 0 else '#d62728' for s in sentiments]

        fig, ax = plt.subplots(figsize=(7, max(2, 0.4 * len(themes))))
        y_pos = range(len(themes))
        ax.barh(y_pos, sentiments, color=colors_list, align='center')
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(theme_names, fontsize=8)
        ax.invert_yaxis()  # Muestra el más positivo arriba
        ax.set_xlabel('Sentimiento Promedio')
        ax.set_title('Sentimiento por Tema Clave', fontsize=12)
        ax.axvline(0, color='grey', linewidth=0.5)
        ax.set_xlim([-1, 1])

        fig.tight_layout()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='PNG', dpi=150)
        plt.close(fig)
        img_buffer.seek(0)
        return img_buffer
    except Exception as e:
        print(f"Error generando gráfico de sentimiento por tema: {e}")
        return None


def _generate_word_cloud_image(topic_counts, positive=True):
    """Genera una nube de palabras para temas positivos o negativos."""
    try:
        from wordcloud import WordCloud

        # Filtrar temas por sentimiento
        filtered_topics = {
            topic: count
            for topic, count, sentiment in topic_counts
            if (positive and sentiment > 0.1) or (not positive and sentiment < -0.1)
        }

        if not filtered_topics:
            return None

        wc = WordCloud(width=350, height=180, background_color="white", colormap="viridis" if positive else "Reds")
        wc.generate_from_frequencies(filtered_topics)

        img_buffer = io.BytesIO()
        wc.to_image().save(img_buffer, format='PNG')
        img_buffer.seek(0)
        return img_buffer
    except ImportError:
        print("WARN: WordCloud no instalado. Omitiendo nubes de palabras. Instalar con: pip install wordcloud")
        return None
    except Exception as e:
        print(f"Error generando nube de palabras: {e}")
        return None


def _generate_charts_as_image(data):
    """Genera un gráfico de Visibilidad vs. Sentimiento y lo guarda en un buffer de memoria."""
    fig, ax1 = plt.subplots(figsize=(7, 3.5))

    ax1.set_xlabel('Días del Periodo')
    ax1.set_ylabel('Visibilidad (%)', color='tab:blue')
    ax1.plot(data['time_series']['dates'], data['time_series']['visibility'], color='tab:blue', marker='o', label='Visibilidad')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Sentimiento Promedio', color='tab:green')
    ax2.plot(data['time_series']['dates'], data['time_series']['sentiment'], color='tab:green', marker='x', linestyle='--', label='Sentimiento')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.set_ylim(-1, 1)

    fig.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='PNG', dpi=150)
    plt.close(fig)
    img_buffer.seek(0)
    return img_buffer


def _generate_competitor_vs_sentiment_image(dates, competitor_counts, sentiment):
    """Gráfico de líneas: menciones del competidor líder vs. sentimiento diario."""
    try:
        if not dates or not competitor_counts or not sentiment:
            return None
        fig, ax1 = plt.subplots(figsize=(7, 2.6))
        x = list(range(1, len(dates) + 1))
        ax1.plot(x, competitor_counts, color='#d62728', marker='o', label='Menciones competidor')
        ax1.set_ylabel('Menciones competidor', color='#d62728')
        ax1.tick_params(axis='y', labelcolor='#d62728')
        ax2 = ax1.twinx()
        ax2.plot(x, sentiment, color='#2ca02c', linestyle='--', marker='x', label='Sentimiento')
        ax2.set_ylabel('Sentimiento', color='#2ca02c')
        ax2.tick_params(axis='y', labelcolor='#2ca02c')
        ax2.set_ylim(-1, 1)
        ax1.set_xlabel('Días del periodo')
        fig.tight_layout()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='PNG', dpi=150)
        plt.close(fig)
        img_buffer.seek(0)
        return img_buffer
    except Exception:
        return None


def _generate_sentiment_histogram_image(hist_data):
    """Genera un histograma visual del sentimiento a partir de bins agregados."""
    try:
        if not hist_data:
            return None
        bins = [str(it.get('bin', '')) for it in hist_data]
        counts = [int(it.get('count', 0)) for it in hist_data]
        fig, ax = plt.subplots(figsize=(6.0, 2.2))
        ax.bar(range(len(counts)), counts, color='#7b6cff')
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bins, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Menciones', fontsize=8)
        ax.set_title('Histograma de Sentimiento', fontsize=10)
        fig.tight_layout()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='PNG', dpi=150)
        plt.close(fig)
        img_buffer.seek(0)
        return img_buffer
    except Exception:
        return None


def _generate_topic_sentiment_heatmap_image(correlations):
    """Genera un heatmap (barra divergente) de tema→sentimiento promedio (-1 a 1)."""
    try:
        if not correlations:
            return None
        # Ordena por valor para un gradiente legible
        corr_sorted = sorted(correlations, key=lambda x: float(x[1]))
        topics = [str(t) for t, _ in corr_sorted]
        values = [float(v) for _, v in corr_sorted]

        import matplotlib.colors as mcolors
        cmap = plt.get_cmap('RdYlGn')
        norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
        colors = [cmap(norm(v)) for v in values]

        height = max(2.0, 0.32 * len(topics) + 0.8)
        fig, ax = plt.subplots(figsize=(6.2, height))
        y = list(range(len(topics)))
        ax.barh(y, values, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(topics, fontsize=8)
        ax.set_xlim(-1.0, 1.0)
        ax.axvline(0.0, color='grey', linewidth=0.8)
        ax.set_xlabel('Sentimiento promedio', fontsize=9)
        ax.set_title('Correlación Tema → Sentimiento', fontsize=11)
        fig.tight_layout()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='PNG', dpi=150)
        plt.close(fig)
        img_buffer.seek(0)
        return img_buffer
    except Exception:
        return None


def _create_pdf_report_consulting(content, data):
    """Construye un PDF completo en memoria, con un estilo de consultoría."""
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=4))

    # Margen y helper de cabecera/pie
    M = inch
    def draw_header_footer():
        try:
            p.setFont("Helvetica", 8)
            p.drawRightString(width - M, 0.6*inch, f"Página {p.getPageNumber()}")
        except Exception:
            pass

    def write_paragraph(text, x, y, style_name='Justify', font_size=10):
        style = styles[style_name]
        style.fontSize = font_size
        para = Paragraph(text.replace('\n', '<br/>'), style)
        w, h = para.wrapOn(p, width - x - inch, height)
        if y - h < inch:
            p.showPage(); y = height - inch
        para.drawOn(p, x, y - h)
        return y - h - (font_size * 1.2)

    y = height - inch

    # Sección 1: Resumen Ejecutivo (defensiva)
    exec_sum = (content.get('executive_summary') or {}) if isinstance(content, dict) else {}
    p.setFont("Helvetica-Bold", 16)
    p.drawString(inch, y, exec_sum.get('title', '1. Resumen Ejecutivo'))
    y -= 20
    p.setFont("Helvetica", 10)
    p.drawString(inch, y, f"Periodo de Análisis: {data.get('start_date','N/A')} - {data.get('end_date','N/A')}")
    y -= 30
    headline = exec_sum.get('headline') or 'Sin titular disponible.'
    y = write_paragraph(f"<i><b>Headline:</b> {headline}</i>", inch, y, font_size=11)
    y -= 10
    key_findings = exec_sum.get('key_findings') or []
    if key_findings:
        y = write_paragraph("<b>Hallazgos Clave:</b><br/>- " + "<br/>- ".join(key_findings), inch, y)
    overall = exec_sum.get('overall_assessment') or 'N/A'
    y = write_paragraph(f"<b>Evaluación General:</b><br/>{overall}", inch, y)
    y -= 20

    # Inserta gráfico de tendencia (Visibilidad y Sentimiento)
    try:
        chart_buf = _generate_charts_as_image(data)
        if chart_buf:
            img = ImageReader(chart_buf)
            p.drawImage(img, inch, y - 200, width=width - 2*inch, height=180, mask='auto')
            y -= 200
    except Exception:
        pass

    # Tabla de KPIs
    try:
        k = data.get('kpis', {})
        table_data = [
            ["KPI", "Valor"],
            ["Visibilidad (%)", f"{k.get('visibility_score', 0):.1f}%"],
            ["Δ Visibilidad", f"{k.get('visibility_delta', 0):+.1f} pts"],
            ["Sentimiento", f"{k.get('sentiment_avg', 0):.2f}"],
            ["SOV líder", f"{k.get('sov_score', 0):.1f}%"],
            ["Brecha vs líder (SOV)", f"{k.get('sov_gap_vs_leader', 0):.1f} pts"],
            ["Total menciones", f"{k.get('total_mentions', 0)}"],
        ]
        tbl = Table(table_data, colWidths=[2.8*inch, 2.8*inch])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ]))
        w, h = tbl.wrapOn(p, width - 2*inch, height)
        if y - h < inch:
            p.showPage(); y = height - inch
        tbl.drawOn(p, inch, y - h)
        y -= (h + 20)
    except Exception:
        pass

    p.showPage(); y = height - inch

    # --- NUEVO: Sección de Diagnóstico de Visibilidad y Reputación ---
    draw_header_footer()
    p.setFont("Helvetica-Bold", 14)
    p.drawString(M, y, "2. Diagnóstico de Visibilidad y Reputación")
    y -= 30

    # Gráfico de Sentimiento por Tema
    try:
        pos_themes = data.get('top_positive_themes', [])
        neg_themes = data.get('top_negative_themes', [])
        sentiment_chart_buf = _generate_sentiment_by_topic_chart(pos_themes, neg_themes)
        if sentiment_chart_buf:
            img = ImageReader(sentiment_chart_buf)
            img_height = min(250, 20 + 20 * (len(pos_themes) + len(neg_themes)))
            p.drawImage(img, M, y - img_height, width=width - 2*M, height=img_height - 10, preserveAspectRatio=True)
            y -= (img_height + 15)
    except Exception as e:
        print(f"Error al dibujar gráfico de sentimiento: {e}")

    # Nubes de Palabras
    try:
        topic_sentiment_data = [
            (t, data['topic_counts'][t], data['topic_sent_sum'][t] / data['topic_counts'][t])
            for t in (data.get('topic_counts') or {})
            if data['topic_counts'].get(t)
        ]
        pos_cloud_buf = _generate_word_cloud_image(topic_sentiment_data, positive=True)
        neg_cloud_buf = _generate_word_cloud_image(topic_sentiment_data, positive=False)
        if pos_cloud_buf and neg_cloud_buf:
            if y < 220:
                p.showPage(); y = height - inch; draw_header_footer()
            p.setFont("Helvetica-Bold", 11)
            p.drawString(M, y, "Temas Positivos Dominantes")
            p.drawString(width/2 + 0.5*inch, y, "Temas Negativos Recurrentes")
            y -= 200
            pos_img = ImageReader(pos_cloud_buf)
            neg_img = ImageReader(neg_cloud_buf)
            p.drawImage(pos_img, M, y, width=3.5*inch, height=180, preserveAspectRatio=True)
            p.drawImage(neg_img, width/2 + 0.5*inch, y, width=3.5*inch, height=180, preserveAspectRatio=True)
            y -= 20
    except Exception as e:
        print(f"Error al dibujar nubes de palabras: {e}")


    # --- NUEVO: Sección de Análisis Competitivo ---
    p.showPage(); y = height - inch
    draw_header_footer()
    comp_landscape = content.get('competitive_landscape', {})
    p.setFont("Helvetica-Bold", 14)
    p.drawString(M, y, comp_landscape.get('title', '3. Análisis del Panorama Competitivo'))
    y -= 30

    y = write_paragraph(f"<b>Posicionamiento General:</b><br/>{comp_landscape.get('overall_positioning', 'N/A')}", M, y)
    y -= 10
    y = write_paragraph(f"<b>Análisis del Líder ({data.get('kpis', {}).get('sov_leader_name', 'N/A')}):</b><br/>{comp_landscape.get('leader_analysis', 'N/A')}", M, y)
    y -= 15

    p.setFont("Helvetica-Bold", 12)
    p.drawString(M, y, "Oportunidades frente a Debilidades de Competidores")
    y -= 20

    for weakness in comp_landscape.get('competitor_weaknesses', [])[:8]:
        if y < inch * 2:
            p.showPage(); y = height - inch; draw_header_footer()
        text = f"<b>Competidor:</b> {weakness.get('competitor', 'N/A')}<br/>"
        text += f"<b>Tema de Debilidad:</b> {weakness.get('theme', 'N/A')}<br/>"
        text += f"<b>Análisis Estratégico:</b> {weakness.get('analysis', 'N/A')}"
        y = write_paragraph(text, M, y)
        y -= 15

    # Sección 2: Análisis Granular (Deep Dive)
    deep = content.get('deep_dive_analysis') or {}
    p.setFont("Helvetica-Bold", 14)
    p.drawString(inch, y, deep.get('title', 'Análisis Granular'))
    y -= 30

    # Tabla: SOV por Modelo de IA
    try:
        header = [["Modelo IA", "Marca", "SOV"]]
        rows = header + (data.get('sov_by_model') or [])
        tbl = Table(rows, colWidths=[2.0*inch, 3.0*inch, 1.4*inch])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ]))
        w, h = tbl.wrapOn(p, width - 2*inch, height)
        if y - h < inch:
            p.showPage(); y = height - inch
        tbl.drawOn(p, inch, y - h)
        y -= (h + 12)
        if deep.get('visibility_by_model'):
            y = write_paragraph(f"<b>Lectura:</b> {deep['visibility_by_model']}", inch, y)
    except Exception:
        pass

    # Tabla: SOV por Tema
    try:
        header = [["Tema", "Marca", "SOV"]]
        rows = header + (data.get('sov_by_topic') or [])
        tbl = Table(rows, colWidths=[2.0*inch, 3.0*inch, 1.4*inch])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ]))
        w, h = tbl.wrapOn(p, width - 2*inch, height)
        if y - h < inch:
            p.showPage(); y = height - inch
        tbl.drawOn(p, inch, y - h)
        y -= (h + 12)
        if deep.get('visibility_by_topic'):
            y = write_paragraph(f"<b>Lectura:</b> {deep['visibility_by_topic']}", inch, y)
    except Exception:
        pass

    # Tabla: Comparativa de Sentimiento por Marca
    try:
        header = [["Marca", "Sentimiento Promedio"]]
        rows = header + (data.get('sentiment_comparison') or [])
        tbl = Table(rows, colWidths=[4.0*inch, 2.0*inch])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ]))
        w, h = tbl.wrapOn(p, width - 2*inch, height)
        if y - h < inch:
            p.showPage(); y = height - inch
        tbl.drawOn(p, inch, y - h)
        y -= (h + 12)
        if deep.get('sentiment_comparison'):
            y = write_paragraph(f"<b>Lectura:</b> {deep['sentiment_comparison']}", inch, y)
    except Exception:
        pass

    # Tablas: Top temas positivos / negativos
    try:
        if y < inch * 2:
            p.showPage(); y = height - inch
        p.setFont("Helvetica-Bold", 14)
        p.drawString(inch, y, "Temas por Sentimiento")
        y -= 22
        pos = [["Tema", "Sentimiento", "Menciones"]] + (data.get('top_positive_themes') or [])
        neg = [["Tema", "Sentimiento", "Menciones"]] + (data.get('top_negative_themes') or [])
        tpos = Table(pos, colWidths=[3.2*inch, 1.2*inch, 1.2*inch])
        tpos.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
        ]))
        tneg = Table(neg, colWidths=[3.2*inch, 1.2*inch, 1.2*inch])
        tneg.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
        ]))
        w1, h1 = tpos.wrapOn(p, (width - 2*inch) / 2 - 6, height)
        w2, h2 = tneg.wrapOn(p, (width - 2*inch) / 2 - 6, height)
        tpos.drawOn(p, inch, y - h1)
        tneg.drawOn(p, inch + (width - 2*inch) / 2 + 6, y - h2)
        y -= max(h1, h2) + 10
    except Exception:
        pass

    if y < inch * 3: p.showPage(); y = height - inch

    # --- NUEVO: Secciones por Categoría con citas ---
    try:
        categories = data.get('by_category') or []
        corpus = data.get('corpus_by_category') or []
        corpus_map = {c.get('category'): c.get('texts') for c in corpus}
        for cat in categories:
            if y < inch * 2:
                p.showPage(); y = height - inch; draw_header_footer()
            cname = cat.get('category', 'Sin categoría')
            p.setFont("Helvetica-Bold", 14)
            p.drawString(inch, y, f"{cname}")
            y -= 18
            y = write_paragraph(
                f"<b>Sentimiento medio:</b> {cat.get('sentiment_avg', 0.0):.2f}", inch, y
            )
            y = write_paragraph(
                f"<b>Oportunidades:</b> " + ", ".join(cat.get('top_opportunities') or []), inch, y
            )
            y = write_paragraph(
                f"<b>Riesgos:</b> " + ", ".join(cat.get('top_risks') or []), inch, y
            )
            quotes = (cat.get('quotes') or [])
            if not quotes and corpus_map.get(cname):
                # usar 1-3 extractos del corpus como citas
                quotes = [t.get('excerpt','') for t in corpus_map.get(cname, [])[:3] if t.get('excerpt')]
            if quotes:
                y -= 6
                p.setFont("Helvetica-Bold", 12)
                p.drawString(inch, y, "Citas destacadas")
                y -= 16
                for q in quotes[:3]:
                    y = write_paragraph(f"\u201C{q}\u201D", inch, y)
                    y -= 6
            y -= 6
    except Exception:
        pass

    # Sección nueva: Intención del Usuario
    try:
        intents = data.get('user_intent') or {}
        dist = intents.get('distribution') or []
        exs = intents.get('examples') or {}
        if dist:
            p.setFont("Helvetica-Bold", 14)
            p.drawString(inch, y, "Intención del Usuario")
            y -= 22
            rows = [["Intención", "Menciones", "%"]] + [[d.get('intent',''), str(d.get('count','')), f"{d.get('pct',0):.1f}%"] for d in dist]
            tbl = Table(rows, colWidths=[2.2*inch, 1.2*inch, 1.0*inch])
            tbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
            ]))
            w, h = tbl.wrapOn(p, width - 2*inch, height)
            if y - h < inch:
                p.showPage(); y = height - inch
            tbl.drawOn(p, inch, y - h)
            y -= (h + 10)
            # ejemplos rápidos
            bullets = []
            for k in ["informacional", "comparativa", "transaccional"]:
                ex = exs.get(k) or []
                if ex:
                    bullets.append(f"<b>{k.title()}:</b> " + " | ".join(ex))
            if bullets:
                y = write_paragraph("<br/>".join(bullets), inch, y)
                y -= 8
    except Exception:
        pass

    # Sección 4 (legacy): Recomendaciones - opcional
    recom = content.get('recommendations') or {}
    if isinstance(recom, dict) and recom:
        p.setFont("Helvetica-Bold", 14)
        p.drawString(inch, y, recom.get('title', '4. Recomendaciones'))
        y -= 30
        y = write_paragraph(f"<b>Perspectiva de Mercado:</b><br/>{recom.get('market_outlook', 'N/A')}", inch, y)
        for lever in recom.get('strategic_levers', []):
            y = write_paragraph(f"<b>{lever.get('lever_title','')}", inch, y, font_size=12)
            y = write_paragraph(lever.get('description', ''), inch + 0.2*inch, y)
            actions = lever.get('recommended_actions', []) or []
            if actions:
                actions_text = "- " + "<br/>- ".join(actions)
                y = write_paragraph(f"<b>Acciones Recomendadas:</b><br/>{actions_text}", inch + 0.2*inch, y)
            y -= 15

    # Sección 4-bis: Correlaciones clave
    try:
        corr = data.get('correlation_events', [])
        if corr:
            if y < inch * 2:
                p.showPage(); y = height - inch
            p.setFont("Helvetica-Bold", 14)
            p.drawString(inch, y, "Análisis de Correlaciones")
            y -= 24
            rows = [["Fecha", "Competidor", "Menciones", "Caída Sent."]]
            for ev in corr:
                rows.append([
                    ev.get('date',''), ev.get('competitor',''),
                    str(ev.get('competitor_mentions','')),
                    str(ev.get('sentiment_drop','')),
                ])
            ctbl = Table(rows, colWidths=[1.2*inch, 2.0*inch, 1.2*inch, 1.2*inch])
            ctbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ]))
            w, h = ctbl.wrapOn(p, width - 2*inch, height)
            if y - h < inch:
                p.showPage(); y = height - inch
            ctbl.drawOn(p, inch, y - h)
            y -= (h + 20)
    except Exception:
        pass

    # Sección: Evolución de SOV por temas clave (series)
    try:
        topic_series = data.get('topic_sov_timeseries') or []
        if topic_series:
            if y < inch * 2:
                p.showPage(); y = height - inch
            p.setFont("Helvetica-Bold", 14)
            p.drawString(inch, y, "Evolución SOV por Temas Clave")
            y -= 18
            # Genera un gráfico simple con matplotlib para hasta 3 series
            try:
                import matplotlib.pyplot as _plt
                fig, ax = _plt.subplots(figsize=(6.8, 2.2))
                for serie in topic_series[:3]:
                    xs = list(range(1, len(serie.get('values') or []) + 1))
                    ax.plot(xs, serie.get('values') or [], label=str(serie.get('topic')))
                ax.set_ylabel('SOV (%)')
                ax.set_xlabel('Días del periodo')
                ax.legend(fontsize=7, loc='upper left', ncol=3)
                fig.tight_layout()
                img_buffer = io.BytesIO()
                _plt.savefig(img_buffer, format='PNG', dpi=150)
                _plt.close(fig)
                img_buffer.seek(0)
                img = ImageReader(img_buffer)
                p.drawImage(img, inch, y - 170, width=width - 2*inch, height=150, mask='auto')
                y -= 175
            except Exception:
                pass
    except Exception:
        pass

    # Sección: Resumen de Alertas Clave
    try:
        alerts = data.get('alerts_summary') or []
        if alerts:
            if y < inch * 2:
                p.showPage(); y = height - inch
            p.setFont("Helvetica-Bold", 14)
            p.drawString(inch, y, "Resumen de Alertas Clave")
            y -= 18
            bullets = "- " + "\n- ".join(alerts[:6])
            y = write_paragraph(bullets, inch, y)
            y -= 10
    except Exception:
        pass

    # Sección: Análisis Comparativo Destacado
    try:
        comp = data.get('comparative_highlight') or {}
        if comp:
            if y < inch * 2:
                p.showPage(); y = height - inch
            p.setFont("Helvetica-Bold", 14)
            p.drawString(inch, y, comp.get('title', 'Análisis Comparativo Destacado'))
            y -= 18
            # Mini gráfico barras 2 columnas
            labels = comp.get('metric_labels') or []
            values = comp.get('metric_values') or []
            try:
                import matplotlib.pyplot as _plt
                fig, ax = _plt.subplots(figsize=(4.0, 1.8))
                ax.bar(range(len(values)), values, color=['#1f77b4', '#ff7f0e'])
                ax.set_xticks(range(len(values)))
                ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
                ax.set_ylim(0, max(100, max(values or [0]) + 5))
                fig.tight_layout()
                img_buffer = io.BytesIO()
                _plt.savefig(img_buffer, format='PNG', dpi=150)
                _plt.close(fig)
                img_buffer.seek(0)
                img = ImageReader(img_buffer)
                p.drawImage(img, inch, y - 130, width=300, height=110, mask='auto')
                y -= 135
            except Exception:
                pass
            analysis_text = comp.get('analysis_text') or ''
            if analysis_text:
                y = write_paragraph(analysis_text, inch, y)
                y -= 8
    except Exception:
        pass

    # Sección: Análisis de Contenido Competitivo (benchmarking y oportunidades)
    try:
        items = data.get('content_competitive') or []
        if items:
            if y < inch * 2:
                p.showPage(); y = height - inch
            p.setFont("Helvetica-Bold", 14)
            p.drawString(inch, y, "Análisis de Contenido Competitivo y Oportunidades")
            y -= 18
            header = [["Competidor", "Debilidad Detectada", "#", "Acción de Contenido"]]
            rows = header + [[it.get('competitor',''), it.get('theme',''), str(it.get('count',0)), it.get('suggestion_title','')] for it in items[:8]]
            tbl = Table(rows, colWidths=[1.6*inch, 2.3*inch, 0.5*inch, 2.2*inch])
            tbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('ALIGN', (2,1), (2,-1), 'RIGHT'),
            ]))
            w, h = tbl.wrapOn(p, width - 2*inch, height)
            if y - h < inch:
                p.showPage(); y = height - inch
            tbl.drawOn(p, inch, y - h)
            y -= (h + 8)
            # texto guía
            first = items[0] if items else None
            if first and first.get('suggestion_text'):
                y = write_paragraph(first.get('suggestion_text'), inch, y)
                y -= 6
    except Exception:
        pass

    p.save()
    buffer.seek(0)
    return buffer


def create_nielsen_style_prompt(aggregated_data):
    """Crea el prompt avanzado para que la IA genere el contenido del informe."""
    prompt = f"""
    **ROL Y OBJETIVO:**
    Actúa como un Analista de Inteligencia de Mercado Senior para "The Core School", una escuela superior especializada en entretenimiento y artes audiovisuales. Tu misión es redactar un informe ejecutivo de alto impacto, similar a los que producen consultoras como Nielsen o IRI, basándote en los datos cuantitativos y cualitativos proporcionados. El informe debe ser analítico, prescriptivo y orientado a la toma de decisiones estratégicas para el equipo directivo.

    **CONTEXTO DE NEGOCIO:**
    The Core School compite con otras escuelas especializadas y universidades tradicionales por atraer a jóvenes interesados en carreras creativas. Los "padres" son una audiencia secundaria clave que influye en la decisión. El objetivo es aumentar la visibilidad, mejorar la reputación y optimizar la captación de alumnos.

    **DATOS DE MERCADO (Periodo: {aggregated_data['start_date']} a {aggregated_data['end_date']}):**

    1.  **MÉTRICAS PRINCIPALES (KPIs):**
        -   Visibilidad de Marca Global: {aggregated_data['kpis']['visibility_score']:.1f}%
        -   Evolución de Visibilidad (vs. periodo anterior): {aggregated_data['kpis']['visibility_delta']:+.1f} pts
        -   Sentimiento Promedio de la Marca: {aggregated_data['kpis']['sentiment_avg']:.2f} (escala -1 a 1)
        -   Share of Voice (SOV) vs. Competencia: {aggregated_data['kpis']['sov_score']:.1f}%

    2.  **ANÁLISIS DE LA COMPETENCIA:**
        -   Ranking de Share of Voice: {json.dumps(aggregated_data['competitor_ranking'], indent=2, ensure_ascii=False)}
        -   Temas Clave asociados a Competidores: {json.dumps(aggregated_data['competitor_themes'], indent=2, ensure_ascii=False)}

    3.  **ANÁLISIS TEMÁTICO (DIAGNÓSTICO INTERNO):**
        -   Temas con Sentimiento más Positivo: {json.dumps(aggregated_data['top_positive_themes'], indent=2, ensure_ascii=False)}
        -   Temas con Sentimiento más Negativo (Puntos de Dolor): {json.dumps(aggregated_data['top_negative_themes'], indent=2, ensure_ascii=False)}
        -   Tendencias Emergentes Detectadas: {json.dumps(aggregated_data['emerging_trends'], indent=2, ensure_ascii=False)}

    4.  **DATOS CUALITATIVOS (VOZ DEL MERCADO):**
        -   Oportunidades Estratégicas Clave (más frecuentes): {json.dumps(aggregated_data['top_opportunities'], indent=2, ensure_ascii=False)}
        -   Riesgos y Amenazas Principales (más frecuentes): {json.dumps(aggregated_data['top_risks'], indent=2, ensure_ascii=False)}
        -   Citas Textuales Representativas: {json.dumps(aggregated_data['key_quotes'], indent=2, ensure_ascii=False)}

    **ESTRUCTURA DEL INFORME (RESPONDE ÚNICAMENTE CON ESTE JSON):**
    {{
      "executive_summary": {{
        "title": "Informe Ejecutivo de Visibilidad e Inteligencia de Mercado",
        "period": "{aggregated_data['start_date']} - {aggregated_data['end_date']}",
        "headline": "Insight principal en una frase impactante (ej: 'Crecimiento en visibilidad impulsado por la reputación del programa de VFX, pero amenazado por la agresiva campaña de precios de la competencia').",
        "key_findings": ["Hallazgo principal sobre la visibilidad y reputación.", "Hallazgo clave sobre la posición competitiva y SOV.", "Hallazgo relevante sobre la percepción de la audiencia."],
        "overall_assessment": "Evaluación general del desempeño en el periodo, concluyendo si la posición de la marca ha mejorado, empeorado o se ha mantenido estable, y por qué."
      }},
      "strategic_levers": {{
        "title": "Palancas de Decisión y Acciones Recomendadas",
        "levers": [
          {{
            "lever_title": "Capitalizar en Oportunidad Clave (ej: Liderazgo en Formación Práctica)",
            "description": "Análisis profundo de la oportunidad más importante y por qué es crucial.",
            "recommended_actions": ["Acción de Marketing específica.", "Acción de Producto/Académica.", "Acción de Comunicación."]
          }},
          {{
            "lever_title": "Mitigar Riesgo Principal (ej: Percepción de Alto Coste)",
            "description": "Análisis del riesgo más significativo y su posible impacto.",
            "recommended_actions": ["Acción Financiera/Comercial.", "Acción de Contenido."]
          }}
        ]
      }},
      "correlation_analysis": {{
        "title": "Análisis de Correlaciones y Diagnóstico",
        "insights": ["Observación que conecta dos o más puntos de datos.", "Observación sobre un competidor.", "Observación sobre una debilidad o 'blind spot'."]
      }}
    }}
    """
    return prompt


def _create_pdf_report(content, data):
    """Construye un PDF en memoria a partir del contenido JSON generado por la IA."""
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    styles = getSampleStyleSheet()
    M = 0.75 * inch

    # Rutas y helpers de cabecera/pie
    project_root = os.path.dirname(os.path.dirname(__file__))
    logo_path = os.path.join(project_root, 'frontend', 'public', 'the-core-logo.png')

    def draw_header_footer():
        try:
            # Cabecera
            if os.path.exists(logo_path):
                p.drawImage(logo_path, M, height - M - 20, width=20, height=20, mask='auto')
            p.setFont("Helvetica-Bold", 10)
            p.drawString(M + 28, height - M - 8, f"{data.get('brand','')} — Informe Ejecutivo")
            p.setLineWidth(0.3)
            p.line(M, height - M - 24, width - M, height - M - 24)
            # Pie
            p.setFont("Helvetica", 9)
            p.line(M, M * 0.8, width - M, M * 0.8)
            p.drawString(M, M * 0.5, f"Periodo: {data['start_date']} a {data['end_date']}")
            p.drawRightString(width - M, M * 0.5, f"Página {p.getPageNumber()}")
        except Exception:
            pass

    def write_paragraph(text, x, y, style_name='Normal', font_size=10):
        style = styles[style_name]
        style.fontSize = font_size
        para = Paragraph(text.replace('\n', '<br/>'), style)
        w, h = para.wrapOn(p, width - x - inch, height)
        if y - h < inch:
            # Pie y cabecera de nueva página
            draw_header_footer()
            p.showPage()
            draw_header_footer()
            y = height - inch
        para.drawOn(p, x, y - h)
        return y - h - (font_size * 1.2)

    y = height - inch

    # Título y subtítulo
    draw_header_footer()
    p.setFont("Helvetica-Bold", 20)
    p.drawString(M + 30, y, content['executive_summary']['title'])
    y -= 25
    p.setFont("Helvetica", 11)
    p.drawString(M + 30, y, f"Periodo de Análisis: {data['start_date']} a {data['end_date']}")
    y -= 40

    # Headline
    y = write_paragraph(f"<i><b>Insight Principal:</b> {content['executive_summary']['headline']}</i>", M, y, font_size=12)
    y -= 20

    # KPIs principales
    k = data.get('kpis', {})
    kpi_lines = [
        f"Visibilidad de Marca: {k.get('visibility_score', 0):.1f}%",
        f"Evolución vs periodo anterior: {k.get('visibility_delta', 0):+.1f} pts",
        f"Sentimiento medio: {k.get('sentiment_avg', 0):.2f}",
        f"Share of Voice (SOV): {k.get('sov_score', 0):.1f}%",
    ]
    y = write_paragraph("<b>KPIs del período</b><br/>- " + "<br/>- ".join(kpi_lines), M, y)
    y -= 10

    # Gráfico: Tendencia de visibilidad (línea)
    try:
        ts = data.get('visibility_timeseries', [])
        if len(ts) >= 2:
            lp_w, lp_h = 420, 140
            lp = Drawing(lp_w, lp_h)
            points = [(i + 1, float(item.get('value', 0))) for i, item in enumerate(ts)]
            plot = LinePlot()
            plot.x = 40; plot.y = 28
            plot.height = lp_h - 48; plot.width = lp_w - 60
            plot.data = [points]
            plot.lines[0].strokeColor = colors.HexColor('#1f77b4')
            plot.lines[0].strokeWidth = 1.8
            plot.lines[0].symbol = makeMarker('Circle')
            plot.lines[0].symbol.size = 3
            xs = [x for x, _ in points]
            ys = [y for _, y in points]
            plot.xValueAxis.valueMin = min(xs)
            plot.xValueAxis.valueMax = max(xs)
            plot.yValueAxis.valueMin = 0
            plot.yValueAxis.valueMax = max(100, max(ys) + 5)
            plot.yValueAxis.labels.fontSize = 8
            plot.xValueAxis.labels.fontSize = 8
            lp.add(plot)
            renderPDF.draw(lp, p, M, y - lp_h)
            y -= lp_h + 10
    except Exception:
        pass

    # Resumen Ejecutivo
    p.setFont("Helvetica-Bold", 14)
    p.drawString(M, y, "1. Resumen Ejecutivo")
    y -= 30
    findings_text = "- " + "<br/>- ".join(content['executive_summary']['key_findings'])
    y = write_paragraph(f"<b>Hallazgos Clave:</b><br/>{findings_text}", M, y)
    y = write_paragraph(f"<b>Evaluación General:</b><br/>{content['executive_summary']['overall_assessment']}", M, y)
    y -= 20

    # Palancas de Decisión
    p.setFont("Helvetica-Bold", 14)
    p.drawString(M, y, "2. Palancas de Decisión Estratégicas")
    y -= 30
    for lever in content['strategic_levers']['levers']:
        y = write_paragraph(f"<b>{lever['lever_title']}</b>", M, y, font_size=12)
        y = write_paragraph(lever['description'], M + 0.2*inch, y)
        actions_text = "- " + "<br/>- ".join(lever['recommended_actions'])
        y = write_paragraph(f"<b>Acciones Recomendadas:</b><br/>{actions_text}", M + 0.2*inch, y)
        y -= 15

    # 3. Análisis de Correlaciones
    try:
        p.setFont("Helvetica-Bold", 14)
        p.drawString(M, y, "3. Correlaciones, Solapamientos y Diagnóstico")
        y -= 25
        corr_ins = content.get('correlation_analysis', {}).get('insights') or []
        if corr_ins:
            corr_text = "- " + "<br/>- ".join(corr_ins)
            y = write_paragraph(corr_text, M, y)
        y -= 10
        # Heatmap tema→sentimiento
        hm_buf = _generate_topic_sentiment_heatmap_image(data.get('topic_sentiment_correlations') or [])
        if hm_buf:
            img = ImageReader(hm_buf)
            img_h = 16 + 12 * min(12, len(data.get('topic_sentiment_correlations') or []))
            img_h = max(160, min(420, img_h))
            p.drawImage(img, M, y - img_h, width=width - 2*M, height=img_h, mask='auto')
            y -= img_h + 8
        # Sub-sección: Anomalías
        anomalies = (content.get('anomaly_analysis', {}) or {}).get('anomalies') or []
        if anomalies:
            p.setFont("Helvetica-Bold", 11)
            p.drawString(M, y, "Anomalías y Señales Tempranas")
            y -= 16
            an_text = "- " + "<br/>- ".join(anomalies[:6])
            y = write_paragraph(an_text, M, y)
            y -= 6
        # Mini-tabla: Días outlier (desde data)
        outliers = (data.get('outlier_days') or [])[:6]
        if outliers:
            p.setFont("Helvetica-Bold", 11)
            p.drawString(M, y, "Días Atípicos (z-score)")
            y -= 14
            table = [["Fecha", "Sentimiento", "z"]] + [[o.get('date'), str(o.get('sentiment')), str(o.get('z'))] for o in outliers]
            tbl = Table(table, colWidths=[1.6*inch, 1.2*inch, 0.8*inch])
            tbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
            ]))
            w, h = tbl.wrapOn(p, width - 2*M, 100)
            if y - h < inch:
                p.showPage(); y = height - inch
            tbl.drawOn(p, M, y - h)
            y -= (h + 8)
    except Exception:
        pass

    # 4. Ranking de competidores y gráfico SOV
    try:
        p.setFont("Helvetica-Bold", 14)
        p.drawString(M, y, "4. Competencia: Ranking SOV")
        y -= 22
        # Tabla simple
        table_x = M
        p.setFont("Helvetica-Bold", 10)
        p.drawString(table_x, y, "#")
        p.drawString(table_x + 15, y, "Marca")
        p.drawString(table_x + 220, y, "SOV")
        y -= 12
        p.setFont("Helvetica", 10)
        for r in data.get('competitor_ranking', [])[:8]:
            p.drawString(table_x, y, str(r.get('rank')))
            p.drawString(table_x + 15, y, str(r.get('name')))
            p.drawRightString(table_x + 260, y, str(r.get('sov')))
            y -= 12
        y -= 6
        # Pie chart SOV a la derecha
        chart_data = data.get('sov_chart_data', [])[:6]
        if chart_data:
            pie_w, pie_h = 220, 140
            d = Drawing(pie_w, pie_h)
            pie = Pie()
            pie.x = 60; pie.y = 10
            pie.width = 100; pie.height = 100
            pie.data = [max(0.1, float(it['sov'])) for it in chart_data]
            pie.labels = [it['name'] for it in chart_data]
            pie.sideLabels = True
            pie.slices.strokeWidth = 0.25
            pie.slices.popout = 0
            d.add(pie)
            renderPDF.draw(d, p, width - M - pie_w, y - pie_h + 18)
            y -= pie_h + 12
    except Exception:
        pass

    # 4-bis. Competidor líder vs. sentimiento
    try:
        comp = data.get('competitor_series') or {}
        dates = data.get('time_series', {}).get('dates') or []
        counts = comp.get('counts') or []
        sent = data.get('time_series', {}).get('sentiment') or []
        if comp.get('name') and counts:
            if y < inch * 2:
                p.showPage(); y = height - inch
            p.setFont("Helvetica-Bold", 12)
            p.drawString(M, y, f"Competidor líder vs. Sentimiento: {comp.get('name')}")
            y -= 16
            img_buf = _generate_competitor_vs_sentiment_image(dates, counts, sent)
            if img_buf:
                img = ImageReader(img_buf)
                p.drawImage(img, M, y - 180, width=width - 2*M, height=170, mask='auto')
                y -= 180
    except Exception:
        pass

    # 5. Solapamientos Riesgos-Competidores
    try:
        overlaps = data.get('overlap_events', [])
        if overlaps:
            if y < inch * 2:
                p.showPage(); y = height - inch
            p.setFont("Helvetica-Bold", 14)
            p.drawString(M, y, "5. Solapamientos: Riesgos y Picos de Competidores")
            y -= 22
            rows = [["Fecha", "Competidor", "Menciones", "Riesgos Citados"]]
            for ev in overlaps[:12]:
                rows.append([
                    ev.get('date',''), ev.get('competitor',''), str(ev.get('competitor_mentions','')),
                    ", ".join(ev.get('risks', [])[:3])
                ])
            otbl = Table(rows, colWidths=[1.2*inch, 2.0*inch, 1.2*inch, 3.0*inch])
            otbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('ALIGN', (2,1), (2,-1), 'RIGHT'),
            ]))
            w, h = otbl.wrapOn(p, width - 2*inch, height)
            if y - h < inch:
                p.showPage(); y = height - inch
            otbl.drawOn(p, M, y - h)
            y -= (h + 16)
    except Exception:
        pass

    # 6. Distribuciones (Data-Rich)
    try:
        dists = data.get('distributions') or {}
        if dists:
            if y < inch * 2:
                p.showPage(); y = height - inch
            p.setFont("Helvetica-Bold", 14)
            p.drawString(M, y, "6. Distribuciones y Sesgos de Origen")
            y -= 22
            # Conteos por Motor
            mb_engine = [["Motor", "Menciones"]] + (dists.get('mentions_by_engine') or [])
            t1 = Table(mb_engine, colWidths=[3.0*inch, 1.4*inch])
            t1.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('ALIGN', (1,1), (1,-1), 'RIGHT'),
            ]))
            w, h = t1.wrapOn(p, (width - 2*M) / 2 - 6, height)
            t1.drawOn(p, M, y - h)
            # Conteos por Tema
            mb_topic = [["Tema", "Menciones"]] + (dists.get('mentions_by_topic') or [])
            t2 = Table(mb_topic, colWidths=[3.0*inch, 1.4*inch])
            t2.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('ALIGN', (1,1), (1,-1), 'RIGHT'),
            ]))
            w2, h2 = t2.wrapOn(p, (width - 2*M) / 2 - 6, height)
            t2.drawOn(p, M + (width - 2*M) / 2 + 6, y - h2)
            y -= max(h, h2) + 10

            # Histograma de sentimiento (tabla compacta)
            hist = dists.get('sentiment_hist') or []
            img_buf = _generate_sentiment_histogram_image(hist)
            if img_buf:
                img = ImageReader(img_buf)
                p.drawImage(img, M, y - 160, width=width - 2*M, height=150, mask='auto')
                y -= 160
    except Exception:
        pass

    # --- NUEVO: Sección del Plan de Acción Estratégico ---
    p.showPage(); y = height - inch
    draw_header_footer()
    plan = content.get('strategic_action_plan', {})
    p.setFont("Helvetica-Bold", 14)
    p.drawString(M, y, plan.get('title', '4. Plan de Acción Estratégico'))
    y -= 30

    y = write_paragraph(f"<b>Perspectiva de Mercado:</b><br/>{plan.get('market_outlook', 'N/A')}", M, y)
    y -= 20

    recommendations = plan.get('strategic_recommendations', [])
    for i, rec in enumerate(recommendations):
        if y < inch * 2.5:
            p.showPage(); y = height - inch; draw_header_footer()
        p.setFont("Helvetica-Bold", 12)
        p.drawString(M, y, f"Recomendación #{i+1}: {rec.get('recommendation', 'N/A')}")
        y -= 20
        rec_text = f"<b>Detalles:</b> {rec.get('details', 'N/A')}<br/><br/>"
        rec_text += f"<b>KPIs de seguimiento:</b> {rec.get('kpis', 'N/A')}<br/>"
        rec_text += f"<b>Plazo estimado:</b> {rec.get('timeline', 'N/A')}<br/>"
        rec_text += f"<b>Prioridad:</b> {rec.get('priority', 'N/A')}"
        y = write_paragraph(rec_text, M + 0.2*inch, y)
        y -= 20

    # Cierre con cabecera/pie dibujados
    draw_header_footer()

    p.save()
    buffer.seek(0)
    return buffer


# --- Detección de marcas reutilizable ---
BRAND_SYNONYMS = {
    "The Core School": ["the core", "the core school", "thecore"],
    "U-TAD": ["u-tad", "utad"],
    "ECAM": ["ecam"],
    "TAI": ["tai"],
    "CES": ["ces"],
    "CEV": ["cev"],
    "FX Barcelona Film School": ["fx barcelona", "fx barcelona film school", "fx animation"],
    "Septima Ars": ["septima ars", "séptima ars"],
}

def _normalize_list(xs):
    out = []
    if not xs:
        return out
    try:
        for t in xs:
            s = str(t or '').lower().strip()
            if s:
                out.append(s)
    except Exception:
        pass
    return out

def _detect_brands(topics_list, resp_text, title_text, payload):
    found = []
    tlist = _normalize_list(topics_list)
    # key_topics: igualdad o substring
    for canon, alts in BRAND_SYNONYMS.items():
        for a in alts:
            if any(a == t or a in t for t in tlist):
                found.append(canon)
                break
    # payload.brands: igualdad o substring
    try:
        brands_payload = []
        if isinstance(payload, dict):
            brands_payload = _normalize_list((payload.get('brands') or []))
        for canon, alts in BRAND_SYNONYMS.items():
            for a in alts:
                if any(a == b or a in b for b in brands_payload):
                    if canon not in found:
                        found.append(canon)
                    break
    except Exception:
        pass
    # substrings en texto/título
    txt = f"{resp_text or ''} {title_text or ''}"
    for canon, alts in BRAND_SYNONYMS.items():
        if any(a in txt for a in alts):
            if canon not in found:
                found.append(canon)
    return found

# ───────────── Helpers estadísticos para el analista ─────────────
def _histogram(values, bins=10, range_min=-1.0, range_max=1.0):
    try:
        if not values:
            return []
        step = (range_max - range_min) / max(bins, 1)
        hist = [0 for _ in range(bins)]
        for v in values:
            if v is None:
                continue
            try:
                x = float(v)
            except Exception:
                continue
            idx = int((x - range_min) / step)
            if idx < 0:
                idx = 0
            if idx >= bins:
                idx = bins - 1
            hist[idx] += 1
        edges = [range_min + i * step for i in range(bins + 1)]
        return [{"bin": f"{edges[i]:.2f}..{edges[i+1]:.2f}", "count": hist[i]} for i in range(bins)]
    except Exception:
        return []


def _dict_counts(d):
    try:
        items = sorted(d.items(), key=lambda x: x[1], reverse=True)
        return [[str(k), int(v)] for k, v in items]
    except Exception:
        return []


def _topic_sentiment_correlation(topic_counts, topic_sent_sum):
    try:
        out = []
        for t, cnt in topic_counts.items():
            if cnt < 3:
                continue
            avg_s = topic_sent_sum.get(t, 0.0) / max(cnt, 1)
            out.append([t, float(f"{avg_s:.4f}")])
        out.sort(key=lambda x: x[1])
        # Devuelve colas: negativas y positivas
        left = out[:10]
        right = out[-10:] if len(out) > 10 else []
        return left + right
    except Exception:
        return []


def _detect_outlier_days(days_sorted, series_sentiment):
    try:
        if not days_sorted or not series_sentiment or len(series_sentiment) < 3:
            return []
        import statistics
        mu = statistics.mean(series_sentiment)
        sigma = statistics.pstdev(series_sentiment) or 0.0001
        outliers = []
        for d, s in zip(days_sorted, series_sentiment):
            z = (s - mu) / sigma
            if abs(z) >= 1.5:
                outliers.append({"date": d.strftime('%Y-%m-%d'), "sentiment": round(s, 3), "z": round(z, 2)})
        return outliers[:10]
    except Exception:
        return []

def ensure_taxonomy_table():
    """Crea la tabla de taxonomía si no existe."""
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS topic_taxonomy (
            id SERIAL PRIMARY KEY,
            brand TEXT NOT NULL,
            category TEXT NOT NULL,
            keywords JSONB DEFAULT '[]'::jsonb,
            emoji TEXT DEFAULT NULL,
            UNIQUE(brand, category)
        );
        """
    )
    conn.commit(); cur.close(); conn.close()

def get_category_catalog_for_brand(brand: str | None) -> list[str]:
    """Obtiene el catálogo de categorías para una brand desde topic_taxonomy.
    Si no hay filas, devuelve el catálogo por defecto (_CATEGORY_KEYWORDS).
    """
    try:
        ensure_taxonomy_table()
        brand_eff = brand or os.getenv('DEFAULT_BRAND', 'The Core School')
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT category FROM topic_taxonomy WHERE brand = %s", (brand_eff,))
        rows = cur.fetchall(); cur.close(); conn.close()
        return [r[0] for r in rows] if rows else list(_CATEGORY_KEYWORDS.keys())
    except Exception:
        return list(_CATEGORY_KEYWORDS.keys())

def parse_filters(request):
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')
    
    # Da prioridad a las fechas específicas si se proporcionan
    if start_date_str and end_date_str:
        start_date = parse_date(start_date_str)
        # Añadimos un día al end_date para incluir el día completo en la consulta
        end_date = parse_date(end_date_str) + timedelta(days=1) 
    else:
        # Lógica de fallback si no se especifican fechas
        range_param = request.args.get('range', '30d')
        end_date = datetime.now()
        if range_param == '24h': start_date = end_date - timedelta(hours=24)
        elif range_param == '7d': start_date = end_date - timedelta(days=7)
        elif range_param == '90d': start_date = end_date - timedelta(days=90)
        else: start_date = end_date - timedelta(days=30)
    
    model = request.args.get('model', 'all')
    source = request.args.get('source', 'all')
    topic = request.args.get('topic', 'all')
    brand = request.args.get('brand', os.getenv('DEFAULT_BRAND', 'The Core School'))
    sentiment = request.args.get('sentiment', 'all')
    hide_bots = request.args.get('hideBots', '0') == '1'
    status_param = request.args.get('status', 'active')
    
    print(f"DEBUG: Filtrando desde {start_date} hasta {end_date}") # Para depuración
    
    return {
        'start_date': start_date,
        'end_date': end_date,
        'model': model,
        'source': source,
        'topic': topic,
        'brand': brand,
        'sentiment': sentiment,
        'hide_bots': hide_bots,
        'status': status_param,
    }


# --- Normalización y agrupación de Topics ---

def _strip_accents(text: str) -> str:
    if not text:
        return ''
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in nfkd if not unicodedata.combining(c)])


def _normalize_topic_key(text: str) -> str:
    txt = _strip_accents((text or '').strip().lower())
    txt = re.sub(r"[\-_]+", " ", txt)
    txt = re.sub(r"[^a-z0-9\s]+", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


_TOPIC_SYNONYMS = {
    # canonical_key: list of alternative keys
    "target audience research": ["audience research", "research audience", "research target audience"],
    "motivation triggers": ["motivation trigger", "triggers", "motivational triggers"],
    "motivations": ["motivations analysis", "motivation analysis"],
    "digital trends": ["trends digital", "online trends"],
    "industry perception": ["perception industry", "brand perception"],
    "competitive analysis": ["competitor analysis", "competition analysis"],
    "competitor benchmark": ["competitor benchmarking", "benchmark competitors"],
    "brand monitoring": ["monitoring brand"],
    "share of voice": ["sov", "voice share"],
    "industry buzz": ["buzz industry"],
    "trends analysis": ["trends", "trend analysis"],
    "employment outcomes": ["employability", "employment"],
    "job market": ["labour market", "labor market"],
    "brand partnerships": ["partnerships", "partnerships brand"],
    "reputation drivers": ["drivers reputation"],
    "student voice": ["students voice"],
    "student expectations": ["expectations students", "students expectations"],
    "innovation perception": ["perception innovation"],
    "future outlook": ["outlook future"],
}


def canonicalize_topic(topic: str) -> str:
    """Devuelve un nombre de topic canónico y legible para agrupar variantes similares."""
    key = _normalize_topic_key(topic or '')
    if not key:
        return 'Uncategorized'

    # Exact matches from synonyms
    for canonical, alts in _TOPIC_SYNONYMS.items():
        if key == canonical:
            return canonical.title()
        if key in alts:
            return canonical.title()

    # Fuzzy match against canonical names
    best_name = None
    best_score = 0.0
    for canonical in list(_TOPIC_SYNONYMS.keys()):
        score = SequenceMatcher(None, key, canonical).ratio()
        if score > best_score:
            best_score = score
            best_name = canonical
    if best_score >= 0.88 and best_name:
        return best_name.title()

    # Fallback: Title Case of cleaned text
    return ' '.join(w.capitalize() for w in key.split()) or 'Uncategorized'


# Nueva función: agrupar topics con IA en categorías estratégicas
def group_topics_with_ai(topics_rows: list[dict]) -> list[dict]:
    try:
        topics_list_str = "\n".join([f"- {t.get('topic','')} | count={t.get('count',0)} | avg_sentiment={t.get('avg_sentiment',0.0):.2f}" for t in topics_rows])
        
        # --- INICIO DE LA MODIFICACIÓN ---
        # El nuevo prompt con categorías más granulares y mejores instrucciones
        prompt = f"""
Eres un analista de inteligencia de mercado para "The Core School". Tu misión es agrupar una lista de temas en las siguientes categorías estratégicas predefinidas. Sé muy estricto y preciso.

**Categorías Predefinidas (USA SOLO ESTAS):**
- **Menciones de Marca Propia:** Exclusivamente para temas que se refieren a "The Core School".
- **Menciones de Competidores Directos:** Para otras escuelas de cine y audiovisual (ej. U-tad, ECAM, TAI, ESCAC).
- **Menciones de Universidades Tradicionales:** Para universidades generalistas que ofrecen grados relacionados (ej. Complutense, Rey Juan Carlos).
- **Programas y Grados:** Temas sobre áreas de estudio específicas (ej. "dirección de cine", "ingeniería de software", "comunicación audiovisual").
- **Becas y Admisiones:** Temas sobre el proceso de entrada y costes (ej. "becas", "precio", "admisiones").
- **Salidas Profesionales:** Temas relacionados con el empleo post-graduación (ej. "salidas laborales", "empleabilidad").
- **Temas Generales del Sector:** Conversaciones amplias sobre la industria (ej. "sector audiovisual", "tendencias cine").

**Instrucciones:**
1.  Asigna CADA tema de la lista a una de las 7 categorías.
2.  Si un tema no encaja claramente, asígnalo a "Temas Generales del Sector".
3.  Calcula el sentimiento medio y las ocurrencias totales para cada categoría.
4.  Devuelve el resultado ÚNICAMENTE en formato JSON, sin texto adicional.

**Formato de Salida JSON Esperado:**
```json
[
  {{
    "group_name": "Menciones de Competidores Directos",
    "avg_sentiment": 0.65,
    "total_occurrences": 15,
    "topics": [
      {{ "topic": "u-tad", "count": 8, "avg_sentiment": 0.7 }},
      {{ "topic": "ecam", "count": 7, "avg_sentiment": 0.5 }}
    ]
  }}
]
```
Lista de Temas a Analizar:
{topics_list_str}
"""
# --- FIN DE LA MODIFICACIÓN ---

        raw = fetch_response(prompt, model="gpt-4o-mini", temperature=0.0)
        
        # Extraer el bloque JSON de la respuesta
        import re as _re
        match = _re.search(r"\[\s*\{[\s\S]*\}\s*\]", raw)
        if match:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return data
        return []
    except Exception:
        # En caso de error, devuelve una estructura vacía para no romper el frontend.
        return []

# Categorías temáticas para agrupar prompts similares (ES/EN)
_CATEGORY_KEYWORDS = {
    "Audience & Research": [
        "audience", "público", "publico", "target", "investigacion", "investigación", "research"
    ],
    "Motivation & Triggers": [
        "motivacion", "motivación", "motivation", "trigger", "triggers", "detonante", "inspiracion", "inspiración"
    ],
    "Parents & Family Concerns": [
        "padres", "familia", "parents", "preocupaciones", "concerns", "temores"
    ],
    "Competition & Benchmarking": [
        "competencia", "competidores", "competitor", "benchmark", "comparacion", "comparación", "competitive"
    ],
    "Brand & Reputation": [
        "marca", "brand", "reputacion", "reputación", "monitoring", "percepcion", "percepción"
    ],
    "Digital Trends & Marketing": [
        "tendencias", "trends", "search", "busquedas", "búsquedas", "marketing", "digital", "social", "plataformas"
    ],
    "Industry & Market": [
        "industria", "industry", "mercado", "market", "buzz"
    ],
    "Students & Experience": [
        "estudiantes", "students", "alumnos", "experiencia", "expectativas", "voice"
    ],
    "Innovation & Technology": [
        "innovacion", "innovación", "ia", "ai", "vr", "ar", "tecnologia", "tecnología", "produccion virtual", "virtual"
    ],
    "Employment & Jobs": [
        "empleabilidad", "empleo", "empleos", "trabajos", "jobs", "salarios", "job market", "outcomes"
    ],
    "Share of Voice & Monitoring": [
        "share of voice", "sov", "voz", "monitoring"
    ],
    "Future Outlook & Trends": [
        "futuro", "future", "outlook"
    ],
    "Partnerships & Collaborations": [
        "colaboraciones", "partnerships", "partners", "alianzas"
    ],
    # Nuevas categorías
    "Admissions & Enrollment": [
        "admisiones", "admission", "inscripcion", "inscripción", "matricula", "matrícula", "proceso", "requisitos",
        "plazos", "solicitud"
    ],
    "Scholarships & Cost": [
        "becas", "beca", "coste", "costo", "precio", "prices", "financiacion", "financiación", "roi", "retorno"
    ],
    "Curriculum & Programs": [
        "curriculo", "currículo", "curriculum", "plan de estudios", "programas", "asignaturas", "grados", "fp",
        "masters", "másteres", "itinerarios"
    ],
    "Alumni & Success Stories": [
        "alumni", "egresados", "casos de exito", "casos de éxito", "testimonios", "trayectorias", "salidas"
    ],
    "Campus & Facilities": [
        "campus", "instalaciones", "equipamiento", "platós", "estudios", "laboratorios", "recursos"
    ],
    "Events & Community": [
        "eventos", "open day", "jornadas", "ferias", "festival", "comunidad", "networking", "meetup"
    ],
    # Nuevo: categoría específica para comunicación audiovisual y media
    "Audiovisual & Media": [
        "comunicacion audiovisual", "comunicación audiovisual", "audiovisual", "cine", "rodaje",
        "montaje", "postproduccion", "postproducción", "sonido", "vfx", "fx", "efectos visuales",
        "animacion", "animación", "guion", "guión", "fotografia", "fotografía", "produccion",
        "producción", "realizacion", "realización", "edicion", "edición"
    ],
    # Más tecnología/ingeniería para evitar solapamientos genéricos
    "Innovation & Technology": [
        "innovacion", "innovación", "tecnologia", "tecnología", "ia", "ai", "vr", "ar",
        "realidad virtual", "realidad aumentada", "machine learning", "big data", "iot", "ciencia de datos",
        "transformacion digital", "transformación digital"
    ],
    # Categoría específica si se desea separar de Innovación
    "Engineering & Software": [
        "ingenieria de software", "ingeniería de software", "desarrollo de software",
        "programacion", "programación", "programacion de videojuegos", "desarrollo backend",
        "desarrollo frontend", "devops", "arquitectura de software", "sistemas distribuidos",
        "bases de datos", "algoritmos", "computer science"
    ],
}


def _tokenize(text: str) -> set:
    t = _normalize_topic_key(text)
    return set(re.findall(r"[a-z0-9]+", t))


def categorize_prompt(topic: str | None, query_text: str | None) -> str:
    """Asigna categoría usando la versión detallada (reglas + scoring)."""
    try:
        detailed = categorize_prompt_detailed(topic, query_text)
        return detailed.get("category") or (topic or '')
    except Exception:
        return (topic or '')


def categorize_prompt_detailed(topic: str | None, query_text: str | None) -> dict:
    """Versión detallada: devuelve categoría, puntuación, alternativas y sugerencia.
    Regla: combina keywords de taxonomía, similitud con nombre de categoría y patrones de frases.
    """
    brand = request.args.get('brand') or os.getenv('DEFAULT_BRAND', 'The Core School')

    # 1) Cargar taxonomía/keywords
    try:
        ensure_taxonomy_table()
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT category, COALESCE(keywords,'[]'::jsonb) FROM topic_taxonomy WHERE brand = %s", (brand,))
        rows = cur.fetchall(); cur.close(); conn.close()
        categories_source = { r[0]: list(r[1] or []) for r in rows } if rows else _CATEGORY_KEYWORDS
    except Exception:
        categories_source = _CATEGORY_KEYWORDS

    full_text = (topic or '') + ' ' + (query_text or '')
    tokens = _tokenize(full_text)

    # 2) Patrones complementarios (ES/EN)
    PATTERNS = [
        (r"alumni|casos\s+de\s+exito|éxito|testimonios|trayectorias", "Alumni & Success Stories"),
        # Programas: consultas de intención de estudio y centros de formación
        (r"(d[oó]nde\s+estudiar|centros?\s+de\s+formaci[oó]n|grado\s+en|grados?\s+de|m[aá]ster(?:es)?\s+en)", "Curriculum & Programs"),
        (r"plan\s+de\s+estudios|curriculum|asignaturas|programas", "Curriculum & Programs"),
        (r"empleo|empleabilidad|salidas|profesiones|trabajo|job\s*market", "Employment & Jobs"),
        (r"becas|precio|coste|costo|financiaci[oó]n|roi", "Scholarships & Cost"),
        # Benchmarking: comparativas explícitas tipo "mejor universidad/centro/escuela" o "X vs Y"
        (r"(mejor\s+(universidad|centro|escuela)|\bvs\b|comparar|comparaci[oó]n)", "Competition & Benchmarking"),
        (r"competencia|competidores|benchmark", "Competition & Benchmarking"),
        (r"marca|reputaci[oó]n|monitor", "Brand & Reputation"),
        (r"tendencias|digital|marketing|redes", "Digital Trends & Marketing"),
        (r"campus|instalaciones|recursos", "Campus & Facilities"),
        (r"innovaci[oó]n|tecnolog[ií]a|ia|ai|vr|ar|programaci[oó]n|software|ingenier[íi]a", "Engineering & Software"),
        (r"estudiantes|experiencia|expectativas", "Students & Experience"),
        (r"padres|familia|preocupaciones", "Parents & Family Concerns"),
        (r"comunicaci[oó]n\s+audivisual|comunicaci[oó]n\s+aud[ií]visual|cine|rodaje|montaje|postproducci[oó]n|sonido|vfx|animaci[oó]n|gu[ií]on|fotograf[ií]a|realizaci[oó]n|edici[oó]n", "Audiovisual & Media"),
    ]

    import re
    bonus_pattern: dict[str, int] = {}
    q = full_text.lower()
    for rx, cat in PATTERNS:
        if re.search(rx, q):
            # Los patrones tienen prioridad fuerte
            bonus_pattern[cat] = bonus_pattern.get(cat, 0) + 5

    # Desambiguación temprana entre Audiovisual & Media y Engineering & Software
    audiovisual_terms = [
        "comunicacion audiovisual", "comunicación audiovisual", "cine", "rodaje", "montaje", "postproduccion",
        "postproducción", "sonido", "vfx", "animacion", "animación", "guion", "guión", "fotografia",
        "fotografía", "produccion", "producción", "realizacion", "realización", "edicion", "edición"
    ]
    software_terms = [
        "ingenieria", "ingeniería", "software", "programacion", "programación", "desarrollo", "backend",
        "frontend", "devops", "algoritmos", "bases de datos", "sistemas", "arquitectura"
    ]
    aud_hits = sum(1 for t in audiovisual_terms if t in q)
    soft_hits = sum(1 for t in software_terms if t in q)
    if aud_hits >= soft_hits + 2 and aud_hits >= 2:
        return {
            "category": "Audiovisual & Media",
            "confidence": 0.85,
            "alternatives": [{"category": "Engineering & Software", "score": 0.5}],
            "suggestion": None,
            "suggestion_is_new": False,
            "closest_existing": "Audiovisual & Media",
            "closest_score": 0.9,
            "is_close_match": True,
        }
    if soft_hits >= aud_hits + 2 and soft_hits >= 2:
        return {
            "category": "Engineering & Software",
            "confidence": 0.85,
            "alternatives": [{"category": "Audiovisual & Media", "score": 0.5}],
            "suggestion": None,
            "suggestion_is_new": False,
            "closest_existing": "Engineering & Software",
            "closest_score": 0.9,
            "is_close_match": True,
        }

    # 3) Scoring por categoría
    from difflib import SequenceMatcher
    scores: list[tuple[str, float]] = []
    for cat, kw_list in categories_source.items():
        score = 0.0
        # coincidencia de keywords
        for kw in kw_list:
            kwt = _tokenize(kw)
            if kwt & tokens:
                score += 2.0
        # similitud fuzzy con nombre
        score += SequenceMatcher(None, _normalize_topic_key(' '.join(tokens)), _normalize_topic_key(cat)).ratio() * 1.5
        # bonus por patrones
        score += bonus_pattern.get(cat, 0)
        scores.append((cat, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    best_cat, best_score = (scores[0] if scores else ((topic or ''), 0.0))
    second_score = scores[1][1] if len(scores) > 1 else 0.0
    confidence = 0.0
    if best_score > 0:
        # normalización simple en función de separación y magnitud
        confidence = min(1.0, max(0.0, (best_score - second_score + 1) / (best_score + 1)))

    suggestion = None
    suggestion_is_new = False
    # similitud a categorías existentes a partir del texto del prompt
    base_text = (query_text or topic or '')
    known = list(categories_source.keys())
    closest_existing = None
    closest_score = 0.0
    for cat in known:
        s = SequenceMatcher(None, _normalize_topic_key(base_text), _normalize_topic_key(cat)).ratio()
        if s > closest_score:
            closest_score = s; closest_existing = cat
    is_close_match = bool(closest_existing and closest_score >= 0.75)
    # Si la confianza es baja o no hay patrones claros, sugerir crear tópico a partir de frase clave
    if confidence < 0.5:
        # heurística: Usa primera coincidencia de PATTERNS como etiqueta si no está ya
        sug = None
        for _, cat in PATTERNS:
            if bonus_pattern.get(cat):
                sug = cat
                break
        if not sug:
            # fallback: usa canónico del topic/consulta
            sug = (topic or query_text or 'Uncategorized')
        # si hay un existente muy similar, preferirlo
        if is_close_match:
            suggestion = closest_existing
            suggestion_is_new = False
        else:
            suggestion_is_new = sug not in set(known)
            suggestion = sug

    return {
        "category": best_cat,
        "confidence": round(float(confidence), 2),
        "alternatives": [{"category": c, "score": round(float(s), 2)} for c, s in scores[:5]],
        "suggestion": suggestion,
        "suggestion_is_new": suggestion_is_new,
        "closest_existing": closest_existing,
        "closest_score": round(float(closest_score), 2),
        "is_close_match": is_close_match,
    }


# --- ENDPOINTS DE LA API ---
@app.route('/api/taxonomy', methods=['GET'])
def list_taxonomy():
    try:
        brand = request.args.get('brand') or os.getenv('DEFAULT_BRAND', 'The Core School')
        ensure_taxonomy_table()
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT id, category, COALESCE(keywords,'[]'::jsonb), COALESCE(emoji,'') FROM topic_taxonomy WHERE brand = %s ORDER BY category", (brand,))
        rows = cur.fetchall(); cur.close(); conn.close()
        items = [{"id": r[0], "category": r[1], "keywords": r[2] or [], "emoji": r[3] or None} for r in rows]
        return jsonify({"brand": brand, "items": items})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/taxonomy', methods=['POST'])
def create_taxonomy_item():
    try:
        payload = request.get_json(silent=True) or {}
        brand = payload.get('brand') or os.getenv('DEFAULT_BRAND', 'The Core School')
        category = payload.get('category'); keywords = payload.get('keywords', []); emoji = payload.get('emoji')
        if not category:
            return jsonify({"error": "category is required"}), 400
        ensure_taxonomy_table()
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("INSERT INTO topic_taxonomy (brand, category, keywords, emoji) VALUES (%s,%s,%s,%s) RETURNING id", (brand, category, json.dumps(keywords), emoji))
        new_id = cur.fetchone()[0]; conn.commit(); cur.close(); conn.close()
        return jsonify({"id": new_id, "brand": brand, "category": category, "keywords": keywords, "emoji": emoji}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/taxonomy/<int:item_id>', methods=['PATCH'])
def update_taxonomy_item(item_id: int):
    try:
        payload = request.get_json(silent=True) or {}
        fields = []; values = []
        if 'category' in payload: fields.append('category = %s'); values.append(payload['category'])
        if 'keywords' in payload: fields.append('keywords = %s'); values.append(json.dumps(payload['keywords']))
        if 'emoji' in payload: fields.append('emoji = %s'); values.append(payload['emoji'])
        if not fields:
            return jsonify({"error": "no fields to update"}), 400
        sql = f"UPDATE topic_taxonomy SET {', '.join(fields)} WHERE id = %s RETURNING id, brand, category, keywords, emoji"
        conn = get_db_connection(); cur = conn.cursor(); cur.execute(sql, tuple(values + [item_id]))
        row = cur.fetchone(); conn.commit(); cur.close(); conn.close()
        if not row:
            return jsonify({"error": "not found"}), 404
        return jsonify({"id": row[0], "brand": row[1], "category": row[2], "keywords": row[3], "emoji": row[4]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/taxonomy/<int:item_id>', methods=['DELETE'])
def delete_taxonomy_item(item_id: int):
    try:
        conn = get_db_connection(); cur = conn.cursor(); cur.execute("DELETE FROM topic_taxonomy WHERE id = %s", (item_id,))
        deleted = cur.rowcount; conn.commit(); cur.close(); conn.close()
        if deleted == 0:
            return jsonify({"error": "not found"}), 404
        return jsonify({"message": f"taxonomy item {item_id} deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        ensure_taxonomy_table()
        conn = get_db_connection()
        conn.close()
        safe_cfg = {k: ("***" if k == "password" else v) for k, v in DB_CONFIG.items()}
        return jsonify({"status": "healthy", "db": safe_cfg})
    except Exception as e:
        safe_cfg = {k: ("***" if k == "password" else v) for k, v in DB_CONFIG.items()}
        return jsonify({"status": "unhealthy", "error": str(e), "db": safe_cfg}), 500

@app.route('/api/mentions', methods=['GET'])
def get_mentions():
    """Obtener menciones con todos los campos enriquecidos."""
    try:
        filters = parse_filters(request)
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Usamos '<' en la fecha final para ser precisos
        # Construcción dinámica de filtros
        where_clauses = ["m.created_at >= %s AND m.created_at < %s"]
        params = [filters['start_date'], filters['end_date']]
        if filters.get('model') and filters['model'] != 'all':
            where_clauses.append("m.engine = %s")
            params.append(filters['model'])
        if filters.get('source') and filters['source'] != 'all':
            where_clauses.append("m.source = %s")
            params.append(filters['source'])
        if filters.get('topic') and filters['topic'] != 'all':
            where_clauses.append("COALESCE(q.category, q.topic) = %s")
            params.append(filters['topic'])
        # status filter (default active)
        if filters.get('status') and filters['status'] != 'all':
            where_clauses.append("COALESCE(m.status, 'active') = %s")
            params.append(filters['status'])
        # hide bots
        if filters.get('hide_bots'):
            where_clauses.append("COALESCE(m.is_bot, FALSE) = FALSE")

        where_sql = " AND ".join(where_clauses)

        base_query = f"""
        SELECT 
            m.id, m.engine, m.source, m.response, m.sentiment, m.created_at,
            m.key_topics, LOWER(COALESCE(m.source_title,'')) AS title,
            i.payload
        FROM mentions m
        JOIN queries q ON m.query_id = q.id
        LEFT JOIN insights i ON i.id = m.generated_insight_id
        WHERE {where_sql}
        ORDER BY m.created_at DESC LIMIT %s OFFSET %s
        """
        cur.execute(base_query, params + [limit, offset])
        rows = cur.fetchall()
        
        mentions = []
        for row in rows:
            mentions.append({
                "id": row[0], "engine": row[1], "source": row[2],
                "response": row[3], "sentiment": float(row[4] or 0.0),
                "created_at": row[5].isoformat() if row[5] else None,
                "key_topics": row[6] or [],
                "title": row[7], "source_url": row[8],
                "language": row[9] or "unknown"
            })

        count_query = f"""
        SELECT COUNT(*)
        FROM mentions m
        JOIN queries q ON m.query_id = q.id
        WHERE {where_sql}
        """
        cur.execute(count_query, params)
        total = cur.fetchone()[0] or 0
        
        cur.close()
        conn.close()
        
        return jsonify({ "mentions": mentions, "pagination": { "total": total, "limit": limit, "offset": offset, "has_next": offset + limit < total } })
    except Exception as e:
        print(f"Error en get_mentions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/visibility', methods=['GET'])
def get_visibility():
    """Visibilidad = menciones de la marca / total de menciones (con filtros aplicados).
    Devuelve score del periodo, delta y serie temporal en porcentaje (0–100%).
    Parámetros opcionales:
      - granularity: 'day' (por defecto) o 'hour' (que agrupa por poll_id).
    """
    try:
        filters = parse_filters(request)
        brand = request.args.get('brand') or os.getenv('DEFAULT_BRAND', 'The Core School')
        granularity = request.args.get('granularity', 'day').lower()
        conn = get_db_connection(); cur = conn.cursor()

        where = ["m.created_at >= %s AND m.created_at < %s"]
        params = [filters['start_date'], filters['end_date']]
        if filters.get('model') and filters['model'] != 'all':
            where.append("m.engine = %s"); params.append(filters['model'])
        if filters.get('source') and filters['source'] != 'all':
            where.append("m.source = %s"); params.append(filters['source'])
        if filters.get('topic') and filters['topic'] != 'all':
            where.append("COALESCE(q.category, q.topic) = %s"); params.append(filters['topic'])
        where_sql = " AND ".join(where)

        synonyms = [s.lower() for s in BRAND_SYNONYMS.get(brand, [brand.lower()])]
        like_patterns = [f"%{s}%" for s in synonyms]

        series = []
        num_sum = 0
        den_sum = 0

        if granularity == 'hour':
            query = f"""
                WITH poll_stats AS (
                    SELECT
                        poll_id,
                        MIN(m.created_at) as poll_start_time,
                        COUNT(*) as total_mentions,
                        COUNT(CASE WHEN
                            EXISTS (
                                SELECT 1 FROM jsonb_array_elements_text(COALESCE(to_jsonb(m.key_topics),'[]'::jsonb)) kt
                                WHERE LOWER(TRIM(kt)) = ANY(%s)
                            ) OR LOWER(COALESCE(m.response,'')) LIKE ANY(%s)
                              OR LOWER(COALESCE(m.source_title,'')) LIKE ANY(%s)
                              OR EXISTS (
                                SELECT 1 FROM jsonb_array_elements(COALESCE(i.payload->'brands','[]'::jsonb)) b
                                WHERE LOWER(TRIM(CASE WHEN jsonb_typeof(b)='object' THEN COALESCE(b->>'name','') ELSE TRIM(BOTH '"' FROM b::text) END)) = ANY(%s)
                              )
                        THEN 1 END) as brand_mentions
                    FROM mentions m
                    JOIN queries q ON m.query_id = q.id
                    LEFT JOIN insights i ON i.id = m.generated_insight_id
                    WHERE {where_sql} AND m.poll_id IS NOT NULL
                    GROUP BY poll_id
                )
                SELECT
                    poll_start_time,
                    total_mentions,
                    brand_mentions,
                    (brand_mentions::float / GREATEST(total_mentions, 1)) * 100.0 as visibility_pct
                FROM poll_stats
                ORDER BY poll_start_time;
            """
            cur.execute(query, tuple([synonyms, like_patterns, like_patterns, synonyms] + params))
            rows = cur.fetchall()

            # Serie por hora convertida a zona Europe/Madrid
            tz_madrid = pytz.timezone('Europe/Madrid')
            series = []
            for row in rows:
                dt_utc = row[0]
                try:
                    dt_madrid = (dt_utc.replace(tzinfo=pytz.UTC) if dt_utc.tzinfo is None else dt_utc.astimezone(pytz.UTC)).astimezone(tz_madrid)
                except Exception:
                    dt_madrid = dt_utc
                series.append({"date": dt_madrid.isoformat(), "value": round(row[3], 1)})

            # Score del periodo (ponderado por volumen)
            pooled_total = sum(int(r[1] or 0) for r in rows)
            pooled_brand = sum(int(r[2] or 0) for r in rows)
            num_sum = pooled_brand
            den_sum = pooled_total

        else: # granularidad 'day'
            cur.execute(f"SELECT DATE(m.created_at) AS d, COUNT(*) FROM mentions m JOIN queries q ON m.query_id = q.id WHERE {where_sql} GROUP BY DATE(m.created_at) ORDER BY DATE(m.created_at)", tuple(params))
            total_rows = cur.fetchall()

            cur.execute(f"""
                SELECT DATE(m.created_at) AS d, COUNT(*)
                FROM mentions m
                JOIN queries q ON m.query_id = q.id
                LEFT JOIN insights i ON i.id = m.generated_insight_id
                WHERE {where_sql}
                  AND (
                    EXISTS (
                      SELECT 1 FROM jsonb_array_elements_text(COALESCE(to_jsonb(m.key_topics),'[]'::jsonb)) kt
                      WHERE LOWER(TRIM(kt)) = ANY(%s)
                    )
                    OR LOWER(COALESCE(m.response,'')) LIKE ANY(%s)
                    OR LOWER(COALESCE(m.source_title,'')) LIKE ANY(%s)
                    OR EXISTS (
                      SELECT 1 FROM jsonb_array_elements(COALESCE(i.payload->'brands','[]'::jsonb)) b
                      WHERE LOWER(TRIM(CASE WHEN jsonb_typeof(b)='object' THEN COALESCE(b->>'name','') ELSE TRIM(BOTH '"' FROM b::text) END)) = ANY(%s)
                    )
                  )
                GROUP BY DATE(m.created_at)
                ORDER BY DATE(m.created_at)
            """, tuple(params + [synonyms, like_patterns, like_patterns, synonyms]))
            core_rows = cur.fetchall()

            from datetime import timedelta
            core_by_day = {d: int(c or 0) for d, c in core_rows}
            total_by_day = {d: int(c or 0) for d, c in total_rows}
            start_day = filters['start_date'].date()
            end_day = filters['end_date'].date()
            day = start_day
            from datetime import datetime as _dt
            while day <= end_day:
                num = int(core_by_day.get(day, 0))
                den = int(total_by_day.get(day, 0))
                num_sum += num
                den_sum += den
                pct = round((num / max(den, 1)) * 100.0, 1)
                tz_madrid = pytz.timezone('Europe/Madrid')
                ts_dt = _dt.combine(day, _dt.min.time()).replace(tzinfo=pytz.UTC).astimezone(tz_madrid)
                series.append({"date": ts_dt.isoformat(), "ts": int(ts_dt.timestamp() * 1000), "value": pct})
                day += timedelta(days=1)

        visibility_score = round((num_sum / max(den_sum, 1)) * 100.0, 1)

        n = len(series)
        delta = 0.0
        if n >= 2:
            mid = n // 2
            first = sum(p['value'] for p in series[:mid]) / max(mid, 1)
            second = sum(p['value'] for p in series[mid:]) / max(n - mid, 1)
            delta = round(second - first, 1)

        cur.close(); conn.close()
        return jsonify({"visibility_score": visibility_score, "delta": delta, "series": series, "granularity": granularity})
    except Exception as e:
        print(f"Error en get_visibility: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/visibility/ranking', methods=['GET'])
def get_visibility_ranking():
    """Ranking de visibilidad por marca.
    Para cada marca detectada en respuestas (menciones), calcula
    porcentaje = (# de respuestas en las que aparece la marca) / (total de respuestas)
    con los filtros aplicados (fecha, modelo, source, topic). No filtra por brand
    para que funcione con cualquier cliente/competencia.
    """
    try:
        filters = parse_filters(request)
        conn = get_db_connection(); cur = conn.cursor()

        where = ["m.created_at >= %s AND m.created_at < %s"]
        params = [filters['start_date'], filters['end_date']]
        if filters.get('model') and filters['model'] != 'all':
            where.append("m.engine = %s"); params.append(filters['model'])
        if filters.get('source') and filters['source'] != 'all':
            where.append("m.source = %s"); params.append(filters['source'])
        # Filtro directo por topic (desde queries)
        if filters.get('topic') and filters['topic'] != 'all':
            where.append("COALESCE(q.category, q.topic) = %s"); params.append(filters['topic'])
        where_sql = " AND ".join(where)

        # Helper: trae payloads de insights para el rango indicado
        def fetch_rows(p):
            cur.execute(f"""
                SELECT m.id, m.key_topics, LOWER(COALESCE(m.response,'')) AS resp,
                       LOWER(COALESCE(m.source_title,'')) AS title, i.payload
                FROM mentions m
                JOIN queries q ON m.query_id = q.id
                LEFT JOIN insights i ON i.id = m.generated_insight_id
                WHERE {where_sql}
            """, tuple(p))
            return cur.fetchall()

        # Traer filas del período actual
        rows = fetch_rows(params)
        total_responses = len(rows)

        def compute_counts(rows_in):
            from collections import defaultdict
            counts_local = defaultdict(int)
            for _id, key_topics, resp, title, payload in rows_in:
                detected = _detect_brands(key_topics, resp, title, payload)
                seen = set()
                for brand_name in detected:
                    if brand_name in seen:
                        continue
                    seen.add(brand_name)
                    counts_local[brand_name] += 1
            return counts_local

        counts = compute_counts(rows)

        colors = ["bg-blue-500", "bg-red-500", "bg-blue-600", "bg-yellow-500", "bg-gray-800"]
        # Calcular comparativa con el periodo anterior
        prev_params = list(params)
        curr_start = params[0]
        curr_end = params[1]
        if isinstance(curr_start, datetime) and isinstance(curr_end, datetime):
            delta_range = curr_end - curr_start
            prev_start = curr_start - delta_range
            prev_end = curr_start
        else:
            # Fallback conservador: un rango del mismo tamaño hacia atrás
            from datetime import timedelta
            prev_end = params[0]
            prev_start = prev_end - (params[1] - params[0] if isinstance(params[1], type(params[0])) else timedelta(days=30))
        prev_params[0] = prev_start
        prev_params[1] = prev_end

        rows_prev = fetch_rows(prev_params)
        total_prev = len(rows_prev)
        counts_prev = compute_counts(rows_prev)

        ranking = []
        denom = max(total_responses, 1)
        prev_denom = max(total_prev, 1)
        pairs = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (brand_name, count) in enumerate(pairs):
            pct_curr = (count / denom) * 100.0
            prev_pct = (counts_prev.get(brand_name, 0) / prev_denom) * 100.0 if prev_denom else 0.0
            diff = round(pct_curr - prev_pct, 1)
            ranking.append({
                "rank": i + 1,
                "name": brand_name,
                "score": f"{pct_curr:.1f}%",
                "change": f"{diff:+.1f}%",
                "positive": diff >= 0,
                "color": colors[i % len(colors)],
                "selected": False,
            })

        cur.close(); conn.close()
        return jsonify({"ranking": ranking, "total_responses": total_responses})
    except Exception as e:
        print(f"Error en get_visibility_ranking: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/industry/ranking', methods=['GET'])
def get_industry_ranking():
    """Share of Voice general y por tema basado en detección por key_topics.
    - overall_ranking: porcentaje por marca sobre el total de ocurrencias de marcas detectadas
    - by_topic: mismo cálculo pero segmentado por q.topic
    Filtros: mismos de parse_filters (fecha, modelo, source, topic)
    """
    try:
        filters = parse_filters(request)
        conn = get_db_connection(); cur = conn.cursor()

        where = ["m.created_at >= %s AND m.created_at < %s"]
        params = [filters['start_date'], filters['end_date']]
        if filters.get('model') and filters['model'] != 'all':
            where.append("m.engine = %s"); params.append(filters['model'])
        if filters.get('source') and filters['source'] != 'all':
            where.append("m.source = %s"); params.append(filters['source'])
        # Filtro directo por topic en menciones
        if filters.get('topic') and filters['topic'] != 'all':
            where.append("q.topic = %s"); params.append(filters['topic'])
        where_sql = " AND ".join(where)

        # Obtener menciones con su topic y los campos necesarios para detección por sinónimos
        cur.execute(f"""
            SELECT q.topic, m.key_topics,
                   LOWER(COALESCE(m.response,'')) AS resp,
                   LOWER(COALESCE(m.source_title,'')) AS title,
                   i.payload
            FROM mentions m
            JOIN queries q ON m.query_id = q.id
            LEFT JOIN insights i ON i.id = m.generated_insight_id
            WHERE {where_sql}
        """, tuple(params))
        rows = cur.fetchall()

        # Detectar marcas por sinónimos + payload.brands. Contamos presencia por mención
        from collections import defaultdict
        overall_counts = defaultdict(int)
        topic_counts = defaultdict(lambda: defaultdict(int))
        for topic, key_topics, resp, title, payload in rows:
            detected = _detect_brands(key_topics, resp, title, payload)
            if not detected:
                continue
            seen = set()
            for canon in detected:
                if canon in seen:
                    continue
                seen.add(canon)
                overall_counts[canon] += 1
                topic_counts[topic or 'Uncategorized'][canon] += 1

        colors = ["bg-blue-500", "bg-red-500", "bg-blue-600", "bg-yellow-500", "bg-gray-800"]
        # overall ranking
        total_mentions = sum(overall_counts.values()) or 1
        overall_sorted = sorted(overall_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        overall = []
        for i, (norm_name, c) in enumerate(overall_sorted):
            overall.append({
                "rank": i + 1,
                "name": norm_name,
                "score": f"{(c/total_mentions)*100:.1f}%",
                "change": "+0.0%",
                "positive": True,
                "color": colors[i % len(colors)],
                "selected": norm_name == "The Core School"
            })

        # by_topic ranking
        by_topic = {}
        for t, counts in topic_counts.items():
            tot = sum(counts.values()) or 1
            pairs = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
            rk = []
            for i, (norm_name, c) in enumerate(pairs):
                rk.append({
                    "rank": i + 1,
                    "name": norm_name,
                    "score": f"{(c/tot)*100:.1f}%",
                    "change": "+0.0%",
                    "positive": True,
                    "color": colors[i % len(colors)],
                    "selected": norm_name == "The Core School"
                })
            by_topic[t] = rk

        cur.close(); conn.close()
        return jsonify({"overall_ranking": overall, "by_topic": by_topic})
    except Exception as e:
        print(f"Error en get_industry_ranking: {e}")
        return jsonify({"error": str(e)}), 500

# ... (El resto de tus endpoints se quedan igual)

@app.route('/api/insights/<int:insight_id>', methods=['GET'])
def get_insight_by_id(insight_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, query_id, payload, created_at FROM insights WHERE id = %s", (insight_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        
        if not row:
            return jsonify({"error": "Insight not found"}), 404

        insight = { "id": row[0], "query_id": row[1], "payload": row[2], "created_at": row[3].isoformat() if row[3] else None }
        return jsonify(insight)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
@app.route('/api/insights', methods=['GET'])
def get_insights():
    try:
        filters = parse_filters(request)
        # Parámetros opcionales
        limit = int(request.args.get('limit', 1000))
        brand = filters.get('brand')

        # Resolver project_id a partir de la marca si no se pasa explícito
        project_id = request.args.get('project_id', None)
        if project_id is None:
            try:
                conn = get_db_connection(); cur = conn.cursor()
                cur.execute("SELECT id FROM queries WHERE COALESCE(brand, topic) = %s ORDER BY id ASC LIMIT 1", (brand,))
                row = cur.fetchone()
                cur.close(); conn.close()
                project_id = int(row[0]) if row else 1
            except Exception:
                project_id = 1
        else:
            project_id = int(project_id)

        from src.reports import aggregator as agg
        session = agg.get_session()
        try:
            clusters = agg.aggregate_clusters_for_report(
                session,
                project_id,
                start_date=filters.get('start_date').strftime('%Y-%m-%d') if filters.get('start_date') else None,
                end_date=(filters.get('end_date') - timedelta(days=1)).strftime('%Y-%m-%d') if filters.get('end_date') else None,
                max_rows=max(100, min(5000, limit)),
            )
        finally:
            session.close()

        # Enriquecer con un nombre de tema simple (heurística: primeras palabras del primer ejemplo)
        def _label_topic(cluster_obj: dict) -> str:
            try:
                examples = cluster_obj.get('example_mentions') or []
                if not examples:
                    return "(tema)"
                summary = (examples[0].get('summary') or '').strip()
                if not summary:
                    return "(tema)"
                words = summary.split()
                return " ".join(words[:8])
            except Exception:
                return "(tema)"

        clusters_out = []
        for c in clusters or []:
            clusters_out.append({
                "cluster_id": c.get("cluster_id"),
                "topic_name": _label_topic(c),
                "volume": c.get("count", 0),
                "avg_sentiment": c.get("avg_sentiment", 0.0),
                "top_sources": c.get("top_sources", []),
                "examples": c.get("example_mentions", []),
            })

        # Mezcla SIEMPRE insights clásicos para garantizar volumen suficiente en UI
        classic_insights: list[dict] = []
        try:
            from src.reports import aggregator as agg2
            sess2 = agg2.get_session()
            try:
                # Si hay muy pocos clusters, intenta sin project_id para traer catálogo general
                summary = agg2.get_agent_insights_data(sess2, project_id if len(clusters_out) >= 10 else None, limit=800)
            finally:
                sess2.close()

            buckets = (summary or {}).get("buckets", {})
            # Convertir buckets en filas compatibles
            def rows_from_list(key: str, arr: list[str | dict]) -> list[dict]:
                out: list[dict] = []
                for i, it in enumerate(arr or []):
                    if isinstance(it, str):
                        payload = {key: [it]}
                    elif isinstance(it, dict):
                        payload = {key: [it.get("text") or it.get("title") or it.get("opportunity") or it.get("risk") or it.get("trend") or ""]}
                    else:
                        continue
                    out.append({
                        "id": i + 1,
                        "query_id": project_id,
                        "payload": payload,
                        "created_at": None,
                    })
                return out

            classic_insights.extend(rows_from_list("opportunities", buckets.get("opportunities", [])))
            classic_insights.extend(rows_from_list("risks", buckets.get("risks", [])))
            classic_insights.extend(rows_from_list("trends", buckets.get("trends", [])))
            # quotes y ctas
            if isinstance(buckets.get("ctas"), list) and buckets["ctas"]:
                classic_insights.append({
                    "id": 99901,
                    "query_id": project_id,
                    "payload": {"calls_to_action": [x if isinstance(x, str) else str(x) for x in buckets["ctas"]]},
                    "created_at": None,
                })
            if isinstance(buckets.get("quotes"), list) and buckets["quotes"]:
                classic_insights.append({
                    "id": 99902,
                    "query_id": project_id,
                    "payload": {"quotes": [x if isinstance(x, str) else str(x) for x in buckets["quotes"]]},
                    "created_at": None,
                })
        except Exception:
            classic_insights = []

        # NUEVO: stream plano de payloads sin Top-N por bucket para que el frontend
        # pueda construir todas las tarjetas si lo desea
        try:
            from src.reports import aggregator as agg3
            sess3 = agg3.get_session()
            try:
                raw_rows = agg3.get_agent_insight_payloads(
                    sess3,
                    project_id,
                    start_date=filters.get('start_date').strftime('%Y-%m-%d') if filters.get('start_date') else None,
                    end_date=(filters.get('end_date') - timedelta(days=1)).strftime('%Y-%m-%d') if filters.get('end_date') else None,
                    limit=max(1000, limit)
                )
            finally:
                sess3.close()
        except Exception:
            raw_rows = []

        return jsonify({
            "project_id": project_id,
            "period": {
                "start": filters.get('start_date').isoformat() if filters.get('start_date') else None,
                "end": filters.get('end_date').isoformat() if filters.get('end_date') else None,
            },
            "clusters": clusters_out,
            "insights": classic_insights,
            "insights_raw": raw_rows,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/sentiment', methods=['GET'])
def get_sentiment():
    try:
        filters = parse_filters(request)
        granularity = request.args.get('granularity', 'day').lower()
        conn = get_db_connection(); cur = conn.cursor()

        where_clauses = ["m.created_at >= %s AND m.created_at < %s"]
        params = [filters['start_date'], filters['end_date']]
        if filters.get('model') and filters['model'] != 'all':
            where_clauses.append("m.engine = %s"); params.append(filters['model'])
        if filters.get('source') and filters['source'] != 'all':
            where_clauses.append("m.source = %s"); params.append(filters['source'])
        if filters.get('topic') and filters['topic'] != 'all':
            where_clauses.append("q.topic = %s"); params.append(filters['topic'])
        if filters.get('brand'):
            where_clauses.append("COALESCE(q.brand, '') = %s"); params.append(filters['brand'])

        where_sql = " AND ".join(where_clauses)
        join_sql = "JOIN queries q ON m.query_id = q.id"

        # Detección robusta de marca (sinónimos + texto + título + payload.brands)
        brand = request.args.get('brand') or os.getenv('DEFAULT_BRAND', 'The Core School')
        synonyms = [s.lower() for s in BRAND_SYNONYMS.get(brand, [brand.lower()])]
        like_patterns = [f"%{s}%" for s in synonyms]

        timeseries = []
        total_neg = 0
        total_neu = 0
        total_pos = 0

        if granularity == 'hour':
            # Agregación por ejecución del poll (poll_id) usando hora de inicio del poll
            query = f"""
                WITH rows AS (
                    SELECT 
                        m.poll_id,
                        m.created_at,
                        COALESCE(m.sentiment, 0) AS sent,
                        (
                            EXISTS (
                                SELECT 1 FROM jsonb_array_elements_text(COALESCE(to_jsonb(m.key_topics),'[]'::jsonb)) kt
                                WHERE LOWER(TRIM(kt)) = ANY(%s)
                            )
                            OR LOWER(COALESCE(m.response,'')) LIKE ANY(%s)
                            OR LOWER(COALESCE(m.source_title,'')) LIKE ANY(%s)
                            OR EXISTS (
                                SELECT 1 FROM jsonb_array_elements(COALESCE(i.payload->'brands','[]'::jsonb)) b
                                WHERE LOWER(TRIM(CASE WHEN jsonb_typeof(b)='object' THEN COALESCE(b->>'name','') ELSE TRIM(BOTH '"' FROM b::text) END)) = ANY(%s)
                            )
                        ) AS is_brand
                    FROM mentions m
                    {join_sql}
                    LEFT JOIN insights i ON i.id = m.generated_insight_id
                    WHERE {where_sql} AND m.poll_id IS NOT NULL
                )
                SELECT 
                    MIN(created_at) AS poll_start_time,
                    SUM(CASE WHEN is_brand THEN 1 ELSE 0 END) AS total_brand,
                    SUM(CASE WHEN is_brand AND sent > 0.3 THEN 1 ELSE 0 END) AS pos_brand,
                    SUM(CASE WHEN is_brand AND sent BETWEEN -0.3 AND 0.3 THEN 1 ELSE 0 END) AS neu_brand,
                    SUM(CASE WHEN is_brand AND sent < -0.3 THEN 1 ELSE 0 END) AS neg_brand
                FROM rows
                GROUP BY poll_id
                ORDER BY MIN(created_at)
            """
            cur.execute(query, tuple([synonyms, like_patterns, like_patterns, synonyms] + params))
            rows = cur.fetchall()
            tz_madrid = pytz.timezone('Europe/Madrid')
            for poll_start_time, total_b, pos_b, neu_b, neg_b in rows:
                total_b = int(total_b or 0); pos_b = int(pos_b or 0); neu_b = int(neu_b or 0); neg_b = int(neg_b or 0)
                total_neg += neg_b; total_neu += neu_b; total_pos += pos_b
                denom = max(total_b, 1)
                pct = round((pos_b / denom) * 100.0, 1)
                try:
                    dt_madrid = (poll_start_time.replace(tzinfo=pytz.UTC) if poll_start_time.tzinfo is None else poll_start_time.astimezone(pytz.UTC)).astimezone(tz_madrid)
                except Exception:
                    dt_madrid = poll_start_time
                timeseries.append({
                    "date": dt_madrid.isoformat(),
                    "ts": int(dt_madrid.timestamp() * 1000),
                    "value": pct,
                })
        else:
            # Serie diaria considerando solo menciones de la marca
            query = f"""
                WITH rows AS (
                    SELECT 
                        m.created_at,
                        COALESCE(m.sentiment, 0) AS sent,
                        (
                            EXISTS (
                                SELECT 1 FROM jsonb_array_elements_text(COALESCE(to_jsonb(m.key_topics),'[]'::jsonb)) kt
                                WHERE LOWER(TRIM(kt)) = ANY(%s)
                            )
                            OR LOWER(COALESCE(m.response,'')) LIKE ANY(%s)
                            OR LOWER(COALESCE(m.source_title,'')) LIKE ANY(%s)
                            OR EXISTS (
                                SELECT 1 FROM jsonb_array_elements(COALESCE(i.payload->'brands','[]'::jsonb)) b
                                WHERE LOWER(TRIM(CASE WHEN jsonb_typeof(b)='object' THEN COALESCE(b->>'name','') ELSE TRIM(BOTH '"' FROM b::text) END)) = ANY(%s)
                            )
                        ) AS is_brand
                    FROM mentions m
                    {join_sql}
                    LEFT JOIN insights i ON i.id = m.generated_insight_id
                    WHERE {where_sql}
                )
                SELECT 
                    DATE(created_at) AS d,
                    SUM(CASE WHEN is_brand AND sent > 0.3 THEN 1 ELSE 0 END) AS pos,
                    SUM(CASE WHEN is_brand AND sent BETWEEN -0.3 AND 0.3 THEN 1 ELSE 0 END) AS neu,
                    SUM(CASE WHEN is_brand AND sent < -0.3 THEN 1 ELSE 0 END) AS neg
                FROM rows
                GROUP BY DATE(created_at)
                ORDER BY DATE(created_at)
            """
            cur.execute(query, tuple([synonyms, like_patterns, like_patterns, synonyms] + params))
            rows = cur.fetchall()
            from datetime import datetime as _dt
            tz_madrid = pytz.timezone('Europe/Madrid')
            for d_bucket, pos, neu, neg in rows:
                pos = int(pos or 0); neu = int(neu or 0); neg = int(neg or 0)
                total = pos + neu + neg
                total_neg += neg; total_neu += neu; total_pos += pos
                pct_pos = (pos / max(total, 1)) * 100.0
                ts_dt = _dt.combine(d_bucket, _dt.min.time()).replace(tzinfo=pytz.UTC).astimezone(tz_madrid)
                timeseries.append({"date": ts_dt.isoformat(), "ts": int(ts_dt.timestamp() * 1000), "value": round(pct_pos, 1)})

        distribution = {"negative": int(total_neg), "neutral": int(total_neu), "positive": int(total_pos)}

        # Top negativas/positivas de la marca
        brand_filter_sql = f"""
            (
                EXISTS (
                    SELECT 1 FROM jsonb_array_elements_text(COALESCE(to_jsonb(m.key_topics),'[]'::jsonb)) kt
                    WHERE LOWER(TRIM(kt)) = ANY(%s)
                )
                OR LOWER(COALESCE(m.response,'')) LIKE ANY(%s)
                OR LOWER(COALESCE(m.source_title,'')) LIKE ANY(%s)
                OR EXISTS (
                    SELECT 1 FROM jsonb_array_elements(COALESCE(i.payload->'brands','[]'::jsonb)) b
                    WHERE LOWER(TRIM(CASE WHEN jsonb_typeof(b)='object' THEN COALESCE(b->>'name','') ELSE TRIM(BOTH '"' FROM b::text) END)) = ANY(%s)
                )
            )
        """

        neg_sql = f"""
            SELECT m.id, m.summary, m.key_topics, m.source_title, m.source_url, m.sentiment, m.created_at
            FROM mentions m
            {join_sql}
            LEFT JOIN insights i ON i.id = m.generated_insight_id
            WHERE {where_sql} AND {brand_filter_sql} AND COALESCE(m.sentiment, 0) < -0.3 AND COALESCE(m.confidence_score, 0) >= 0.6
            ORDER BY m.sentiment ASC NULLS FIRST, m.created_at DESC
            LIMIT 20
        """
        cur.execute(neg_sql, tuple(params + [synonyms, like_patterns, like_patterns, synonyms]))
        neg_rows = cur.fetchall()
        negatives = [
            {
                "id": r[0],
                "summary": r[1],
                "key_topics": r[2] or [],
                "source_title": r[3],
                "source_url": r[4],
                "sentiment": float(r[5] or 0.0),
                "created_at": r[6].isoformat() if r[6] else None,
            }
            for r in neg_rows
        ]

        pos_sql = f"""
            SELECT m.id, m.summary, m.key_topics, m.source_title, m.source_url, m.sentiment, m.created_at
            FROM mentions m
            {join_sql}
            LEFT JOIN insights i ON i.id = m.generated_insight_id
            WHERE {where_sql} AND {brand_filter_sql} AND COALESCE(m.sentiment, 0) > 0.3 AND COALESCE(m.confidence_score, 0) >= 0.6
            ORDER BY m.sentiment DESC NULLS LAST, m.created_at DESC
            LIMIT 20
        """
        cur.execute(pos_sql, tuple(params + [synonyms, like_patterns, like_patterns, synonyms]))
        pos_rows = cur.fetchall()
        positives = [
            {
                "id": r[0],
                "summary": r[1],
                "key_topics": r[2] or [],
                "source_title": r[3],
                "source_url": r[4],
                "sentiment": float(r[5] or 0.0),
                "created_at": r[6].isoformat() if r[6] else None,
            }
            for r in pos_rows
        ]

        cur.close(); conn.close()

        return jsonify({
            "timeseries": timeseries,
            "distribution": distribution,
            "negatives": negatives,
            "positives": positives,
            "granularity": granularity,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/topics-cloud', methods=['GET'])
def get_topics_cloud():
    try:
        filters = parse_filters(request)
        conn = get_db_connection(); cur = conn.cursor()

        where_clauses = ["m.created_at >= %s AND m.created_at < %s"]
        params = [filters['start_date'], filters['end_date']]
        join_sql = "JOIN queries q ON m.query_id = q.id"
        if filters.get('model') and filters['model'] != 'all':
            where_clauses.append("m.engine = %s"); params.append(filters['model'])
        if filters.get('source') and filters['source'] != 'all':
            where_clauses.append("m.source = %s"); params.append(filters['source'])
        if filters.get('topic') and filters['topic'] != 'all':
            where_clauses.append("q.topic = %s"); params.append(filters['topic'])
        where_sql = " AND ".join(where_clauses)

        # Desagregar key_topics tolerando TEXT[] o JSONB: convertir siempre con to_jsonb
        cur.execute(f"""
            SELECT LOWER(TRIM(topic)) AS t, COUNT(*) AS c, AVG(sent) AS avg_sentiment
            FROM (
                SELECT COALESCE(m.sentiment, 0) AS sent,
                       jsonb_array_elements_text(COALESCE(to_jsonb(m.key_topics), '[]'::jsonb)) AS topic
                FROM mentions m
                {join_sql}
                WHERE {where_sql}
            ) s
            WHERE topic IS NOT NULL AND topic <> ''
            GROUP BY LOWER(TRIM(topic))
            ORDER BY COUNT(*) DESC
            LIMIT 100
        """, tuple(params))
        rows = cur.fetchall()
        topics = [{"topic": r[0], "count": int(r[1] or 0), "avg_sentiment": float(r[2] or 0.0)} for r in rows]

        # Agrupación opcional (puede ser costosa). Si ?groups=0, saltar IA.
        groups = []
        groups_flag = (request.args.get('groups', '1') or '1').lower()
        if groups_flag in ('1', 'true', 'yes', 'y'):
            # Clave de caché por ventana temporal + filtros principales
            try:
                cache_key = "|".join([
                    str(filters.get('brand') or ''),
                    str(filters.get('model') or ''),
                    str(filters.get('source') or ''),
                    str(filters.get('topic') or ''),
                    (filters.get('start_date') or datetime.min).isoformat(),
                    (filters.get('end_date') or datetime.max).isoformat(),
                ])
            except Exception:
                cache_key = None

            if cache_key:
                cached = _groups_cache_get(cache_key)
                if cached is not None:
                    groups = cached
                else:
                    try:
                        groups = group_topics_with_ai(topics) or []
                        _groups_cache_set(cache_key, groups)
                    except Exception:
                        groups = []
            else:
                try:
                    groups = group_topics_with_ai(topics) or []
                except Exception:
                    groups = []

        cur.close(); conn.close()
        return jsonify({"topics": topics, "groups": groups})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/dashboard-kpis', methods=['GET'])
def get_dashboard_kpis():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        day_ago = datetime.now() - timedelta(hours=24)
        cur.execute("SELECT COUNT(*) FROM mentions WHERE created_at >= %s", [day_ago])
        mentions_24h = cur.fetchone()[0] or 0
        cur.execute("SELECT COUNT(*) FROM queries WHERE enabled = true")
        active_queries = cur.fetchone()[0] or 0
        cur.execute("SELECT COUNT(*) FROM mentions WHERE created_at >= %s AND sentiment < -0.5", [day_ago])
        alerts_24h = cur.fetchone()[0] or 0
        cur.execute("SELECT COUNT(CASE WHEN sentiment > 0.2 THEN 1 END), COUNT(*) FROM mentions WHERE created_at >= %s", [day_ago])
        pos_total = cur.fetchone()
        pos = pos_total[0] or 0
        total = pos_total[1] or 0
        positive_sentiment = (pos / max(total, 1)) * 100 if total else 0
        cur.close()
        conn.close()
        
        return jsonify({
            "mentions_24h": mentions_24h, "mentions_24h_delta": 0,
            "positive_sentiment": round(positive_sentiment, 1), "positive_sentiment_delta": 0,
            "alerts_triggered": alerts_24h, "alerts_delta": 0,
            "active_queries": active_queries, "queries_delta": 0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/topics', methods=['GET'])
def get_topics():
    """Lista de temas (topics) de las queries configuradas."""
    try:
        # Fuente de verdad dinámica: DISTINCT COALESCE(category, topic) desde queries
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT TRIM(COALESCE(category, topic)) AS topic
            FROM queries
            WHERE enabled = TRUE
              AND COALESCE(category, topic) IS NOT NULL
              AND TRIM(COALESCE(category, topic)) <> ''
            ORDER BY topic
            """
        )
        topics = [row[0] for row in cur.fetchall()]
        cur.close(); conn.close()

        return jsonify({"topics": topics})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Lista de modelos (engine) usados para generar menciones."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT engine
            FROM mentions
            WHERE engine IS NOT NULL AND engine <> ''
            ORDER BY engine
        """)
        models = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/sources', methods=['GET'])
def get_sources():
    """Lista de fuentes (source) detectadas en menciones."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT source
            FROM mentions
            WHERE source IS NOT NULL AND source <> ''
            ORDER BY source
        """)
        sources = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        return jsonify({"sources": sources})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/prompts', methods=['GET'])
def get_prompts_grouped():
    """Devuelve prompts agrupados por topic, con métricas básicas."""
    try:
        filters = parse_filters(request)
        conn = get_db_connection()
        cur = conn.cursor()

        # Fuente de verdad de topics: lo que está en la tabla queries

        # Obtenemos los prompts (queries) activos y sus métricas
        # Aplicamos filtro por modelo en el LEFT JOIN para no convertirlo en INNER JOIN
        engine_on = " AND m.engine = %s" if (filters.get('model') and filters['model'] != 'all') else ""
        params_exec = [filters['start_date'], filters['end_date']]
        if engine_on:
            params_exec.append(filters['model'])
        topic_filter_sql = ""
        if filters.get('topic') and filters['topic'] != 'all':
            topic_filter_sql = " AND COALESCE(q.category, q.topic) = %s"
            params_exec.append(filters['topic'])
        cur.execute(
            f"""
            SELECT q.id, q.query, COALESCE(q.category, q.topic) AS topic,
                   COUNT(m.id) AS executions,
                   COUNT(DISTINCT DATE(m.created_at)) AS active_days
            FROM queries q
            LEFT JOIN mentions m ON m.query_id = q.id
                 AND m.created_at >= %s AND m.created_at < %s{engine_on}
            WHERE q.enabled = TRUE{topic_filter_sql}
            GROUP BY q.id, q.query, COALESCE(q.category, q.topic)
            """,
            tuple(params_exec)
        )

        rows = cur.fetchall()

        # Cargar menciones por prompt para calcular métricas individuales de marca
        where_engine = " AND m.engine = %s" if (filters.get('model') and filters['model'] != 'all') else ""
        params_mentions = [filters['start_date'], filters['end_date']]
        if where_engine:
            params_mentions.append(filters['model'])
        topic_filter_sql2 = ""
        if filters.get('topic') and filters['topic'] != 'all':
            topic_filter_sql2 = " AND COALESCE(q.category, q.topic) = %s"
            params_mentions.append(filters['topic'])
        cur.execute(
            f"""
            SELECT m.query_id, m.key_topics,
                   LOWER(COALESCE(m.response,'')) AS resp,
                   LOWER(COALESCE(m.source_title,'')) AS title,
                   i.payload
            FROM mentions m
            JOIN queries q ON m.query_id = q.id
            LEFT JOIN insights i ON i.id = m.generated_insight_id
            WHERE m.created_at >= %s AND m.created_at < %s{where_engine}{topic_filter_sql2}
              AND q.enabled = TRUE
            """,
            tuple(params_mentions)
        )
        mention_rows = cur.fetchall()
        cur.close(); conn.close()

        # Convertimos a estructura agrupada por topic usando la nueva función de IA
        from collections import defaultdict
        # Agrupar menciones por query_id
        mentions_by_qid = defaultdict(list)

        # Diccionario de marcas y sinónimos (alineado con otros endpoints)
        BRAND_SYNONYMS = {
            "The Core School": ["the core", "the core school", "thecore"],
            "U-TAD": ["u-tad", "utad"],
            "ECAM": ["ecam"],
            "TAI": ["tai"],
            "CES": ["ces"],
            "CEV": ["cev"],
            "FX Barcelona Film School": ["fx barcelona", "fx barcelona film school", "fx animation"],
            "Septima Ars": ["septima ars", "séptima ars"],
        }

        def _normalize_list(xs):
            out = []
            if not xs:
                return out
            try:
                for t in xs:
                    s = str(t or '').lower().strip()
                    if s:
                        out.append(s)
            except Exception:
                pass
            return out

        def _extract_brands_from_payload(payload):
            names = []
            try:
                if isinstance(payload, dict):
                    brands = payload.get('brands') or []
                    if isinstance(brands, list):
                        for item in brands:
                            if isinstance(item, dict):
                                name = (item.get('name') or '').strip()
                                if name:
                                    names.append(name)
                            elif isinstance(item, str):
                                name = item.strip()
                                if name:
                                    names.append(name)
            except Exception:
                return []
            return names

        for qid, key_topics, resp, title, payload in mention_rows:
            mentions_by_qid[qid].append({
                "topics": key_topics,
                "resp": resp,
                "title": title,
                "payload": payload if isinstance(payload, dict) else {},
            })
        topics_map = defaultdict(list)
        total_days = max((filters['end_date'] - filters['start_date']).days, 1)

        prompt_items = []
        for pid, query, qtopic, executions, active_days in rows:
            # Usar el topic almacenado en queries directamente (fuente de verdad)
            category = (qtopic or 'Uncategorized')

            visibility_score = round((active_days / total_days) * 100, 1)

            # Métricas individuales por prompt
            brand = request.args.get('brand') or os.getenv('DEFAULT_BRAND', 'The Core School')
            brand_present_responses = 0
            total_responses_prompt = 0
            brand_presence_count = 0  # veces que aparece la marca
            all_brands_presence_total = 0  # suma de presencias de todas las marcas

            for m in mentions_by_qid.get(pid, []):
                total_responses_prompt += 1
                present = _extract_brands_from_payload(m.get('payload'))
                # deduplicar por mención
                present_norm = []
                seen = set()
                for b in present:
                    n = b.lower().strip()
                    if n and n not in seen:
                        seen.add(n); present_norm.append(b)
                if present_norm:
                    all_brands_presence_total += len(present_norm)
                if any((brand or '').lower().strip() == (b or '').lower().strip() for b in present_norm):
                    brand_present_responses += 1
                    brand_presence_count += 1

            visibility_score_individual = round((brand_present_responses / max(total_responses_prompt, 1)) * 100.0, 1)
            share_of_voice_individual = round((brand_presence_count / max(all_brands_presence_total, 1)) * 100.0, 1)
            prompt_items.append({
                "category": category,
                "id": pid,
                "query": query,
                "visibility_score": visibility_score,
                "visibility_score_individual": visibility_score_individual,
                "share_of_voice_individual": share_of_voice_individual,
                "executions": int(executions or 0),
            })

        group_exec_totals = defaultdict(int)
        for it in prompt_items:
            group_exec_totals[it["category"]] += it["executions"]

        for it in prompt_items:
            topic_name = it["category"]
            total_mentions_topic = max(group_exec_totals.get(topic_name, 0), 1)
            share_of_voice = round((it["executions"] / total_mentions_topic) * 100, 1)
            topics_map[topic_name].append({
                "id": it["id"],
                "query": it["query"],
                "visibility_score": it["visibility_score"],
                "share_of_voice": share_of_voice,
                "executions": it["executions"],
                # --- métricas individuales por prompt ---
                "visibility_score_individual": it["visibility_score_individual"],
                "share_of_voice_individual": it["share_of_voice_individual"],
            })

        topics = []
        for topic_name, prompts in topics_map.items():
            prompts_sorted = sorted(prompts, key=lambda p: p["visibility_score"], reverse=True)
            for idx, p in enumerate(prompts_sorted, 1):
                p["rank"] = idx
            topics.append({
                "topic": topic_name,
                "prompts": prompts_sorted,
                "topic_total_mentions": int(group_exec_totals.get(topic_name, 0)),
            })

        topics.sort(key=lambda t: t["topic_total_mentions"], reverse=True)

        return jsonify({"topics": topics})
    except Exception as e:
        print(f"Error en get_prompts_grouped: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/prompts', methods=['POST'])
def create_prompt():
    """Crea un nuevo prompt (fila en `queries`).
    Espera JSON: { query: str, topic?: str, brand?: str, language?: str, enabled?: bool }
    Devuelve: { id, query, topic, brand, language, enabled, created_at }
    """
    try:
        payload = request.get_json(silent=True) or {}
        query_text = (payload.get('query') or '').strip()
        if not query_text:
            return jsonify({"error": "'query' es requerido"}), 400

        topic = (payload.get('topic') or None)
        brand = (payload.get('brand') or None)
        language = (payload.get('language') or 'en')
        enabled = bool(payload.get('enabled', True))

        conn = get_db_connection(); cur = conn.cursor()

        # Comprobar si ya existe un prompt con el mismo texto
        cur.execute("SELECT id, query, brand, topic, language, enabled, created_at FROM queries WHERE query = %s", (query_text,))
        row = cur.fetchone()
        if row:
            cur.close(); conn.close()
            return jsonify({
                "id": row[0], "query": row[1], "brand": row[2], "topic": row[3],
                "language": row[4], "enabled": bool(row[5]),
                "created_at": row[6].isoformat() if row[6] else None
            }), 200

        # Determinar categoría para almacenar en topic canónico (IA + fallback)
        available_topics = get_category_catalog_for_brand(brand)
        canonical_topic = categorize_prompt_with_ai(query_text, available_topics)

        cur.execute(
            """
            INSERT INTO queries (query, brand, topic, language, enabled)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, query, brand, topic, language, enabled, created_at
            """,
            (query_text, brand, canonical_topic, language, enabled)
        )
        new_row = cur.fetchone()
        conn.commit()
        cur.close(); conn.close()

        return jsonify({
            "id": new_row[0], "query": new_row[1], "brand": new_row[2], "topic": new_row[3],
            "language": new_row[4], "enabled": bool(new_row[5]),
            "created_at": new_row[6].isoformat() if new_row[6] else None
        }), 201
    except Exception as e:
        print(f"Error en create_prompt: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/prompts/<int:query_id>', methods=['PATCH'])
def update_prompt(query_id: int):
    """Actualiza un prompt; re-categoriza automáticamente el topic si cambia query/topic."""
    try:
        payload = request.get_json(silent=True) or {}
        fields = []
        values = []

        # Preparar campos editables
        qtext = payload.get('query')
        tpc = payload.get('topic')
        brand = payload.get('brand')
        language = payload.get('language')
        enabled = payload.get('enabled')

        # Si viene query o topic, recalculamos categoría con IA
        if qtext is not None or tpc is not None:
            # traer valores actuales si faltan
            conn = get_db_connection(); cur = conn.cursor()
            cur.execute("SELECT query, topic FROM queries WHERE id = %s", (query_id,))
            row = cur.fetchone()
            if not row:
                cur.close(); conn.close()
                return jsonify({"error": "Prompt not found"}), 404
            current_query, current_topic = row
            cur.close(); conn.close()
            new_query = qtext if qtext is not None else current_query
            # determinar topics disponibles
            available_topics = get_category_catalog_for_brand(brand)
            cat = categorize_prompt_with_ai(new_query, available_topics)
            fields.append('topic = %s'); values.append(cat)
            if qtext is not None:
                fields.append('query = %s'); values.append(qtext)
        else:
            if qtext is not None:
                fields.append('query = %s'); values.append(qtext)
            if tpc is not None:
                fields.append('topic = %s'); values.append(tpc)

        if brand is not None:
            fields.append('brand = %s'); values.append(brand)
        if language is not None:
            fields.append('language = %s'); values.append(language)
        if enabled is not None:
            fields.append('enabled = %s'); values.append(bool(enabled))

        if not fields:
            return jsonify({"error": "No fields to update"}), 400

        sql = f"UPDATE queries SET {', '.join(fields)} WHERE id = %s RETURNING id, query, brand, topic, language, enabled, created_at"
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute(sql, tuple(values + [query_id]))
        row = cur.fetchone()
        conn.commit(); cur.close(); conn.close()
        if not row:
            return jsonify({"error": "Prompt not found"}), 404

        return jsonify({
            "id": row[0], "query": row[1], "brand": row[2], "topic": row[3],
            "language": row[4], "enabled": bool(row[5]),
            "created_at": row[6].isoformat() if row[6] else None
        })
    except Exception as e:
        print(f"Error en update_prompt: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/prompts/<int:query_id>', methods=['DELETE'])
def delete_prompt(query_id: int):
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("DELETE FROM queries WHERE id = %s", (query_id,))
        deleted = cur.rowcount
        conn.commit(); cur.close(); conn.close()
        if deleted == 0:
            return jsonify({"error": "Prompt not found"}), 404
        return jsonify({"message": f"Prompt {query_id} deleted"})
    except Exception as e:
        print(f"Error en delete_prompt: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/prompts/<int:query_id>', methods=['GET'])
def get_prompt_details(query_id: int):
    """
    Detalle de un prompt: devuelve métricas clave, datos para gráficos 
    (visibilidad y SOV a lo largo del tiempo) y sus ejecuciones.
    """
    try:
        filters = parse_filters(request)
        conn = get_db_connection()
        cur = conn.cursor()

        # 1. Obtener datos básicos del prompt
        cur.execute("SELECT query, topic, brand FROM queries WHERE id = %s", (query_id,))
        row = cur.fetchone()
        if not row:
            cur.close(); conn.close()
            return jsonify({"error": "Prompt not found"}), 404
        query_text, topic, brand = row
        brand_to_check = brand or os.getenv('DEFAULT_BRAND', 'The Core School')

        # 2. Construir filtros de la consulta
        where_clauses = ["m.query_id = %s", "m.created_at >= %s", "m.created_at < %s"]
        params = [query_id, filters['start_date'], filters['end_date']]
        if filters.get('model') and filters['model'] != 'all':
            where_clauses.append("m.engine = %s")
            params.append(filters['model'])
        where_sql = " AND ".join(where_clauses)

        # 3. Obtener todas las menciones para este prompt con los filtros
        cur.execute(
            f"""
            SELECT m.id, m.created_at, m.engine, m.response,
                   m.key_topics, LOWER(COALESCE(m.response,'')) AS resp,
                   LOWER(COALESCE(m.source_title,'')) AS title, i.payload
            FROM mentions m
            LEFT JOIN insights i ON i.id = m.generated_insight_id
            WHERE {where_sql}
            ORDER BY m.created_at DESC
            """,
            tuple(params)
        )
        mention_rows = cur.fetchall()

        # 4. Calcular métricas, tendencias y ejecuciones
        total_responses_prompt = len(mention_rows)
        executions = [{"id": r[0], "created_at": r[1].isoformat(), "engine": r[2], "response": r[3]} for r in mention_rows]

        from collections import defaultdict
        daily_brand_mentions = defaultdict(int)
        daily_total_mentions = defaultdict(int)
        daily_brand_presence = defaultdict(int)
        daily_all_brands_presence = defaultdict(int)
        # Contadores por hora para series de alta resolución
        hourly_total_mentions = defaultdict(int)
        hourly_brand_mentions = defaultdict(int)
        hourly_brand_presence = defaultdict(int)
        hourly_all_brands_presence = defaultdict(int)
        brand_distribution = defaultdict(int)
        display_by_norm = {}
        all_trends = set()

        def _extract_brands_from_payload(payload):
            names = []
            try:
                if isinstance(payload, dict):
                    brands = payload.get('brands') or []
                    if isinstance(brands, list):
                        for item in brands:
                            if isinstance(item, dict):
                                name = (item.get('name') or '').strip()
                                if name:
                                    names.append(name)
                            elif isinstance(item, str):
                                name = item.strip()
                                if name:
                                    names.append(name)
            except Exception:
                return []
            return names

        for mention in mention_rows:
            _, created_at, _, _, key_topics, resp, title, payload = mention
            mention_date = created_at.date()
            daily_total_mentions[mention_date] += 1
            hour_key = created_at.replace(minute=0, second=0, microsecond=0)
            hourly_total_mentions[hour_key] += 1

            present_brands = _extract_brands_from_payload(payload)
            # deduplicación por mención
            seen = set(); present_norm = []
            for b in present_brands:
                n = b.lower().strip()
                if n and n not in seen:
                    seen.add(n); present_norm.append(b)

            if any((brand_to_check or '').lower().strip() == (b or '').lower().strip() for b in present_norm):
                daily_brand_mentions[mention_date] += 1
                daily_brand_presence[mention_date] += 1
                hourly_brand_mentions[hour_key] += 1
                hourly_brand_presence[hour_key] += 1

            if present_norm:
                daily_all_brands_presence[mention_date] += len(present_norm)
                hourly_all_brands_presence[hour_key] += len(present_norm)
                for b in present_norm:
                    n = b.lower().strip()
                    display_by_norm.setdefault(n, b)
                    brand_distribution[n] += 1

            if payload and isinstance(payload, dict):
                trends = payload.get('trends', [])
                if isinstance(trends, list):
                    for trend in trends:
                        if isinstance(trend, str): all_trends.add(trend.strip())

        # Calcular métricas totales
        total_brand_mentions = sum(daily_brand_mentions.values())
        total_brand_presence = sum(daily_brand_presence.values())
        total_all_brands_presence = sum(daily_all_brands_presence.values())

        visibility_score = round((total_brand_mentions / max(total_responses_prompt, 1)) * 100.0, 1)
        share_of_voice = round((total_brand_presence / max(total_all_brands_presence, 1)) * 100.0, 1)

        # 5. Calcular datos para los gráficos (por hora para más precisión)
        granularity = request.args.get('granularity', 'hour').lower()
        timeseries = []
        sov_timeseries = []
        if granularity == 'hour':
            from datetime import datetime as _dt, timezone as _tz
            t = filters['start_date'].replace(minute=0, second=0, microsecond=0)
            now_aware = _dt.now(filters['end_date'].tzinfo or _tz.utc)
            end_cap = min(filters['end_date'], now_aware)
            end_t = end_cap.replace(minute=0, second=0, microsecond=0)
            last_vis = None
            last_sov = None
            while t <= end_t:
                total = hourly_total_mentions[t]
                brand_m = hourly_brand_mentions[t]
                vis = (brand_m / max(total, 1)) * 100 if total > 0 else (last_vis if last_vis is not None else 0.0)
                timeseries.append({"date": t.isoformat(), "ts": int(t.timestamp() * 1000), "value": round(vis, 1)})
                if total > 0:
                    last_vis = vis

                brand_p = hourly_brand_presence[t]
                all_p = hourly_all_brands_presence[t]
                sov = (brand_p / max(all_p, 1)) * 100 if all_p > 0 else (last_sov if last_sov is not None else 0.0)
                sov_timeseries.append({"date": t.isoformat(), "ts": int(t.timestamp() * 1000), "value": round(sov, 1)})
                if all_p > 0:
                    last_sov = sov
                t += timedelta(hours=1)
        else:
            day_iterator = filters['start_date'].date()
            from datetime import datetime as _dt
            while day_iterator < filters['end_date'].date():
                total_day = daily_total_mentions[day_iterator]
                brand_mentions_day = daily_brand_mentions[day_iterator]
                visibility_day = (brand_mentions_day / max(total_day, 1)) * 100
                ts_dt = _dt.combine(day_iterator, _dt.min.time())
                timeseries.append({"date": ts_dt.isoformat(), "ts": int(ts_dt.timestamp() * 1000), "value": round(visibility_day, 1)})

                brand_presence_day = daily_brand_presence[day_iterator]
                all_brands_day = daily_all_brands_presence[day_iterator]
                sov_day = (brand_presence_day / max(all_brands_day, 1)) * 100
                sov_timeseries.append({"date": ts_dt.isoformat(), "ts": int(ts_dt.timestamp() * 1000), "value": round(sov_day, 1)})

                day_iterator += timedelta(days=1)

        cur.close(); conn.close()

        # Limitar brand_distribution a top-10 con nombres display originales
        dist_pairs = sorted(brand_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
        dist_out = { display_by_norm.get(k, k): v for k, v in dist_pairs }

        return jsonify({
            "id": query_id,
            "query": query_text,
            "topic": topic,
            "visibility_score": visibility_score,
            "share_of_voice": share_of_voice,
            "total_executions": total_responses_prompt,
            "trends": sorted(list(all_trends)),
            "timeseries": timeseries,
            "sov_timeseries": sov_timeseries,
            "brand_distribution": dist_out,
            "executions": executions[:100],
            "granularity": granularity,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/categorize', methods=['POST'])
def categorize_endpoint():
    """Devuelve categoría detectada + confianza, alternativas y sugerencia."""
    try:
        payload = request.get_json(silent=True) or {}
        query_text = payload.get('query')

        if not query_text or len((query_text or '').strip()) < 10:
            return jsonify({"category": None, "confidence": 0, "suggestion": None})

        # Determinar catálogo de categorías "generales" para clasificar
        # 1) Si existe una taxonomía de la marca, usarla
        # 2) Si no, usar las categorías por defecto (_CATEGORY_KEYWORDS)
        try:
            ensure_taxonomy_table()
            brand = request.args.get('brand') or os.getenv('DEFAULT_BRAND', 'The Core School')
            conn = get_db_connection(); cur = conn.cursor()
            cur.execute("SELECT category FROM topic_taxonomy WHERE brand = %s", (brand,))
            rows = cur.fetchall(); cur.close(); conn.close()
            catalog = [r[0] for r in rows] if rows else list(_CATEGORY_KEYWORDS.keys())
        except Exception:
            catalog = list(_CATEGORY_KEYWORDS.keys())

        # Paso 1: intento con el clasificador LLM acotado al catálogo
        best_from_ai = categorize_prompt_with_ai(query_text, catalog)

        # Paso 2: si la IA no logra clasificar (Inclasificable), usar heurística detallada
        detailed = None
        if best_from_ai == "Inclasificable":
            try:
                detailed = categorize_prompt_detailed(None, query_text)
            except Exception:
                detailed = None

        # Resolver categoría final y metadatos de respuesta
        category = None
        confidence = 0.8
        alternatives = []
        suggestion = None
        suggestion_is_new = False
        closest_existing = None
        closest_score = 0.0
        is_close_match = False

        if detailed:
            category = detailed.get("category") or "Inclasificable"
            confidence = float(detailed.get("confidence") or 0.8)
            alternatives = detailed.get("alternatives") or []
            suggestion = detailed.get("suggestion") or category
            suggestion_is_new = bool(detailed.get("suggestion_is_new") or False)
            closest_existing = detailed.get("closest_existing")
            closest_score = float(detailed.get("closest_score") or 0.0)
            is_close_match = bool(detailed.get("is_close_match") or False)
        else:
            category = best_from_ai
            suggestion = best_from_ai

        # Si la categoría propuesta no está en el catálogo, mapear por similitud
        if category != "Inclasificable" and category not in (catalog or []):
            mapped, score = _best_topic_by_similarity(category, catalog)
            closest_existing = mapped
            closest_score = float(score)
            if score >= 0.7:
                category = mapped
                is_close_match = True
            else:
                suggestion_is_new = True

        # Confianza ajustada según si es nueva o mapeada
        if category == "Inclasificable":
            confidence = 0.5
        else:
            confidence = 0.9 if not suggestion_is_new else max(0.7, confidence)

        return jsonify({
            "category": category,
            "confidence": confidence,
            "alternatives": alternatives,
            "suggestion": suggestion or category,
            "suggestion_is_new": suggestion_is_new,
            "closest_existing": closest_existing,
            "closest_score": closest_score,
            "is_close_match": is_close_match,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/prompts/recategorize', methods=['POST'])
def recategorize_prompts():
    """Reclasifica todos los prompts existentes según la taxonomía general.
    Body JSON (opcional): { brand?: str, limit?: int, dry_run?: bool }
    Devuelve resumen y lista de cambios.
    """
    try:
        payload = request.get_json(silent=True) or {}
        brand = payload.get('brand') or request.args.get('brand') or os.getenv('DEFAULT_BRAND', 'The Core School')
        limit = int(payload.get('limit') or 0) or None
        dry_run = bool(payload.get('dry_run', False))

        # Catálogo base
        try:
            ensure_taxonomy_table()
            conn = get_db_connection(); cur = conn.cursor()
            cur.execute("SELECT category FROM topic_taxonomy WHERE brand = %s", (brand,))
            rows = cur.fetchall(); cur.close(); conn.close()
            catalog = [r[0] for r in rows] if rows else list(_CATEGORY_KEYWORDS.keys())
        except Exception:
            catalog = list(_CATEGORY_KEYWORDS.keys())

        # Cargar prompts a procesar
        conn = get_db_connection(); cur = conn.cursor()
        sql = "SELECT id, query, brand, topic FROM queries"
        params = []
        if brand:
            sql += " WHERE (brand = %s OR %s IS NULL)"
            params.extend([brand, brand])
        sql += " ORDER BY id ASC"
        if limit:
            sql += " LIMIT %s"; params.append(limit)
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()

        changes = []
        updated = 0
        unchanged = 0
        inclasificables = 0

        for qid, qtext, qbrand, old_topic in rows:
            try:
                best_from_ai = categorize_prompt_with_ai(qtext, catalog)
            except Exception:
                best_from_ai = "Inclasificable"

            final_cat = best_from_ai
            if best_from_ai == "Inclasificable":
                try:
                    detailed = categorize_prompt_detailed(None, qtext)
                    final_cat = detailed.get("category") or "Inclasificable"
                except Exception:
                    final_cat = "Inclasificable"

            if final_cat != "Inclasificable" and final_cat not in catalog:
                mapped, score = _best_topic_by_similarity(final_cat, catalog)
                if score >= 0.7:
                    final_cat = mapped
                else:
                    # mantener inclasificable para evitar ruido en catálogo
                    final_cat = "Inclasificable"

            if final_cat == (old_topic or ""):
                unchanged += 1
            else:
                if final_cat == "Inclasificable":
                    inclasificables += 1
                if not dry_run:
                    cur2 = conn.cursor()
                    cur2.execute("UPDATE queries SET topic = %s WHERE id = %s", (final_cat, qid))
                    conn.commit(); cur2.close()
                updated += 1
                changes.append({"id": qid, "old": old_topic, "new": final_cat})

        cur.close(); conn.close()

        return jsonify({
            "brand": brand,
            "catalog_size": len(catalog),
            "total": len(rows),
            "updated": updated,
            "unchanged": unchanged,
            "inclasificables": inclasificables,
            "dry_run": dry_run,
            "changes": changes,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/query-visibility/<brand>', methods=['GET'])
def get_query_visibility(brand):
    return jsonify({"brand": brand, "queries": []})

@app.route('/api/mentions/<int:mention_id>/archive', methods=['PATCH'])
def archive_mention(mention_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("UPDATE mentions SET status = 'archived' WHERE id = %s", (mention_id,))
        conn.commit()
        updated_rows = cur.rowcount
        cur.close()
        conn.close()
        
        if updated_rows == 0:
            return jsonify({"error": "Mention not found"}), 404
        
        return jsonify({"message": f"Mention {mention_id} archived successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/reports/generate', methods=['POST'])
def generate_report_endpoint():
    try:
        payload = request.get_json() or {}
        # Soporte multi-cliente/categorías
        filters = {
            'start_date': parse_date(payload.get('start_date')) if payload.get('start_date') else (datetime.utcnow() - timedelta(days=30)),
            'end_date': parse_date(payload.get('end_date')) if payload.get('end_date') else datetime.utcnow(),
            'model': payload.get('model') or 'all',
            'source': payload.get('source') or 'all',
            'topic': payload.get('topic') or 'all',
            'client_id': payload.get('client_id'),
            'brand_id': payload.get('brand_id'),
            'category': payload.get('category') or payload.get('prompt_category'),
        }
        test_mode = bool(payload.get('test_mode', False))
        if not filters:
            return jsonify({"error": "Filtros no proporcionados"}), 400

        # 1) Obtener todos los datos en una sola llamada
        from src.reports import aggregator as agg
        brand_name = payload.get('brand') or os.getenv('DEFAULT_BRAND', 'The Core School')
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT id FROM queries WHERE COALESCE(brand, topic) = %s ORDER BY id ASC LIMIT 1", (brand_name,))
        row = cur.fetchone(); cur.close(); conn.close()
        project_id = int(row[0]) if row else 1

        full_data = agg.get_full_report_data(
            project_id,
            start_date=filters.get('start_date').strftime('%Y-%m-%d'),
            end_date=filters.get('end_date').strftime('%Y-%m-%d'),
            max_rows=5000,
        )

        # 2) Generar PDF con cadena de agentes (incluye Parte 2: Plan de Acción)
        from src.reports.generator import generate_report
        pdf_bytes = generate_report(
            project_id,
            start_date=filters.get('start_date').strftime('%Y-%m-%d'),
            end_date=filters.get('end_date').strftime('%Y-%m-%d'),
        )

        return send_file(
            io.BytesIO(pdf_bytes),
            as_attachment=True,
            download_name=f"Informe_Ejecutivo_{datetime.now().strftime('%Y-%m-%d')}.pdf",
            mimetype='application/pdf'
        )
    except Exception as e:
        print(f"Error generando el informe: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "No se pudo generar el informe. Revisa los logs del backend."}), 500


@app.route('/api/reports/generate/skeleton', methods=['POST'])
def generate_report_skeleton_endpoint():
    try:
        payload = request.get_json() or {}
        company = payload.get('brand') or os.getenv('DEFAULT_BRAND', 'Empresa')
        from src.reports.pdf_writer import build_empty_structure_pdf
        pdf_bytes = build_empty_structure_pdf(company)
        return send_file(
            io.BytesIO(pdf_bytes),
            as_attachment=True,
            download_name=f"Informe_Estructura_{datetime.now().strftime('%Y-%m-%d')}.pdf",
            mimetype='application/pdf'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/reports/generate/v2', methods=['POST'])
def generate_report_endpoint_v2():
    try:
        payload = request.get_json() or {}
        project_id = int(payload.get('project_id') or 1)
        pdf_bytes = generate_report_v2(project_id)
        return send_file(
            io.BytesIO(pdf_bytes),
            as_attachment=True,
            download_name=f"Informe_Geocore_{datetime.now().strftime('%Y-%m-%d')}.pdf",
            mimetype='application/pdf'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True, use_reloader=False)