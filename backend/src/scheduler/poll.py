import os
import time
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Callable, Union, List, Tuple, Dict, Any

import psycopg2
from psycopg2.extras import Json

from src.engines.openai_engine import fetch_response, extract_insights, fetch_response_with_metadata, client as openai_client
from src.engines.perplexity import fetch_perplexity_response, fetch_perplexity_with_metadata
from src.engines.serp import get_search_results as fetch_serp_response
from src.engines.serp import get_search_results_structured
from src.utils.slack import send_slack_alert
from src.engines.sentiment_fixed import analyze_sentiment



logging.basicConfig(
    filename="logs/poll.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

SENTIMENT_THRESHOLD = -0.3

DB_CFG = dict(
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=int(os.getenv("POSTGRES_PORT", 5433)),
    database=os.getenv("POSTGRES_DB", "ai_visibility"),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD", "postgres"),
)

def _parse_int_env(name: str) -> int | None:
    value = os.getenv(name)
    try:
        if value is None:
            return None
        value = str(value).strip()
        return int(value) if value else None
    except Exception:
        return None

# Filtros opcionales para limitar el polling por mercado/cliente/marca
POLL_PROJECT_ID = _parse_int_env("POLL_PROJECT_ID")
POLL_CLIENT_ID = _parse_int_env("POLL_CLIENT_ID")
POLL_BRAND_ID = _parse_int_env("POLL_BRAND_ID")

def summarize_and_extract_topics(text: str) -> Tuple[str, List[str]]:
    prompt = f"""
Analiza el siguiente texto y devuelve un objeto JSON con dos claves:
1. "summary": Un resumen conciso y atractivo del texto en una sola frase (máximo 25 palabras).
2. "key_topics": Una lista de los 3 a 5 temas, marcas o conceptos más importantes mencionados.

Texto a analizar:
\"\"\"{text[:4000]}\"\"\"

Responde únicamente con el JSON.
"""
    try:
        raw_response = fetch_response(prompt, model="gpt-4o-mini", temperature=0.2, max_tokens=300)
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:-3].strip()
        data = json.loads(raw_response)
        summary = data.get("summary", "No se pudo generar un resumen.")
        key_topics = data.get("key_topics", [])
        return summary, key_topics
    except Exception as e:
        logging.error("❌ Error al generar resumen y temas: %s", e)
        return text[:150] + "...", []

def insert_mention(cur, data: Dict[str, Any]):
    cur.execute(
        """
        INSERT INTO mentions (
            query_id, engine, source, response, sentiment, emotion,
            confidence_score, source_title, source_url, language, created_at,
            summary, key_topics, generated_insight_id,
            status, is_bot, spam_score, duplicate_group_id,
            alert_triggered, alert_reason, engine_latency_ms, error,
            model_name, api_status_code, engine_request_id,
            input_tokens, output_tokens, price_usd,
            analysis_latency_ms, total_pipeline_ms, error_category,
            source_domain, source_rank, query_text, query_topic, poll_id,
            client_id, brand_id, category, embedding
        )
        VALUES (
            %(query_id)s, %(engine)s, %(source)s, %(response)s, %(sentiment)s, %(emotion)s,
            %(confidence)s, %(source_title)s, %(source_url)s, %(language)s, %(created_at)s,
            %(summary)s, %(key_topics)s, %(insight_id)s,
            %(status)s, %(is_bot)s, %(spam_score)s, %(duplicate_group_id)s,
            %(alert_triggered)s, %(alert_reason)s, %(engine_latency_ms)s, %(error)s,
            %(model_name)s, %(api_status_code)s, %(engine_request_id)s,
            %(input_tokens)s, %(output_tokens)s, %(price_usd)s,
            %(analysis_latency_ms)s, %(total_pipeline_ms)s, %(error_category)s,
            %(source_domain)s, %(source_rank)s, %(query_text)s, %(query_topic)s, %(poll_id)s,
            %(client_id)s, %(brand_id)s, %(category)s, %(embedding)s::vector
        )
        RETURNING id
        """,
        {
            **data,
            "key_topics": Json(data.get("key_topics", [])),
        },
    )
    return cur.fetchone()[0]

def insert_insights(cur, query_id: int, insights_payload: dict,
                    client_id: int | None, brand_id: int | None,
                    category: str | None, topic: str | None) -> int:
    """Inserta un insight y asocia metadatos.

    Intenta guardar también el topic del prompt en la columna `topic` si existe.
    Si la columna no existe (entornos antiguos), hace fallback sin `topic`.
    """
    try:
        cur.execute(
            "INSERT INTO insights (query_id, payload, client_id, brand_id, category, topic) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
            (query_id, json.dumps(insights_payload), client_id, brand_id, category, topic),
        )
    except Exception as exc:
        # Fallback si la columna topic no existe en este entorno
        msg = str(exc).lower()
        if "column \"topic\"" in msg or "column topic" in msg:
            cur.execute(
                "INSERT INTO insights (query_id, payload, client_id, brand_id, category) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (query_id, json.dumps(insights_payload), client_id, brand_id, category),
            )
        else:
            raise
    return cur.fetchone()[0]

def run_engine(name: str, fetch_fn: Callable[[str], Union[str, list]],
               query_id: int, query_text: str, query_topic: str, query_category: str | None,
               client_id: int | None, brand_id: int | None, cur, poll_id: str) -> None:
    logging.info("▶ %s | query «%s»", name, query_text)

    try:
        pipeline_start = time.time()
        engine_start = pipeline_start
        results = fetch_fn(query_text)
        fetch_ms = int((time.time() - engine_start) * 1000)
        response_text = ""
        source_title = None
        source_url = None
        source_domain = None
        source_rank = None
        model_name = None
        api_status_code = None
        engine_request_id = None
        input_tokens = None
        output_tokens = None
        price_usd = None
        error_category = None

        if name == "serpapi":
            response_text, structured = get_search_results_structured(query_text, top_k=5)
            if structured:
                top = structured[0]
                source_title = top.get("title")
                source_url = top.get("url")
                source_domain = top.get("domain")
                source_rank = top.get("rank")
            else:
                logging.warning("⚠️ serpapi sin resultados para: %s", query_text)
                return
        else:
            if name == "gpt-4":
                engine_start = time.time()
                response_text, meta = fetch_response_with_metadata(query_text, model="gpt-4o-mini")
                fetch_ms = int((time.time() - engine_start) * 1000)
                model_name = meta.get("model_name")
                api_status_code = meta.get("api_status_code")
                engine_request_id = meta.get("engine_request_id")
                input_tokens = meta.get("input_tokens")
                output_tokens = meta.get("output_tokens")
                price_usd = meta.get("price_usd")
                error_category = meta.get("error_category")
            elif name == "pplx-7b-chat":
                engine_start = time.time()
                response_text, meta = fetch_perplexity_with_metadata(query_text)
                fetch_ms = int((time.time() - engine_start) * 1000)
                model_name = meta.get("model_name")
                api_status_code = meta.get("api_status_code")
                engine_request_id = meta.get("engine_request_id")
                input_tokens = meta.get("input_tokens")
                output_tokens = meta.get("output_tokens")
                price_usd = meta.get("price_usd")
                error_category = meta.get("error_category")
            else:
                response_text = results

        if not response_text or not isinstance(response_text, str):
            logging.warning("⚠️ El motor %s no devolvió una respuesta de texto válida para: %s", name, query_text)
            return

        analysis_start = time.time()
        summary, key_topics = summarize_and_extract_topics(response_text)
        # Generar embedding para el summary
        embedding = None
        try:
            emb_res = openai_client.embeddings.create(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"), input=[summary])
            embedding_vec = emb_res.data[0].embedding
            # Convertimos a literal ::vector para el INSERT
            embedding = "[" + ",".join(f"{float(x):.8f}" for x in embedding_vec) + "]"
        except Exception as e:
            logging.warning("⚠️ No se pudo generar embedding: %s", e)
            embedding = None
        target_for_sentiment = summary if summary and isinstance(summary, str) and len(summary) >= 8 else response_text
        sentiment, emotion, confidence = analyze_sentiment(target_for_sentiment)
        analysis_ms = int((time.time() - analysis_start) * 1000)
        
        insight_id = None
        if name in {"gpt-4", "pplx-7b-chat", "serpapi"}:
            insights_payload = extract_insights(response_text)
            if insights_payload:
                insight_id = insert_insights(cur, query_id, insights_payload, client_id, brand_id, query_category, query_topic)

        alert_triggered = sentiment < SENTIMENT_THRESHOLD
        mention_data = {
            "query_id": query_id, "engine": name, "source": name.lower(), "response": response_text,
            "sentiment": sentiment, "emotion": emotion, "confidence": confidence,
            "source_title": source_title, "source_url": source_url, "created_at": datetime.now(timezone.utc),
            "summary": summary, "key_topics": key_topics, "insight_id": insight_id,
            "status": "active", "is_bot": False, "spam_score": 0.0, "duplicate_group_id": None,
            "alert_triggered": alert_triggered, "alert_reason": ("sentiment_below_threshold" if alert_triggered else None),
            "engine_latency_ms": fetch_ms, "error": None,
            "model_name": model_name,
            "api_status_code": api_status_code,
            "engine_request_id": engine_request_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "price_usd": price_usd,
            "analysis_latency_ms": analysis_ms,
            "total_pipeline_ms": int((time.time() - pipeline_start) * 1000),
            "error_category": error_category,
            "source_domain": source_domain,
            "source_rank": source_rank,
            "query_text": query_text,
            "query_topic": query_topic,
            "language": "unknown",
            "poll_id": poll_id,
            "client_id": client_id,
            "brand_id": brand_id,
            "category": query_category,
            "embedding": embedding if embedding is not None else None,
        }

        mention_id = insert_mention(cur, mention_data)

        if alert_triggered:
            send_slack_alert(query_text, sentiment, summary)

        logging.info("✓ %s guardado (mention_id=%s, insight_id=%s)", name, mention_id, insight_id)

    except Exception as exc:
        logging.exception("❌ %s error: %s", name, exc)

def main(loop_once: bool = True, sleep_seconds: int = 6 * 3600):
    logging.info("🔄 Polling service started")
    poll_id = f"poll_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    logging.info(f"🚀 Starting new poll run with ID: {poll_id}")
    while True:
        with psycopg2.connect(**DB_CFG) as conn:
            with conn.cursor() as cur:
                base_sql = (
                    "SELECT id, query, topic, category, client_id, brand_id, "
                    "COALESCE(project_id, id) AS project_id "
                    "FROM queries WHERE enabled = TRUE"
                )
                where_clauses = []
                params: Dict[str, Any] = {}
                if POLL_PROJECT_ID is not None:
                    where_clauses.append("COALESCE(project_id, id) = %(pid)s")
                    params["pid"] = POLL_PROJECT_ID
                if POLL_CLIENT_ID is not None:
                    where_clauses.append("client_id = %(cid)s")
                    params["cid"] = POLL_CLIENT_ID
                if POLL_BRAND_ID is not None:
                    where_clauses.append("brand_id = %(bid)s")
                    params["bid"] = POLL_BRAND_ID
                sql = base_sql + (" AND " + " AND ".join(where_clauses) if where_clauses else "")

                logging.info(
                    "Consultando queries con filtros pid=%s, cid=%s, bid=%s",
                    POLL_PROJECT_ID, POLL_CLIENT_ID, POLL_BRAND_ID,
                )
                cur.execute(sql, params)
                for query_id, query_text, query_topic, query_category, client_id, brand_id, project_id in cur.fetchall():
                    print(f"\n🔍 Buscando menciones para query: {query_text}")
                    for name, fn in (
                        ("gpt-4", lambda q: fetch_response(q, model="gpt-4o-mini")),
                        ("pplx-7b-chat", fetch_perplexity_response),
                        ("serpapi", fetch_serp_response),
                    ):
                        run_engine(name, fn, query_id, query_text, query_topic, query_category, client_id, brand_id, cur, poll_id)
                conn.commit()

        logging.info(f"🛑 Polling cycle finished for poll_id={poll_id}")
        if loop_once:
            break
        time.sleep(sleep_seconds)

if __name__ == "__main__":
    main(loop_once=True)