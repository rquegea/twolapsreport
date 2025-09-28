from typing import Dict, List, Any
import os
from datetime import datetime, timezone

from . import aggregator
from . import plotter
from . import pdf_writer
from ..engines.openai_engine import fetch_response
from ..engines import strategic_prompts as s_prompts
import json
from typing import Optional


def _build_kpi_rows(kpis: Dict) -> List[List[str]]:
    return [
        ["Marca", str(kpis.get("brand_name", "-"))],
        ["Total menciones", str(kpis.get("total_mentions", 0))],
        ["Sentimiento medio", f"{kpis.get('sentiment_avg', 0.0):.2f}"],
        ["SOV", f"{kpis.get('sov', 0.0):.1f}%"],
    ]


def _compact_aggregated_for_prompt(
    aggregated: Dict[str, Any],
    *,
    top_clusters: int = 10,
    examples_per_cluster: int = 3,
    max_sov_entries: int = 12,
    max_series_points: int = 180,
) -> Dict[str, Any]:
    """Reduce el tamaño del bundle conservando señal para el LLM.
    - Prioriza TOP-N clusters por volumen y recorta ejemplos por cluster
    - Limita longitud de series temporales (últimos puntos)
    - Recorta tabla SOV a primeras entradas
    - Reduce buckets de agent_insights para no saturar el prompt
    """
    out: Dict[str, Any] = {}

    # KPIs y metadatos clave
    kpis: Dict[str, Any] = dict(aggregated.get("kpis", {}))
    if isinstance(kpis.get("sov_table"), list):
        kpis["sov_table"] = kpis["sov_table"][:max_sov_entries]
    out["kpis"] = kpis
    if "client_name" in aggregated:
        out["client_name"] = aggregated.get("client_name")
    if "market_competitors" in aggregated:
        out["market_competitors"] = aggregated.get("market_competitors")

    # Series: quedarnos con la parte más reciente para contexto actual
    time_series = aggregated.get("time_series") or {}
    if isinstance(time_series, dict):
        sent_series = list(time_series.get("sentiment_per_day") or [])
        if len(sent_series) > max_series_points:
            sent_series = sent_series[-max_series_points:]
        out["time_series"] = {"sentiment_per_day": sent_series}
    vis_series = aggregated.get("visibility_timeseries") or []
    if isinstance(vis_series, (list, tuple)):
        vis_list = list(vis_series)
        if len(vis_list) > max_series_points:
            vis_list = vis_list[-max_series_points:]
        out["visibility_timeseries"] = vis_list

    # Agent insights (buckets) con límite por categoría
    ai = aggregated.get("agent_insights") or {}
    buckets = {}
    if isinstance(ai, dict):
        raw_buckets = ai.get("buckets") or {}
        if isinstance(raw_buckets, dict):
            for key in ("opportunities", "risks", "trends", "quotes", "pain_points"):
                items = raw_buckets.get(key) or []
                if isinstance(items, list):
                    buckets[key] = items[:10]
        out["agent_insights"] = {"buckets": buckets}

    # Clusters: usar crudos si existen; ordenar por volumen y recortar ejemplos
    clusters = aggregated.get("clusters_raw") or aggregated.get("clusters") or []
    if isinstance(clusters, list):
        def _count(c: Dict[str, Any]) -> int:
            try:
                return int(c.get("count", 0))
            except Exception:
                return 0
        top = sorted(clusters, key=_count, reverse=True)[:top_clusters]
        comp: List[Dict[str, Any]] = []
        for c in top:
            item: Dict[str, Any] = {
                "topic_name": c.get("topic_name") or c.get("label"),
                "count": _count(c),
                "avg_sentiment": float(c.get("avg_sentiment", 0.0) or 0.0),
            }
            ts = c.get("top_sources") or []
            if isinstance(ts, list):
                item["top_sources"] = ts[:5]
            ex = c.get("example_mentions") or []
            if isinstance(ex, list):
                item["example_mentions"] = ex[:examples_per_cluster]
            comp.append(item)
        out["clusters"] = comp

    # Anexos básicos si vienen ya calculados
    for k in ("sentiment_by_category", "topics_top5", "topics_bottom5"):
        if k in aggregated:
            out[k] = aggregated.get(k)

    return out


def _extract_insights_to_json(aggregated: Dict[str, Any]) -> Dict[str, Any]:
    compact = _compact_aggregated_for_prompt(
        aggregated,
        top_clusters=10,
        examples_per_cluster=3,
        max_sov_entries=12,
        max_series_points=180,
    )
    prompt = s_prompts.get_insight_extraction_prompt(compact)
    raw = fetch_response(prompt, model="gpt-4o", temperature=0.2, max_tokens=2048)
    if not raw:
        return {}
    try:
        text = raw.strip()
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        import json
        data = json.loads(text)
        return {
            "executive_summary": data.get("executive_summary", "") if isinstance(data.get("executive_summary"), str) else "",
            "key_findings": data.get("key_findings", []),
            "opportunities": data.get("opportunities", []),
            "risks": data.get("risks", []),
            "recommendations": data.get("recommendations", []),
            "time_series_analysis": data.get("time_series_analysis", {}),
        }
    except Exception:
        return {}


def _generate_full_part2_texts(insights_json: Dict[str, Any], aggregated: Dict[str, Any]) -> Dict[str, str]:
    """Genera todos los textos de la Parte 2 invocando prompts especialistas.
    Aplica limpieza robusta: elimina fences y, si la IA devuelve JSON, lo parsea
    y lo transforma a párrafos/bullets legibles.
    """
    out: Dict[str, str] = {}

    def _strip_fences(text: str) -> str:
        if not text:
            return ""
        s = text.strip()
        # Quitar fences tipo ```json ... ``` o ``` ... ```
        if s.startswith("```json"):
            s = s[len("```json"):].strip()
            if s.endswith("```"):
                s = s[:-3].strip()
        if s.startswith("```") and s.endswith("```"):
            s = s[3:-3].strip()
        # Quitar prefijo "json" si viene suelto
        if s.lower().startswith("json ") or s.lower().startswith("json\n"):
            s = s[4:].strip()
        # Si hay texto previo y luego un JSON, extraer desde el primer '{' o '['
        try:
            first_brace = s.find('{')
            first_brack = s.find('[')
            pos = min([p for p in [first_brace, first_brack] if p != -1]) if (first_brace != -1 or first_brack != -1) else -1
            if pos > 0:
                # Si antes solo hay "json" o espacio, recortar
                prefix = s[:pos].strip().lower()
                if prefix in ("", "json"):
                    s = s[pos:]
        except Exception:
            pass
        return s

    def _to_text_from_structure(data: Any, section: str | None = None) -> str:
        # Especial: plan de acción
        if section == "action_plan":
            plan_items: list[str] = []
            src = None
            if isinstance(data, dict):
                # Estructuras comunes
                for key in ("plan_de_accion_estrategico", "plan_estrategico", "plan", "acciones", "actions"):
                    if isinstance(data.get(key), list):
                        src = data.get(key); break
                if src is None and isinstance(data.get("recommendations"), list):
                    src = data.get("recommendations")
            elif isinstance(data, list):
                src = data
            if isinstance(src, list):
                for it in src:
                    if not isinstance(it, dict):
                        plan_items.append("- " + str(it))
                        continue
                    # Caso A: item con prioridad/accion directamente
                    pr = it.get("prioridad") or it.get("priority")
                    ac = it.get("accion") or it.get("action") or it.get("tarea")
                    plazo = it.get("plazo") or it.get("timeline") or it.get("due")
                    owner = it.get("owner") or it.get("responsable")
                    just = it.get("justificacion") or it.get("justification")
                    recs = it.get("recomendaciones") or it.get("recommendations")
                    if ac:
                        bullet = "- " + (f"[{pr}] " if pr else "") + ac
                        suffix = []
                        if plazo: suffix.append(f"plazo: {plazo}")
                        if owner: suffix.append(f"owner: {owner}")
                        if suffix:
                            bullet += " — (" + ", ".join(suffix) + ")"
                        plan_items.append(bullet)
                        if just:
                            plan_items.append(f"  • Justificación: {just}")
                        if isinstance(recs, list):
                            for r in recs[:5]:
                                plan_items.append("  • " + str(r))
                        continue
                    # Caso B: item con 'acciones' anidadas por plazo u otra dimensión
                    acciones = it.get("acciones") or it.get("actions")
                    plazo_item = it.get("plazo") or it.get("timeline")
                    if isinstance(acciones, list):
                        for a in acciones:
                            if not isinstance(a, dict):
                                plan_items.append("- " + str(a))
                                continue
                            act = a.get("accion") or a.get("action") or a.get("tarea") or json.dumps(a, ensure_ascii=False)
                            pr2 = a.get("prioridad") or a.get("priority") or pr
                            just2 = a.get("justificacion") or a.get("justification")
                            recs2 = a.get("recomendaciones") or a.get("recommendations")
                            line = "- " + (f"[{pr2}] " if pr2 else "") + (f"[{plazo_item}] " if plazo_item else "") + act
                            plan_items.append(line)
                            if just2:
                                plan_items.append(f"  • Justificación: {just2}")
                            if isinstance(recs2, list):
                                for r in recs2[:5]:
                                    plan_items.append("  • " + str(r))
                        continue
                    # Fallback: volcar pares clave-valor legibles
                    plan_items.append("- " + _to_text_from_structure(it, section))
                return "\n".join([ln for ln in plan_items if ln.strip()])
            # fallback genérico
            return _to_text_from_structure({"plan": data})

        # Genérico
        if isinstance(data, str):
            return data.strip()
        if isinstance(data, list):
            lines: list[str] = []
            for it in data:
                if isinstance(it, (dict, list)):
                    txt = _to_text_from_structure(it, section)
                    if txt:
                        for ln in txt.splitlines():
                            lines.append("- " + ln if not ln.startswith("-") else ln)
                else:
                    lines.append("- " + str(it))
            return "\n".join(lines)
        if isinstance(data, dict):
            lines: list[str] = []
            for k, v in data.items():
                key = str(k).strip().replace("_", " ").capitalize()
                if isinstance(v, (dict, list)):
                    sub = _to_text_from_structure(v, section)
                    if sub:
                        lines.append(f"{key}:")
                        lines.extend([("- " + ln if not ln.startswith("-") else ln) for ln in sub.splitlines()])
                else:
                    val = str(v).strip()
                    if val:
                        lines.append(f"- {key}: {val}")
            return "\n".join(lines)
        try:
            return str(data)
        except Exception:
            return ""

    def _normalize_section(raw: str, section: str) -> str:
        s = _strip_fences(raw)
        # Intentar parsear JSON si aplica
        parsed: Any | None = None
        try:
            if s.startswith("{") or s.startswith("["):
                parsed = json.loads(s)
        except Exception:
            parsed = None
        if parsed is not None:
            return _to_text_from_structure(parsed, section)
        return s

    # Resumen Ejecutivo (experto)
    try:
        exec_prompt = s_prompts.get_executive_summary_prompt(aggregated)
        out["executive_summary"] = _normalize_section(fetch_response(exec_prompt, model="gpt-4o", temperature=0.3, max_tokens=900), "executive_summary")
    except Exception:
        out["executive_summary"] = _normalize_section(insights_json.get("executive_summary", ""), "executive_summary")

    # Análisis Competitivo (usa KPIs agregados ya presentes en aggregated)
    try:
        comp_prompt = s_prompts.get_competitive_analysis_prompt(aggregated)
        out["competitive_analysis"] = _normalize_section(fetch_response(comp_prompt, model="gpt-4o-mini", temperature=0.3, max_tokens=900), "competitive_analysis")
    except Exception:
        out["competitive_analysis"] = ""

    # Tendencias y Señales (usa aggregated.trends)
    try:
        trends_prompt = s_prompts.get_trends_anomalies_prompt(aggregated)
        out["trends"] = _normalize_section(fetch_response(trends_prompt, model="gpt-4o-mini", temperature=0.3, max_tokens=900), "trends")
    except Exception:
        out["trends"] = ""

    # Correlaciones Transversales (usar datos del agregador para contexto adicional)
    try:
        base_corr = insights_json.get("time_series_analysis", {}) or {}
        corr_input = {
            **base_corr,
            "basic": aggregated.get("correlations", {}),
            "sov_by_category": ((aggregated.get("sov") or {}).get("current") or {}).get("sov_by_category", {}),
            "sentiment_by_category": aggregated.get("sentiment_by_category", {}),
            "time_series": aggregated.get("time_series", {}),
            "visibility_timeseries": aggregated.get("visibility_timeseries", []),
        }
        corr_prompt = s_prompts.get_correlation_interpretation_prompt(aggregated, corr_input)
        out["correlations"] = _normalize_section(fetch_response(corr_prompt, model="gpt-4o-mini", temperature=0.3, max_tokens=1100), "correlations")
        # Fallback textual si el LLM devuelve vacío: usar métricas calculadas
        if not out["correlations"]:
            try:
                corr = aggregated.get("correlations", {}) or {}
                lines: list[str] = []
                def _fmt_r(d: dict | None, label: str) -> None:
                    if isinstance(d, dict):
                        r = d.get("pearson_r", None)
                        n = d.get("n", None)
                        if r is not None or n:
                            lines.append(f"- {label}: r={(float(r)):.2f} | n={int(n or 0)}" if r is not None else f"- {label}: r=N/D | n={int(n or 0)}")
                _fmt_r(corr.get("sentiment_vs_mentions"), "Sentimiento vs Menciones")
                _fmt_r(corr.get("visibility_vs_mentions"), "Visibilidad vs Menciones")
                _fmt_r(corr.get("visibility_vs_sentiment"), "Visibilidad vs Sentimiento")
                # Mejor lag volumen→sentimiento
                best = ((corr.get("lags_volume_to_sentiment") or {}).get("best") or {}) if isinstance(corr.get("lags_volume_to_sentiment"), dict) else {}
                if best:
                    lag = best.get("lag")
                    r_best = best.get("r")
                    if lag is not None or r_best is not None:
                        try:
                            lines.append(f"- Mejor lag volumen→sentimiento: {int(lag)} días | r={(float(r_best)):.2f}")
                        except Exception:
                            lines.append(f"- Mejor lag volumen→sentimiento: {int(lag) if lag is not None else 'N/D'} días | r=N/D")
                # Top oportunidades/risks por categoría
                quads = corr.get("category_quadrants") or {}
                if isinstance(quads, dict):
                    def _pick(key: str, title: str) -> None:
                        items = quads.get(key) or []
                        if isinstance(items, list) and items:
                            lines.append(title + ":")
                            for it in items[:3]:
                                try:
                                    cat = str(it.get("category"))
                                    sovp = float(it.get("sov_pct", 0.0))
                                    sen = float(it.get("sentiment", 0.0))
                                    lines.append(f"  • {cat} — SOV {sovp:.1f}% | Sent {sen:.2f}")
                                except Exception:
                                    continue
                    _pick("top_opportunities", "Top oportunidades por categoría")
                    _pick("top_risks", "Top riesgos por categoría")
                if lines:
                    out["correlations"] = "Correlaciones Transversales entre Categorías (resumen basado en datos):\n" + "\n".join(lines)
            except Exception:
                pass
    except Exception:
        out["correlations"] = ""

    # Plan de Acción Estratégico (oportunidades + riesgos + recomendaciones)
    try:
        plan_prompt = s_prompts.get_strategic_plan_prompt({
            "opportunities": insights_json.get("opportunities", []),
            "risks": insights_json.get("risks", []),
            "recommendations": insights_json.get("recommendations", []),
        }, client_name=(aggregated.get("client_name") or "Cliente"))
        out["action_plan"] = _normalize_section(fetch_response(plan_prompt, model="gpt-4o", temperature=0.3, max_tokens=1100), "action_plan")
    except Exception:
        out["action_plan"] = ""

    # Resumen Ejecutivo y Hallazgos — ahora con TODO el contexto disponible (incluye SOV por categoría si existe)
    try:
        sov_by_category = ((aggregated.get("sov") or {}).get("current") or {}).get("sov_by_category", {})
        sum_prompt = s_prompts.get_strategic_summary_prompt({
            "executive_summary": insights_json.get("executive_summary", ""),
            "key_findings": insights_json.get("key_findings", []),
            # Contexto ampliado para cubrir todos los apartados del PDF
            "context": {
                "competitive_analysis": out.get("competitive_analysis", ""),
                "trends": out.get("trends", ""),
                "correlations": out.get("correlations", ""),
                "action_plan": out.get("action_plan", ""),
                "sov_by_category": sov_by_category,
                "kpis": aggregated.get("kpis", {}),
            },
        }, client_name=(aggregated.get("client_name") or "Cliente"))
        out["summary_and_findings"] = _normalize_section(
            fetch_response(sum_prompt, model="gpt-4o", temperature=0.3, max_tokens=900),
            "summary_and_findings",
        )
    except Exception:
        out["summary_and_findings"] = out.get("executive_summary", "")

    return out


def _analyze_cluster(cluster_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Llama al Analista de Clusters (Nivel 1) y devuelve un dict con topic_name y key_points.
    Devuelve valores seguros en caso de error.
    """
    try:
        prompt = s_prompts.get_cluster_analyst_prompt(cluster_obj)
        raw = fetch_response(prompt, model="gpt-4o-mini", temperature=0.2, max_tokens=500)
        if not raw:
            return {"topic_name": "(sin nombre)", "key_points": []}
        text = raw.strip()
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        data = json.loads(text)
        topic = data.get("topic_name") or "(sin nombre)"
        pts = data.get("key_points") or []
        if not isinstance(pts, list):
            pts = []
        return {"topic_name": str(topic), "key_points": [str(p) for p in pts][:5]}
    except Exception:
        return {"topic_name": "(sin nombre)", "key_points": []}


def _synthesize_clusters(cluster_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Llama al Sintetizador Estratégico (Nivel 2) con el resumen de clusters."""
    try:
        prompt = s_prompts.get_clusters_synthesizer_prompt([
            {
                "topic_name": c.get("topic_name", "(sin nombre)"),
                "volume": int(c.get("volume", 0)),
                "sentiment": float(c.get("sentiment", 0.0)),
            }
            for c in cluster_summaries
        ])
        raw = fetch_response(prompt, model="gpt-4o", temperature=0.2, max_tokens=900)
        if not raw:
            return {}
        text = raw.strip()
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        return json.loads(text)
    except Exception:
        return {}


def generate_report(project_id: int, clusters: List[Dict[str, Any]] | None = None,
                    save_insights_json: bool = True,
                    *, start_date: str | None = None, end_date: str | None = None,
                    client_brand: str | None = None, competitors: List[str] | None = None) -> bytes:
    session = aggregator.get_session()
    try:
        # 1) Nombre de marca del proyecto (solo para mostrar)
        #    Usar ventana temporal cuando esté disponible
        kpis_name_only = aggregator.get_kpi_summary(session, project_id, start_date=start_date, end_date=end_date, client_brand=client_brand)
        brand_name = kpis_name_only.get("brand_name") or "Empresa"
        # Canonicalizar la marca a su forma oficial si coincide con sinónimos
        try:
            from .aggregator import BRAND_SYNONYMS
            low = str(brand_name).strip().lower()
            for canon, alts in BRAND_SYNONYMS.items():
                variants = [canon.lower()] + [s.lower() for s in alts]
                if low in variants:
                    brand_name = canon
                    break
        except Exception:
            pass

        # 2) Métricas y gráficos del MERCADO del proyecto
        #    SOV del mercado actual por marca: usar detección robusta a nivel de menciones
        sov_pairs_all = aggregator.get_industry_sov_ranking(session, project_id, start_date=start_date, end_date=end_date)
        # Filtrado por competidores si se proporcionan: siempre incluir la marca del cliente
        comp_set = set([(c or "").strip() for c in (competitors or []) if (c or "").strip()])
        if comp_set:
            allowed = {str(brand_name).strip()} | comp_set
            sov_pairs = [(n, v) for n, v in sov_pairs_all if str(n).strip() in allowed]
            if not sov_pairs:
                sov_pairs = [(n, v) for n, v in sov_pairs_all if str(n).strip() == str(brand_name).strip()]
        else:
            sov_pairs = list(sov_pairs_all)
        # Recalcular el SOV del cliente relativo al subset (renormalizando)
        total_subset_pct = float(sum(float(v or 0.0) for _, v in sov_pairs)) or 1.0
        brand_pct_all = next((float(v or 0.0) for n, v in sov_pairs_all if str(n).strip() == str(brand_name).strip()), 0.0)
        brand_sov = round((brand_pct_all / total_subset_pct) * 100.0, 1)

        #    Visibilidad global diaria
        vis_dates, vis_vals = aggregator.get_visibility_series(session, project_id, start_date=start_date, end_date=end_date, client_brand=client_brand)

        #    Sentimiento: serie diaria de promedio [-1, 1] FILTRADO por la marca seleccionada
        sent_evo = aggregator.get_sentiment_evolution(session, project_id, start_date=start_date, end_date=end_date, client_brand=client_brand)

        #    Datos por categoría solo para el anexo y gráficos secundarios
        by_cat = aggregator.get_sentiment_by_category(session, project_id, start_date=start_date, end_date=end_date, client_brand=client_brand)
        top5, bottom5 = aggregator.get_topics_by_sentiment(session, project_id, start_date=start_date, end_date=end_date, client_brand=client_brand)

        #    Total de menciones del periodo filtradas por mercado y, si procede, por marca del cliente
        from sqlalchemy import text as _text
        s_date = start_date or "1970-01-01"
        e_date = end_date or "2999-12-31"
        if client_brand and str(client_brand).strip():
            base = str(client_brand).strip()
            syns = [base.lower()]
            from .aggregator import BRAND_SYNONYMS
            syns.extend([s.lower() for s in BRAND_SYNONYMS.get(base, [])])
            likes = [f"%{s}%" for s in syns]
            total_mentions_row = session.execute(_text("""
                WITH rows AS (
                    SELECT 1 AS hit
                    FROM mentions m
                    JOIN queries q ON q.id = m.query_id
                    LEFT JOIN insights i ON i.id = m.generated_insight_id
                    WHERE COALESCE(q.project_id, q.id) = COALESCE((SELECT project_id FROM queries WHERE id = :pid), :pid)
                      AND m.created_at >= CAST(:start AS date)
                      AND m.created_at < (CAST(:end AS date) + INTERVAL '1 day')
                      AND (
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
                      )
                )
                SELECT COUNT(*) FROM rows
            """), {"pid": int(project_id), "start": s_date, "end": e_date, "syns": syns, "likes": likes}).first()
            total_mentions = int(total_mentions_row[0] if total_mentions_row and total_mentions_row[0] is not None else 0)
        else:
            total_mentions_row = session.execute(_text("""
                SELECT COUNT(*)
                FROM mentions m
                JOIN queries q ON q.id = m.query_id
                WHERE COALESCE(q.project_id, q.id) = COALESCE((SELECT project_id FROM queries WHERE id = :pid), :pid)
                  AND m.created_at >= CAST(:start AS date)
                  AND m.created_at < (CAST(:end AS date) + INTERVAL '1 day')
            """), {"pid": int(project_id), "start": s_date, "end": e_date}).first()
            total_mentions = int(total_mentions_row[0] if total_mentions_row and total_mentions_row[0] is not None else 0)

        #    Motores de IA utilizados en el periodo (para ficha técnica)
        ai_engines_used: list[str] = []
        try:
            rows = session.execute(_text(
                """
                SELECT DISTINCT LOWER(COALESCE(m.engine, '')) AS engine,
                                COALESCE(m.model_name, '') AS model_name
                FROM mentions m
                JOIN queries q ON q.id = m.query_id
                WHERE COALESCE(q.project_id, q.id) = COALESCE((SELECT project_id FROM queries WHERE id = :pid), :pid)
                  AND m.created_at >= CAST(:start AS date)
                  AND m.created_at < (CAST(:end AS date) + INTERVAL '1 day')
                """
            ), {"pid": int(project_id), "start": s_date, "end": e_date}).all()

            labels: set[str] = set()
            for eng, model in rows:
                eng_l = (eng or "").lower().strip()
                mdl = (model or "").strip()
                if not eng_l and not mdl:
                    continue
                # OpenAI
                if eng_l.startswith("gpt") or mdl.lower().startswith("gpt"):
                    labels.add(f"OpenAI {mdl or 'GPT'}")
                    continue
                # Perplexity
                if "pplx" in eng_l or "perplexity" in mdl.lower() or "sonar" in mdl.lower():
                    lbl = f"Perplexity {mdl}".strip()
                    labels.add(lbl if lbl != "Perplexity" else "Perplexity")
                    continue
                # Fallback: usar el nombre del engine si existe
                if eng_l:
                    labels.add(eng_l.upper())
            ai_engines_used = sorted([l for l in labels if l])
        except Exception:
            ai_engines_used = []

        # 3) Insights del agente para Parte 2
        agent_insights = aggregator.get_agent_insights_data(session, project_id, limit=200)
        # Nuevo: permitir inyectar clusters precalculados para evitar recomputar
        if clusters is None:
            clusters = aggregator.aggregate_clusters_for_report(session, project_id, start_date=start_date, end_date=end_date, max_rows=5000)
    finally:
        session.close()

    # KPI: sentimiento medio global del periodo (media de la serie diaria)
    sentiment_avg = (sum(v for _, v in sent_evo) / max(len(sent_evo), 1)) if sent_evo else 0.0

    # Derivar lista simple de competidores desde la tabla de SOV
    # Competidores para prompts: respetar subconjunto si llegó
    if comp_set:
        competitors_list = [n for n, _ in sov_pairs if str(n).strip() != str(brand_name).strip()]
    else:
        competitors_list = [str(n) for n, _ in sov_pairs if str(n).strip() != str(brand_name).strip()]

    # --- Deltas y benchmark (previo periodo de igual longitud) ---
    prev_start_date: str | None = None
    prev_end_date: str | None = None
    try:
        if start_date and end_date:
            from datetime import datetime as _dt, timedelta as _td
            sd = _dt.strptime(start_date, "%Y-%m-%d").date()
            ed = _dt.strptime(end_date, "%Y-%m-%d").date()
            span = (ed - sd).days + 1
            prev_end = sd - _td(days=1)
            prev_start = prev_end - _td(days=span - 1)
            prev_start_date = prev_start.strftime("%Y-%m-%d")
            prev_end_date = prev_end.strftime("%Y-%m-%d")
    except Exception:
        prev_start_date, prev_end_date = None, None

    # Recalcular referencias previas si hay ventana previa
    sov_prev_brand = None
    sent_prev_avg = None
    vis_prev_avg = None
    try:
        if prev_start_date and prev_end_date:
            sess2 = aggregator.get_session()
            try:
                sov_prev_pairs = aggregator.get_industry_sov_ranking(sess2, project_id, start_date=prev_start_date, end_date=prev_end_date)
                sov_prev_brand = next((float(v or 0.0) for n, v in sov_prev_pairs if str(n).strip() == str(brand_name).strip()), 0.0)
                sent_prev_series = aggregator.get_sentiment_evolution(sess2, project_id, start_date=prev_start_date, end_date=prev_end_date, client_brand=client_brand)
                sent_prev_avg = (sum(v for _, v in sent_prev_series) / max(len(sent_prev_series), 1)) if sent_prev_series else 0.0
                vis_prev_dates, vis_prev_vals = aggregator.get_visibility_series(sess2, project_id, start_date=prev_start_date, end_date=prev_end_date, client_brand=client_brand)
                vis_prev_avg = (sum(vis_prev_vals) / max(len(vis_prev_vals), 1)) if vis_prev_vals else 0.0
            finally:
                sess2.close()
    except Exception:
        sov_prev_brand, sent_prev_avg, vis_prev_avg = None, None, None

    # Calcular deltas simples (MoM/WoW dependiendo de la ventana)
    try:
        sov_delta = (brand_sov - float(sov_prev_brand)) if sov_prev_brand is not None else None
    except Exception:
        sov_delta = None
    try:
        vis_curr_avg = (sum(vis_vals) / max(len(vis_vals), 1)) if vis_vals else 0.0
        vis_delta = (vis_curr_avg - float(vis_prev_avg)) if vis_prev_avg is not None else None
    except Exception:
        vis_delta = None
    try:
        sent_delta = (float(sentiment_avg) - float(sent_prev_avg)) if sent_prev_avg is not None else None
    except Exception:
        sent_delta = None

    # Detectar anomalías simples en volumen de menciones, sentimiento y visibilidad
    anomalies: list[dict[str, object]] = []
    try:
        mv_series = aggregator.get_mentions_volume_series(None, project_id, start_date=start_date, end_date=end_date)
        # Porcentaje de cambio día a día y marcar top 2
        changes: list[tuple[str, float]] = []
        for i in range(1, len(mv_series)):
            _, c0 = mv_series[i - 1]
            d1, c1 = mv_series[i]
            base = max(1, int(c0 or 0))
            pct = (int(c1 or 0) - base) / float(base)
            changes.append((d1, pct))
        for d, pct in sorted(changes, key=lambda x: abs(x[1]), reverse=True)[:2]:
            anomalies.append({"date": d, "metric": "menciones", "magnitude": f"{pct:+.0%}", "causal_hypothesis": "Revisión manual sugerida", "confidence": "Media"})
        # Anomalía en sentimiento: pico y valle
        if sent_evo:
            min_day, min_val = min(sent_evo, key=lambda x: x[1])
            max_day, max_val = max(sent_evo, key=lambda x: x[1])
            anomalies.append({"date": max_day, "metric": "sentimiento", "magnitude": f"{max_val:+.2f}", "causal_hypothesis": "Revisión cualitativa de menciones", "confidence": "Baja"})
            anomalies.append({"date": min_day, "metric": "sentimiento", "magnitude": f"{min_val:+.2f}", "causal_hypothesis": "Revisión cualitativa de menciones", "confidence": "Baja"})
        # Anomalías de visibilidad (misma serie que el gráfico)
        if vis_dates and vis_vals and len(vis_vals) >= 2:
            # Deltas día a día en puntos porcentuales
            vis_changes: list[tuple[str, float]] = []
            for i in range(1, len(vis_vals)):
                d = vis_dates[i]
                diff_pp = float(vis_vals[i]) - float(vis_vals[i-1])
                vis_changes.append((d, diff_pp))
            for d, diff_pp in sorted(vis_changes, key=lambda x: abs(x[1]), reverse=True)[:2]:
                anomalies.append({"date": d, "metric": "visibilidad", "magnitude": f"{diff_pp:+.1f} pp", "causal_hypothesis": "Cambio abrupto en apariciones por consulta", "confidence": "Media"})
            # Registrar pico y valle absolutos
            max_idx = int(max(range(len(vis_vals)), key=lambda i: vis_vals[i]))
            min_idx = int(min(range(len(vis_vals)), key=lambda i: vis_vals[i]))
            anomalies.append({"date": vis_dates[max_idx], "metric": "visibilidad", "magnitude": f"{float(vis_vals[max_idx]):.1f}%", "causal_hypothesis": "Máximo de presencia por consulta", "confidence": "Media"})
            anomalies.append({"date": vis_dates[min_idx], "metric": "visibilidad", "magnitude": f"{float(vis_vals[min_idx]):.1f}%", "causal_hypothesis": "Mínimo de presencia por consulta", "confidence": "Media"})
    except Exception:
        pass

    aggregated: Dict[str, Any] = {
        "kpis": {
            "total_mentions": total_mentions,
            "sentiment_avg": float(sentiment_avg),
            "average_sentiment": float(sentiment_avg),  # alias esperado por prompts
            "sov": float(brand_sov),
            "share_of_voice": float(brand_sov),  # alias esperado por prompts
            "sov_table": [(n, v) for n, v in sov_pairs],
            "brand_name": brand_name,
            # Añadir sentimientos por categoría para deep-dives por tema
            "sentiment_by_category": by_cat,
            # Deltas resumidos (si disponibles)
            "delta_sov_prev": float(sov_delta) if sov_delta is not None else None,
            "delta_sentiment_prev": float(sent_delta) if sent_delta is not None else None,
            "delta_visibility_prev": float(vis_delta) if vis_delta is not None else None,
            # Visibilidad media (misma definición que el gráfico)
            "visibility_avg": float((sum(vis_vals) / max(len(vis_vals), 1)) if vis_vals else 0.0),
        },
        "client_name": brand_name,
        "market_competitors": competitors_list,
        "time_series": {
            "sentiment_per_day": [(d, float(v)) for d, v in sent_evo],
        },
        "visibility_timeseries": (vis_dates, vis_vals),
        "agent_insights": agent_insights,
        "clusters_raw": clusters,
        # Datos por categoría SOLO para anexos/gráficos secundarios
        "sentiment_by_category": by_cat,
        "topics_top5": top5,
        "topics_bottom5": bottom5,
        # Cambios y tendencias
        "trends": {
            "anomalies": anomalies,
            "momentum": {
                "sov_delta": float(sov_delta) if sov_delta is not None else None,
                "sentiment_delta": float(sent_delta) if sent_delta is not None else None,
                "visibility_delta": float(vis_delta) if vis_delta is not None else None,
            },
        },
        "previous_period": {
            "start_date": prev_start_date,
            "end_date": prev_end_date,
        },
    }

    # Inyectar SOV por categoría y tendencias ANTES de generar textos para que el resumen no advierta carencia
    try:
        sov_trends_extra = aggregator.get_share_of_voice_and_trends(
            None,
            project_id,
            start_date=start_date or "1970-01-01",
            end_date=end_date or "2999-12-31",
            client_brand=brand_name,
        )
        aggregated["sov"] = sov_trends_extra
    except Exception:
        aggregated["sov"] = {}

    # Nivel 1: análisis por cluster
    cluster_summaries: List[Dict[str, Any]] = []
    for c in (clusters or [])[:12]:  # límite defensivo para rendimiento
        cluster_obj = {
            "count": int(c.get("count", 0)),
            "avg_sentiment": float(c.get("avg_sentiment", 0.0)),
            "top_sources": c.get("top_sources", []),
            "example_mentions": c.get("example_mentions", []),
        }
        analyzed = _analyze_cluster(cluster_obj)
        cluster_summaries.append({
            "topic_name": analyzed.get("topic_name", "(sin nombre)"),
            "key_points": analyzed.get("key_points", []),
            "volume": int(cluster_obj["count"]),
            "sentiment": float(cluster_obj["avg_sentiment"]),
        })

    # Nivel 2: síntesis estratégica a partir de clusters
    synthesis = _synthesize_clusters(cluster_summaries)

    # Cadena de agentes (Parte 2 - estilo twolaps):
    # 1) Extracción de insights (JSON estructurado)
    insights_json = _extract_insights_to_json(aggregated)
    # Guardar JSON para trazabilidad si se solicita
    if save_insights_json and insights_json:
        try:
            backend_dir = os.path.dirname(os.path.dirname(__file__))  # backend/src
            files_dir = os.path.join(os.path.dirname(backend_dir), "files")  # backend/files
            os.makedirs(files_dir, exist_ok=True)
            fname = f"insights_extraction_p{project_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.json"
            with open(os.path.join(files_dir, fname), "w", encoding="utf-8") as f:
                json.dump(insights_json, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # Catálogo de evidencias a partir de clusters
    try:
        evidence_catalog: dict[str, str] = {}
        evidence_index: list[dict[str, str]] = []
        for c in (clusters or [])[:12]:
            topic = str(c.get("topic_name") or c.get("label") or "(sin tema)")
            for ex in (c.get("example_mentions") or [])[:3]:
                mid = str(ex.get("id") or "").strip()
                if not mid:
                    continue
                snippet = str((ex.get("summary") or ex.get("text") or "")[:300])
                if mid not in evidence_catalog:
                    evidence_catalog[mid] = snippet
                    evidence_index.append({"id": mid, "topic": topic, "snippet": snippet})
    except Exception:
        evidence_catalog, evidence_index = {}, []

    # 1.5) Challenger: depurar oportunidades/risks/recomendaciones
    refined = {}
    try:
        challenger_input = {
            "client_name": aggregated.get("client_name"),
            "opportunities": insights_json.get("opportunities", []),
            "risks": insights_json.get("risks", []),
            "recommendations": insights_json.get("recommendations", []),
            "recommendations_structured": [],
            "evidence_catalog": evidence_catalog,
        }
        ch_prompt = s_prompts.get_challenger_critique_prompt(challenger_input, client_name=aggregated.get("client_name"))
        ch_raw = fetch_response(ch_prompt, model="gpt-4o", temperature=0.2, max_tokens=1200)
        if ch_raw:
            t = ch_raw.strip()
            if t.startswith("```json"):
                t = t[len("```json"):].strip();
                if t.endswith("```"):
                    t = t[:-3].strip()
            refined = json.loads(t)
    except Exception:
        refined = {}

    # 2) Generar todos los textos especialistas para Parte 2 usando el JSON de insights
    strategic_sections = _generate_full_part2_texts(insights_json, aggregated)
    agent_summary_text = ""
    try:
        agent_prompt = s_prompts.get_agent_insights_summary_prompt({"agent_insights": agent_insights})
        agent_summary_text = fetch_response(agent_prompt, model="gpt-4o-mini", temperature=0.3, max_tokens=700)
    except Exception:
        agent_summary_text = ""

    # Gráficos alineados a la API: sentimiento como % positivo y visibilidad diaria
    try:
        sent_img = plotter.plot_sentiment_evolution(sent_evo)
    except Exception:
        sent_img = None
    try:
        vis_img = plotter.plot_line_series(vis_dates, vis_vals, title="Puntuación de visibilidad", ylabel="Visibilidad (%)", ylim=(0, 100), color="#000000")
    except Exception:
        vis_img = None

    # Volumen de menciones (para Tendencias)
    try:
        mv_series = aggregator.get_mentions_volume_series(None, project_id, start_date=start_date, end_date=end_date)
        mv_dates = [d for d, _ in mv_series]
        mv_counts = [int(c) for _, c in mv_series]
        mentions_vol_img = plotter.plot_mentions_volume(mv_dates, mv_counts)
    except Exception:
        mentions_vol_img = None

    images = {
        "sentiment_evolution": sent_img,
        "part1_visibility_line": vis_img,
        "sov_pie": plotter.plot_sov_pie([(name, val) for name, val in sov_pairs[:10]]),
        # Anexos
        "sentiment_by_category": plotter.plot_sentiment_by_category(by_cat),
        "topics_top_bottom": plotter.plot_topics_top_bottom(top5, bottom5),
        "mentions_volume": mentions_vol_img,
    }

    # SOV por categoría (cliente) y delta vs periodo previo — reutiliza bundle ya calculado
    try:
        sov_curr = (aggregated.get("sov", {}).get("current", {}) if isinstance(aggregated.get("sov"), dict) else {}) or {}
        cat_map = sov_curr.get("sov_by_category", {}) or {}
        images["sov_by_category"] = plotter.plot_sov_by_category(cat_map)
    except Exception:
        images["sov_by_category"] = None
    try:
        delta_map = ((aggregated.get("sov", {}).get("trends", {}) if isinstance(aggregated.get("sov"), dict) else {}) or {}).get("sov_delta_by_category")
        images["sov_by_category_delta"] = plotter.plot_sov_delta_by_category(delta_map) if isinstance(delta_map, dict) and delta_map else None
    except Exception:
        images["sov_by_category_delta"] = None

    # Imágenes adicionales para correlaciones transversales
    try:
        corr_bundle = aggregated.get("correlations", {}) or {}
        from . import plotter as _pl
        # Dispersión SOV vs Sentimiento por categoría
        quad = corr_bundle.get("category_quadrants", {}) if isinstance(corr_bundle, dict) else {}
        rows = quad.get("rows") or []
        images["sov_sent_scatter"] = _pl.plot_category_sov_vs_sentiment_scatter(rows, quad.get("thresholds", {}))
    except Exception:
        images["sov_sent_scatter"] = None
    try:
        corr_mat = (aggregated.get("correlations", {}) or {}).get("matrix", {})
        labels = corr_mat.get("labels", []) if isinstance(corr_mat, dict) else []
        values = corr_mat.get("values", []) if isinstance(corr_mat, dict) else []
        from . import plotter as _pl
        images["corr_heatmap"] = _pl.plot_correlation_heatmap(labels, values)
    except Exception:
        images["corr_heatmap"] = None
    try:
        lags = ((aggregated.get("correlations", {}) or {}).get("lags_volume_to_sentiment", {}) or {}).get("series", [])
        from . import plotter as _pl
        images["lag_corr"] = _pl.plot_lag_correlation(lags)
    except Exception:
        images["lag_corr"] = None

    # Nueva: Wordcloud cualitativa reciente (corpus global)
    try:
        corpus = aggregator.get_all_mentions_for_period(limit=120, project_id=project_id, client_brand=brand_name, start_date=start_date, end_date=end_date)
        images["wordcloud"] = plotter.plot_wordcloud_from_corpus(corpus)
    except Exception:
        images["wordcloud"] = None

    # Preparar tabla de plan a partir de recomendaciones estructuradas refinadas
    def _normalize_plan_item(it: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "action": it.get("action") or it.get("recommendation") or "",
            "impact": it.get("impact"),
            "effort": it.get("effort"),
            "owner": it.get("owner"),
            "due": it.get("due") or it.get("timeline"),
            "probability": it.get("probability"),
            "confidence": it.get("confidence"),
        }

    structured_recs = []
    try:
        structured_recs = [
            _normalize_plan_item(x) for x in (refined.get("recommendations_refined") or insights_json.get("recommendations_structured") or [])
            if isinstance(x, dict)
        ]
    except Exception:
        structured_recs = []

    # Filtrar campañas de marketing/ads y reforzar con 1-2 acciones operativas si faltan
    try:
        banned = ["campaña", "campana", "marketing", "ads", "sem", "publicidad", "anuncio", "paid", "ppc"]
        def _is_banned(action_text: str | None) -> bool:
            s = (action_text or "").lower()
            return any(k in s for k in banned)
        filtered: list[dict[str, Any]] = []
        for it in structured_recs:
            act = it.get("action") or it.get("recommendation")
            if _is_banned(str(act)):
                continue
            filtered.append(it)
        structured_recs = filtered
        # Añadir acciones operativas si hay menos de 2
        def _append_if_missing(item: dict[str, Any]) -> None:
            existing = [str(x.get("action") or "").strip().lower() for x in structured_recs]
            key = str(item.get("action") or "").strip().lower()
            if key and key not in existing:
                structured_recs.append(item)
        if len(structured_recs) < 3:
            # Programa de Salidas Profesionales — accciones por fases (no marketing)
            fase1 = {
                "action": "Fase 1 (0-30 días): Diseñar y publicar el 'Programa de Salidas Profesionales' (landing, calendario, responsables, FAQs)",
                "metric": "Landing publicada y ≥4 talleres programados",
                "owner": "Empleabilidad",
                "due": "30 días",
                "impact": "Alto",
                "effort": "Medio",
                "probability": "Alta",
                "confidence": "Media",
            }
            fase2 = {
                "action": "Fase 2 (30-60 días): Ejecutar 4 talleres/mes (CV, portfolio, entrevistas, LinkedIn) y lanzar mentoría alumni",
                "metric": "Nº talleres/mes y Nº alumnos en mentoría",
                "owner": "Empleabilidad",
                "due": "60 días",
                "impact": "Alto",
                "effort": "Medio",
                "probability": "Alta",
                "confidence": "Media",
            }
            fase3 = {
                "action": "Fase 3 (60-90 días): Firmar ≥5 convenios de prácticas remuneradas y activar bolsa de empleo (≥10 ofertas/mes)",
                "metric": "Nº convenios firmados y ofertas/mes",
                "owner": "Relaciones Institucionales",
                "due": "90 días",
                "impact": "Alto",
                "effort": "Medio",
                "probability": "Alta",
                "confidence": "Media",
            }
            _append_if_missing(fase1)
            _append_if_missing(fase2)
            _append_if_missing(fase3)
    except Exception:
        pass

    content_for_pdf: Dict[str, Any] = {
        "strategic": strategic_sections,
        "kpi_rows": _build_kpi_rows(aggregated["kpis"]),
        "images": images,
        "agent_insights": {
            "summary": agent_summary_text,
            "buckets": agent_insights.get("buckets", {}),
        },
        # Nuevo: Clusters y Síntesis Estratégica
        "clusters": cluster_summaries,
        "clusters_synthesis": synthesis,
        # Pasamos deep_dives si el agregador los incluyó (backend app.py v1)
        "deep_dives": aggregated.get("deep_dives", []),
        # Nueva tabla de plan y anexo de evidencias
        "action_plan_table": structured_recs,
        "evidence_annex": evidence_index,
        "annex": {
            "evolution_text": "",
            "category_text": "",
            "topics_text": "",
        },
    }

    # --- NUEVO: Fichas por competidor (deep dive real) ---
    competitor_profiles: list[dict[str, Any]] = []
    try:
        # Limitar a 6 para control de longitud
        for comp in (competitors_list or [])[:6]:
            try:
                # KPIs básicos del competidor
                comp_kpis = aggregator.get_kpi_summary(None, project_id, start_date=start_date, end_date=end_date, client_brand=comp)
            except Exception:
                comp_kpis = {"total_mentions": 0, "sentiment_avg": 0.0, "sov": 0.0, "brand_name": comp}

            # SOV por categoría del competidor y menciones por categoría
            try:
                sov_comp = aggregator.get_share_of_voice_and_trends(None, project_id, start_date=start_date or "1970-01-01", end_date=end_date or "2999-12-31", client_brand=comp)
                sov_by_cat_comp = (sov_comp.get("current", {}) or {}).get("sov_by_category", {})
                sov_total_comp = float(((sov_comp.get("current", {}) or {}).get("share_of_voice", 0.0)) or 0.0)
            except Exception:
                sov_by_cat_comp = {}
                sov_total_comp = 0.0

            # Series de sentimiento y visibilidad del competidor
            try:
                sent_comp = aggregator.get_sentiment_evolution(None, project_id, start_date=start_date, end_date=end_date, client_brand=comp)
            except Exception:
                sent_comp = []
            try:
                vis_dates_c, vis_vals_c = aggregator.get_visibility_series(None, project_id, start_date=start_date, end_date=end_date, client_brand=comp)
            except Exception:
                vis_dates_c, vis_vals_c = [], []

            # Top/Bottom temas del competidor
            try:
                top_c, bottom_c = aggregator.get_topics_by_sentiment(None, project_id, start_date=start_date, end_date=end_date, client_brand=comp)
            except Exception:
                top_c, bottom_c = [], []

            # Evidencias: corpus de menciones del competidor (muestras)
            try:
                comp_corpus = aggregator.get_all_mentions_for_period(limit=60, project_id=project_id, client_brand=comp, start_date=start_date, end_date=end_date)
            except Exception:
                comp_corpus = []

            # Narrativa focalizada: desde la perspectiva de la marca cliente analizando al competidor {comp}
            comp_narrative = ""
            try:
                # Calcular visibilidad media del competidor aquí para incluirla en el prompt
                try:
                    vis_avg_c_prompt = float((sum(vis_vals_c) / max(len(vis_vals_c), 1)) if vis_vals_c else 0.0)
                except Exception:
                    vis_avg_c_prompt = 0.0
                comp_agg = {
                    # KPIs del cliente (marca principal del informe)
                    "kpis": aggregated.get("kpis", {}),
                    # Perspectiva: la marca cliente analiza al competidor focal
                    "client_name": str(brand_name),
                    "focus_competitor": str(comp),
                    "competitor_kpis": {
                        "total_mentions": int(comp_kpis.get("total_mentions", 0)),
                        "sentiment_avg": float(comp_kpis.get("sentiment_avg", 0.0) or 0.0),
                        # SOV robusto basado en detección extendida (no solo q.brand)
                        "sov": float(sov_total_comp),
                        "visibility_avg": float(vis_avg_c_prompt),
                    },
                    "sov_by_category_focus": sov_by_cat_comp,
                    # Lista de adversarios vista desde la marca cliente
                    "market_competitors": [b for b in competitors_list if b != comp] + [brand_name],
                }
                prompt_comp = s_prompts.get_competitive_analysis_prompt(comp_agg)
                comp_narrative = fetch_response(prompt_comp, model="gpt-4o-mini", temperature=0.3, max_tokens=700) or ""
            except Exception:
                comp_narrative = ""

            # Flancos accionables (2) con cita textual del corpus si es posible
            comp_flanks: list[str] = []
            try:
                import json as _json
                data_for_flanks = {
                    "brand": comp,
                    "top_topics": top_c,
                    "bottom_topics": bottom_c,
                    "sov_by_category": sov_by_cat_comp,
                    "sample_corpus": comp_corpus[:8],
                }
                flank_prompt = (
                    "Eres un Analista de Inteligencia Competitiva. Con los datos, identifica 2 'flancos' explotables del competidor y propone una acción concreta para nuestra marca. "
                    "Si puedes, incluye una cita breve entre comillas del corpus para evidenciar el flanco. Devuelve 2 bullets, claros y específicos.\n\n"
                    f"DATOS:\n```json\n{_json.dumps(data_for_flanks, ensure_ascii=False)}\n```"
                )
                flank_text = fetch_response(flank_prompt, model="gpt-4o-mini", temperature=0.2, max_tokens=320) or ""
                comp_flanks = [ln.strip() for ln in flank_text.splitlines() if ln.strip()][:4]
            except Exception:
                comp_flanks = []

            # Imágenes por competidor
            try:
                img_sent_c = plotter.plot_sentiment_evolution(sent_comp)
            except Exception:
                img_sent_c = None
            try:
                img_vis_c = plotter.plot_line_series(vis_dates_c, vis_vals_c, title="Visibilidad (%)", ylabel="%", ylim=(0,100), color="#0ea5e9")
            except Exception:
                img_vis_c = None

            # KPIs tabla simple
            try:
                vis_avg_c = float((sum(vis_vals_c) / max(len(vis_vals_c), 1)) if vis_vals_c else 0.0)
            except Exception:
                vis_avg_c = 0.0

            competitor_profiles.append({
                "name": comp,
                "kpis": {
                    "total_mentions": int(comp_kpis.get("total_mentions", 0)),
                    "sentiment_avg": float(comp_kpis.get("sentiment_avg", 0.0) or 0.0),
                    "sov": float(comp_kpis.get("sov", 0.0) or 0.0),
                    "visibility_avg": vis_avg_c,
                },
                "top_topics": top_c,
                "bottom_topics": bottom_c,
                "narrative": comp_narrative.strip(),
                "flanks": comp_flanks[:2],
                "images": {
                    "sentiment_line": img_sent_c,
                    "visibility_line": img_vis_c,
                },
            })
    except Exception:
        competitor_profiles = []

    if competitor_profiles:
        content_for_pdf["competitor_profiles"] = competitor_profiles

    try:
        if sent_evo:
            first_s = sent_evo[0][1]
            last_s = sent_evo[-1][1]
            delta = last_s - first_s
            trend_word = "mejora" if delta > 0.05 else ("empeora" if delta < -0.05 else "se mantiene estable")
            min_day, min_val = min(sent_evo, key=lambda x: x[1])
            max_day, max_val = max(sent_evo, key=lambda x: x[1])
            content_for_pdf["annex"]["evolution_text"] = (
                f"El sentimiento {trend_word} (Δ={delta:.2f}). Mínimo en {min_day} ({min_val:.2f}) y máximo en {max_day} ({max_val:.2f})."
            )
    except Exception:
        pass
    try:
        if by_cat:
            best_cat, best_v = max(by_cat.items(), key=lambda x: x[1])
            worst_cat, worst_v = min(by_cat.items(), key=lambda x: x[1])
            content_for_pdf["annex"]["category_text"] = (
                f"Mejor categoría: {best_cat} ({best_v:.2f}). Peor: {worst_cat} ({worst_v:.2f})."
            )
    except Exception:
        pass
    try:
        if top5 or bottom5:
            top_str = ", ".join([f"{t} ({v:.2f})" for t, v in top5])
            bot_str = ", ".join([f"{t} ({v:.2f})" for t, v in bottom5])
            content_for_pdf["annex"]["topics_text"] = (
                f"Top temas: {top_str}. Bottom: {bot_str}."
            )
    except Exception:
        pass

    # Bundle unificado con imágenes + textos
    content_bundle: Dict[str, Any] = {
        "company_name": brand_name or aggregated.get("client_name") or "Empresa",
        "images": images,
        "strategic": {
            "executive_summary": strategic_sections.get("executive_summary", ""),
            "summary_and_findings": strategic_sections.get("summary_and_findings", ""),
            # Filtrar secciones banales
            "competitive_analysis": ("" if "incrementar el presupuesto" in (strategic_sections.get("competitive_analysis", "").lower()) else strategic_sections.get("competitive_analysis", "")),
            "trends": strategic_sections.get("trends", ""),
            "correlations": strategic_sections.get("correlations", ""),
            "action_plan": ("" if "incrementar el presupuesto" in (strategic_sections.get("action_plan", "").lower()) else strategic_sections.get("action_plan", "")),
            # Metadatos para ficha técnica y captions
            "kpis": aggregated.get("kpis", {}),
            "period": {"start": start_date, "end": end_date},
            "competitors": competitors_list,
            "ai_engines": (ai_engines_used if ai_engines_used else ["OpenAI GPT-4o"]),
            "scope": "Nacional",
            # Deltas y tabla de plan para dashboard
            "deltas": {
                "sov": float(sov_delta) if sov_delta is not None else None,
                "sentiment": float(sent_delta) if sent_delta is not None else None,
                "visibility": float(vis_delta) if vis_delta is not None else None,
            },
            "action_plan_table": structured_recs,
            "evidence_annex": evidence_index,
            "competitor_profiles": competitor_profiles,
        },
    }

    # --- NUEVO: análisis por categoría (Parte 3) ---
    try:
        def _norm(s: str) -> str:
            s = (s or "").lower().strip()
            repl = {
                "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n",
                "&": "and",
            }
            for k, v in repl.items():
                s = s.replace(k, v)
            return " ".join(s.split())

        categories_es = [
            "Audiencia e Investigación",
            "Marca y Reputación",
            "Competencia y Benchmarking",
            "Plan de estudios y Programas",
            "Tendencias Digitales y Marketing",
            "Empleo y Profesiones",
            "Motivaciones y Disparadores",
            "Padres y Preocupaciones",
            "Becas y Coste",
            "Share of Voice y Monitorización",
        ]
        # Sinónimos en inglés para matching flexible
        synonyms: dict[str, list[str]] = {
            "Audiencia e Investigación": ["audience & research", "audience and research", "audience"],
            "Marca y Reputación": ["brand & reputation", "brand and reputation", "reputation"],
            "Competencia y Benchmarking": ["competition & benchmarking", "competition and benchmarking", "benchmarking"],
            "Plan de estudios y Programas": ["curriculum & programs", "curriculum and programs", "programs"],
            "Tendencias Digitales y Marketing": ["digital trends & marketing", "digital trends and marketing", "marketing"],
            "Empleo y Profesiones": ["employment & jobs", "employment and jobs", "jobs"],
            "Motivaciones y Disparadores": ["motivation & triggers", "motivation and triggers", "triggers"],
            "Padres y Preocupaciones": ["parents & family concerns", "parents and family concerns", "parents"],
            "Becas y Coste": ["scholarships & cost", "scholarships and cost", "scholarships"],
            "Share of Voice y Monitorización": ["share of voice & monitoring", "share of voice and monitoring", "share of voice"],
        }

        buckets = agent_insights.get("buckets", {}) or {}

        category_texts: dict[str, str] = {}
        for cat in categories_es:
            keys = [_norm(cat)] + [_norm(x) for x in synonyms.get(cat, [])]
            def _item_matches(it: dict) -> bool:
                fields = []
                for k in ("topic", "category", "section", "label", "area"):
                    v = it.get(k)
                    if isinstance(v, str):
                        fields.append(v)
                tags = it.get("tags") or []
                if isinstance(tags, list):
                    fields.extend([str(t) for t in tags])
                text = it.get("text") or it.get("opportunity") or it.get("risk") or it.get("trend") or ""
                fields.append(text)
                joined = _norm(" ".join([str(f) for f in fields if isinstance(f, str)]))
                return any(k in joined for k in keys)

            ops = [it for it in (buckets.get("opportunities") or []) if isinstance(it, dict) and _item_matches(it)][:3]
            rks = [it for it in (buckets.get("risks") or []) if isinstance(it, dict) and _item_matches(it)][:3]
            trn = [it for it in (buckets.get("trends") or []) if isinstance(it, dict) and _item_matches(it)][:3]

            lines: list[str] = []
            try:
                sov_curr = (aggregated.get("sov") or {}).get("current") or {}
                entry = None
                for k2, v2 in (sov_curr.get("sov_by_category", {}).items() if isinstance(sov_curr.get("sov_by_category", {}), dict) else []):
                    if _norm(str(k2)) in keys or any(_norm(str(k2)) in _norm(kx) for kx in [cat] + synonyms.get(cat, [])):
                        entry = v2; break
                if isinstance(entry, dict):
                    client = float(entry.get("client", 0)); total = float(entry.get("total", 0))
                    pct = (client / max(total, 1.0)) * 100.0
                    lines.append(f"Visibilidad y cuota de conversación (SOV) en la categoría: {pct:.1f}% sobre su total.")
            except Exception:
                pass

            def _fmt(it: dict) -> str:
                return str(it.get("text") or it.get("opportunity") or it.get("risk") or it.get("trend") or "-")

            if ops:
                lines.append("Oportunidades relevantes:")
                lines.extend(["- " + _fmt(x) for x in ops])
            if rks:
                lines.append("Riesgos a vigilar:")
                lines.extend(["- " + _fmt(x) for x in rks])
            if trn:
                lines.append("Tendencias y señales del tema:")
                lines.extend(["- " + _fmt(x) for x in trn])

            # CTA simple al final si existe alguna recomendación en recomendaciones
            try:
                recs = insights_json.get("recommendations") or []
                if isinstance(recs, list) and recs:
                    lines.append("CTA sugerida:")
                    lines.append("- " + str(recs[0]))
            except Exception:
                pass

            # Añadir sentimiento al final, en menor prioridad
            try:
                s_val = by_cat.get(cat) if isinstance(by_cat, dict) else None
                if s_val is None:
                    for k2, v2 in (by_cat.items() if isinstance(by_cat, dict) else []):
                        if _norm(str(k2)) in keys or any(_norm(str(k2)) in _norm(kx) for kx in [cat] + synonyms.get(cat, [])):
                            s_val = v2; break
                if s_val is not None:
                    lines.append(f"Sentimiento promedio (referencia): {float(s_val):.2f}.")
            except Exception:
                pass

            text_block = "\n".join(lines).strip()

            if not text_block:
                # Fallback: usar un prompt breve para generar 1 párrafo por categoría
                try:
                    dd_prompt = s_prompts.get_deep_dive_analysis_prompt(cat, aggregated.get("kpis", {}), brand_name)
                    text_block = fetch_response(dd_prompt, model="gpt-4o-mini", temperature=0.3, max_tokens=300) or ""
                except Exception:
                    text_block = ""

            if text_block:
                category_texts[cat] = text_block

        if category_texts:
            content_bundle["strategic"]["category_analyses"] = category_texts
    except Exception:
        pass

    # Render final del esqueleto con contenido
    return pdf_writer.build_skeleton_from_content(content_bundle)


def generate_hybrid_report(full_data: Dict[str, Any]) -> bytes:
    """Rellena el esqueleto con 3 páginas en dos columnas: SOV, Sentimiento, Visibilidad (últimos 30 días)."""
    from .pdf_writer import build_skeleton_with_content as build
    from . import plotter
    from . import aggregator as agg
    kpis = full_data.get("kpis", {})
    brand_name = kpis.get("brand_name") or full_data.get("brand") or "Empresa"

    # Periodo por defecto: últimos 30 días
    start_date = full_data.get("start_date") or None
    end_date = full_data.get("end_date") or None

    # 1) SOV global (pie) y ranking (lista)
    session = agg.get_session();
    try:
        # Usar el project_id del bundle para limitar el ranking al mercado seleccionado
        pid = int(full_data.get("project_id") or 1)
        sov_pairs = agg.get_industry_sov_ranking(session, project_id=pid, start_date=start_date, end_date=end_date)
        sov_img = plotter.plot_sov_pie([(name, val) for name, val in sov_pairs[:10]])
        # Render ranking como tabla simple en imagen: reutilizamos barh con porcentajes
        try:
            # Renderizar ranking como imagen de lista simple
            labels = [f"{i+1}. {n} — {v:.1f}%" for i, (n, v) in enumerate(sov_pairs[:10])]
            import matplotlib.pyplot as plt
            h = max(1.6, 0.32 * len(labels) + 0.6)
            plt.figure(figsize=(4.2, h))
            for i, txt in enumerate(labels):
                plt.text(0.01, 1.0 - (i+1)/(len(labels)+1), txt, fontsize=9)
            plt.axis('off')
            from .plotter import _tmp_path
            sov_rank_img = _tmp_path("sov_rank_"); plt.tight_layout(); plt.savefig(sov_rank_img, dpi=160, bbox_inches='tight', pad_inches=0.1); plt.close()
        except Exception:
            sov_rank_img = None

        # 2) Sentimiento positivo por día (serie) en una sola columna (sin gráfico derecho)
        sent_series = agg.get_sentiment_positive_series(session, int(full_data.get("project_id") or 1), start_date=start_date, end_date=end_date, client_brand=brand_name)
        sent_img = plotter.plot_line_series([d for d, _ in sent_series], [float(v) for _, v in sent_series], title="% de menciones positivas", ylabel="Positivo (%)", ylim=(0,100), color="#16a34a")
        sent_dist_img = None

        # 3) Visibilidad por día y ranking
        vis_dates, vis_vals = agg.get_visibility_series(session, int(full_data.get("project_id") or 1), start_date=start_date, end_date=end_date, client_brand=brand_name)
        try:
            vis_line_img = plotter.plot_visibility_series(vis_dates, vis_vals)
        except Exception:
            vis_line_img = None
        vis_rank_pairs = agg.get_visibility_ranking(session, project_id=pid, start_date=start_date, end_date=end_date)
        try:
            items = [f"{i+1}. {n} — {v:.1f}%" for i, (n, v) in enumerate(vis_rank_pairs[:10])]
            import matplotlib.pyplot as plt
            h = max(1.6, 0.32 * len(items) + 0.6)
            plt.figure(figsize=(4.2, h))
            for i, txt in enumerate(items):
                plt.text(0.01, 1.0 - (i+1)/(len(items)+1), txt, fontsize=9)
            plt.axis('off')
            from .plotter import _tmp_path
            vis_rank_img = _tmp_path("vis_rank_"); plt.tight_layout(); plt.savefig(vis_rank_img, dpi=160, bbox_inches='tight', pad_inches=0.1); plt.close()
        except Exception:
            vis_rank_img = None
    finally:
        session.close()

    images = {
        "sov_pie": sov_img,
        "sov_ranking_table": sov_rank_img,
        "sentiment_evolution": sent_img,
        "sentiment_distribution": sent_dist_img,
        "visibility_line": vis_line_img,
        "visibility_ranking_table": vis_rank_img,
    }

    return build(brand_name, images)
    evo = full_data.get("time_series", {}).get("sentiment_per_day", [])
    sov = full_data.get("sov", {})
    clusters = full_data.get("clusters", [])

    # Preparar análisis de clusters (Nivel 1 y 2)
    cluster_summaries: List[Dict[str, Any]] = []
    for c in (clusters or [])[:12]:
        cluster_obj = {
            "count": int(c.get("count", 0)),
            "avg_sentiment": float(c.get("avg_sentiment", 0.0)),
            "top_sources": c.get("top_sources", []),
            "example_mentions": c.get("example_mentions", []),
        }
        analyzed = _analyze_cluster(cluster_obj)
        cluster_summaries.append({
            "topic_name": analyzed.get("topic_name", "(sin nombre)"),
            "key_points": analyzed.get("key_points", []),
            "volume": int(cluster_obj["count"]),
            "sentiment": float(cluster_obj["avg_sentiment"]),
            "examples": cluster_obj["example_mentions"][:3],
        })

    synthesis = _synthesize_clusters(cluster_summaries)

    # Construir imágenes usando plotter
    from . import plotter
    images = {}
    try:
        dates = [d for d, _ in evo]
        values = [float(v) for _, v in evo]
        images["sentiment_evolution"] = plotter.plot_sentiment_evolution(evo)
    except Exception:
        images["sentiment_evolution"] = None
    try:
        sov_list = [(name, cnt) for name, cnt in (kpis.get("sov_table", [])[:6] if kpis.get("sov_table") else [])]
        images["sov_pie"] = plotter.plot_sov_pie(sov_list)
    except Exception:
        images["sov_pie"] = None
    try:
        top5 = full_data.get("topics_top5") or []
        bottom5 = full_data.get("topics_bottom5") or []
        images["topics_top_bottom"] = plotter.plot_topics_top_bottom(top5, bottom5)
    except Exception:
        images["topics_top_bottom"] = None

    # Preparar bloques de texto estratégico a partir de síntesis
    strategic_sections = {
        "executive_summary": "",
        "action_plan": "",
        "competitive_analysis": "",
        "trends": "",
        "headline": "",
        "key_findings": [],
    }
    try:
        # Usamos síntesis para Executive y Plan si están disponibles
        if synthesis:
            exec_prompt = s_prompts.get_strategic_summary_prompt({
                "executive_summary": "" ,
                "key_findings": [kp for c in cluster_summaries for kp in c.get("key_points", [])][:6],
            })
            strategic_sections["executive_summary"] = fetch_response(exec_prompt, model="gpt-4o", temperature=0.3, max_tokens=700)
            plan_prompt = s_prompts.get_strategic_plan_prompt({
                "opportunities": [],
                "risks": [],
                "recommendations": synthesis.get("plan_estrategico", []),
            })
            strategic_sections["action_plan"] = fetch_response(plan_prompt, model="gpt-4o", temperature=0.3, max_tokens=900)
            # Headline + key findings simples a partir de puntos de clusters
            try:
                strategic_sections["key_findings"] = [kp for c in cluster_summaries for kp in c.get("key_points", [])][:4]
                if strategic_sections["executive_summary"]:
                    strategic_sections["headline"] = strategic_sections["executive_summary"].split(".")[0][:140]
            except Exception:
                pass
    except Exception:
        pass

    # Armar contenido para PDF híbrido
    kpi_rows = _build_kpi_rows(kpis)
    content_for_pdf: Dict[str, Any] = {
        "strategic": strategic_sections,
        "kpi_rows": kpi_rows,
        "images": images,
        "clusters": cluster_summaries,
        "clusters_synthesis": synthesis,
        "competitive_opportunities": full_data.get("competitive_opportunities", []),
        "kpis": kpis,
    }

    # Parte 1: Dashboard Ejecutivo con 3 KPIs globales
    try:
        # SOV global (donut/pie). Si no hay competencia, omitir donut y usar pie global
        sov_table = (kpis.get("sov_table") or [])
        try:
            if len(sov_table) >= 2:
                images["part1_sov_donut"] = plotter.plot_sov_pie([(b, int(c)) for b, c in sov_table[:8]])
            else:
                images["part1_sov_donut"] = None
        except Exception:
            images["part1_sov_donut"] = None
    except Exception:
        # Fallback final al SOV pie estándar si ya fue calculado más arriba
        images["part1_sov_donut"] = images.get("sov_pie")

    try:
        # Sentimiento global (línea)
        evo = full_data.get("time_series", {}).get("sentiment_per_day", [])
        dates = [d for d, _ in evo]
        vals = [float(v) for _, v in evo]
        images["part1_sentiment_line"] = plotter.plot_sentiment_evolution(evo)
    except Exception:
        images["part1_sentiment_line"] = images.get("sentiment_evolution")

    try:
        # Visibilidad global (línea). Usamos la serie calculada en aggregator
        vis_series = full_data.get("visibility_timeseries") or []
        v_dates = [d for d, _ in vis_series]
        v_vals = [float(v) for _, v in vis_series]
        try:
            # Reutilizar plot_mentions_volume para una línea simple de % visibilidad
            images["part1_visibility_line"] = plotter.plot_line_series(v_dates, v_vals, title="Puntuación de visibilidad", ylabel="Visibilidad (%)", ylim=(0, 100), color="#000000")  # type: ignore[attr-defined]
        except Exception:
            # Fallback simple usando lineplot existente (si estuviera)
            from . import plotter as _pl
            try:
                images["part1_visibility_line"] = _pl.plot_mentions_volume(v_dates, [int(round(x)) for x in v_vals])
            except Exception:
                images["part1_visibility_line"] = None
    except Exception:
        images["part1_visibility_ranking"] = None

    try:
        # Distribución de menciones por categoría (porcentaje sobre total)
        curr = full_data.get("sov", {}).get("current", {})
        cat_map = curr.get("sov_by_category", {})
        dist = {k: int(v.get("total", 0)) for k, v in cat_map.items()}
        try:
            images["part1_category_distribution"] = plotter.plot_category_distribution_donut(dist, title="Distribución de menciones por categoría")  # type: ignore[attr-defined]
        except Exception:
            # Fallback: usar gráfico de sentimiento por categoría si hay datos
            sbc = full_data.get("sentiment_by_category") or {}
            images["part1_category_distribution"] = plotter.plot_sentiment_by_category(sbc)
    except Exception:
        images["part1_category_distribution"] = None

    # Renderizar PDF usando el ESQUELETO como base y añadiendo Parte 1
    try:
        from .pdf_writer import build_skeleton_with_content as _build_skeleton
    except ImportError:
        # Fallback por si el nombre cambia o el módulo aún no exporta
        from . import pdf_writer as _pw  # type: ignore
        _build_skeleton = getattr(_pw, "build_skeleton_with_content", None)
        if _build_skeleton is None:
            # Fallback final: usar la estructura vacía para no romper
            def _build_skeleton(company_name, imgs):
                return _pw.build_empty_structure_pdf(company_name)
    company = kpis.get("brand_name") or full_data.get("brand") or "Empresa"
    pdf_bytes = _build_skeleton(company, images)
    # Insertar tabla de oportunidades si existe
    try:
        if content_for_pdf.get("competitive_opportunities"):
            from .pdf_writer import ReportPDF
            # Reconstruimos el PDF para añadir la tabla (fpdf no permite abrir PDF existente fácilmente)
            # Alternativa: devolvemos el PDF base y dejamos la tabla en página adicional desde build_pdf.
            # Simpler: devolvemos el base; la tabla se renderiza en build_pdf si quisieras integrarlo ahí.
            pass
    except Exception:
        pass
    return pdf_bytes
