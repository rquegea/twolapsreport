import json


def get_detailed_category_prompt(category_name: str, category_data: list, all_summaries: list, kpis: dict) -> str:
    data_json = json.dumps(category_data, indent=2, ensure_ascii=False, default=str)
    summaries_text = "\n- ".join(list(set(all_summaries)))

    _sov_cat = kpis.get('sov_by_category', {}).get(category_name, {"client": 0, "total": 0})
    _client = float(_sov_cat.get('client', 0))
    _total = float(_sov_cat.get('total', 0))
    _sov_pct = (_client / _total * 100.0) if _total > 0 else 0.0

    prompt = f"""
    **ROL:** Eres un Analista de Inteligencia de Mercado Senior.
    **TAREA:** Redacta un resumen ejecutivo para la sección de un informe. Interpreta los datos, no te limites a listarlos. **Integra las métricas clave (KPIs) de forma natural en tu análisis** para dar un contexto cuantitativo.
    
    **SECCIÓN A ANALIZAR:** "{category_name}"

    **KPIs GENERALES DEL PERIODO:**
    - Menciones totales analizadas: {kpis.get('total_mentions', 'N/A')}
    - Sentimiento general promedio: {kpis.get('average_sentiment', 0.0):.2f} (en una escala de -1 a 1)
    - SOV de esta sección (cliente / total en %): {_sov_pct:.2f}%

    **DATOS DETALLADOS DE ESTA SECCIÓN:**
    ```json
    {data_json}
    ```

    **RESUMEN GLOBAL DE TODOS LOS TEMAS DETECTADOS (para contexto):**
    - {summaries_text}

    **INSTRUCCIÓN:**
    Basándote en TODOS los datos proporcionados, escribe un párrafo de análisis de entre 120 y 180 palabras para la sección "{category_name}". Cubre: contexto de mercado, competencia (incluye SOV si aplica) y sentimiento. Compara el sentimiento de esta sección con el promedio general cuando aporte valor. Termina con 1 recomendación práctica.
    """
    return prompt


def get_main_analyst_prompt(aggregated_data: dict, global_mentions_corpus: list[str]) -> str:
    """
    Analista Principal al estilo twolaps: integra KPIs + corpus global de menciones
    para producir un ÚNICO JSON con todos los textos del informe y selección de temas.

    Salida esperada (JSON estricto, sin markdown):
    {
      "headline": "...",
      "evaluacion_general": "...",
      "analisis_profundo": "...",
      "analisis_competencia": "...",
      "analisis_mercado": "...",
      "cualitativo_global": {
        "sintesis_del_hallazgo": "...",
        "causa_raiz": "...",
        "citas_destacadas": ["...", "..."]
      },
      "deep_dive_temas": ["tema 1", "tema 2", "tema 3"]
    }
    """
    data_json = json.dumps(aggregated_data, ensure_ascii=False, indent=2, default=str)
    corpus = "\n\n".join([str(m)[:4000] for m in (global_mentions_corpus or []) if isinstance(m, str)])
    return (
        "Actúa como Analista Principal (Chief Insights Analyst) híbrido, cuantitativo y cualitativo. "
        "Tu misión es sintetizar KPIs, tendencias y el corpus literal de menciones para redactar el Dossier Ejecutivo.\n\n"
        "INSTRUCCIONES CLAVE:\n"
        "1) Integra KPIs (series, SOV, sentimiento), ranking de competidores y temas clave.\n"
        "2) Usa el CORPUS GLOBAL para validar hallazgos y extraer citas representativas (textuales).\n"
        "3) Entrega textos listos para el informe: titular, evaluación general, análisis profundo (con correlaciones),\n"
        "   análisis de competencia y análisis de mercado.\n"
        "4) Devuelve también un bloque cualitativo global (síntesis, causa raíz y 3-6 citas).\n"
        "5) Selecciona 2-3 temas críticos para posibles deep dives.\n"
        "6) Responde EXCLUSIVAMENTE con el JSON pedido. SIN markdown, SIN comentarios, SIN texto adicional.\n\n"
        "DATOS CUANTITATIVOS (JSON):\n" + data_json + "\n\n"
        "CORPUS GLOBAL DE MENCIONES (texto literal, truncado si es largo):\n" + corpus + "\n\n"
        "FORMATO DE SALIDA (JSON ESTRICTO):\n"
        "{\n"
        "  \"headline\": \"Titular conciso que conecte dato + porqué\",\n"
        "  \"evaluacion_general\": \"Conclusión ejecutiva del periodo y drivers principales.\",\n"
        "  \"analisis_profundo\": \"2-4 párrafos conectando KPIs, temas y competencia; incluye correlaciones.\",\n"
        "  \"analisis_competencia\": \"Lectura de SOV y movimientos competitivos relevantes.\",\n"
        "  \"analisis_mercado\": \"Lectura de temas clave y señales de mercado (oportunidades/amenazas).\",\n"
        "  \"cualitativo_global\": {\n"
        "    \"sintesis_del_hallazgo\": \"3-5 frases que sinteticen el insight global basado en menciones.\",\n"
        "    \"causa_raiz\": \"Hipótesis de causa raíz con soporte del corpus.\",\n"
        "    \"citas_destacadas\": [\"cita 1\", \"cita 2\", \"cita 3\"]\n"
        "  },\n"
        "  \"deep_dive_temas\": [\"Tema 1\", \"Tema 2\"]\n"
        "}"
    )


def get_insight_extraction_prompt(aggregated_data: dict) -> str:
    data_json = json.dumps(aggregated_data, ensure_ascii=False, default=str)
    return (
        "Eres un Analista de Datos y Estratega Senior. "
        "Analiza todos los datos proporcionados y devuelve EXCLUSIVAMENTE un objeto JSON con la estructura especificada. "
        "NO incluyas texto explicativo, comentarios ni markdown, solo el JSON. "
        "Los datos a analizar son:\n\n"
        "```json\n" + data_json + "\n```\n\n"
        "Objetivos de análisis (aplícalos al conjunto global y por categoría cuando existan datos):\n"
        "1) Serie temporal - tendencia: Describe si el sentimiento y el volumen son ascendentes, descendentes, estables o volátiles.\n"
        "2) Anomalías: Identifica picos/caídas significativas; indica fechas y magnitud (aprox.).\n"
        "3) Cruce Sentimiento vs Volumen: Explica si aumentos de conversación coinciden con caídas de sentimiento (o viceversa).\n"
        "4) Hipótesis: Aporta explicaciones plausibles de las anomalías (p.ej. campaña de competidor, noticia, lanzamiento).\n"
        "5) Conclusiones estratégicas: Resume oportunidades, riesgos y un plan de acción concreto.\n\n"
        "Estructura esperada del JSON de salida (rellena con tu análisis):\n\n"
        "{\n"
        "  \"executive_summary\": \"2-3 frases con lo más crítico y útil para negocio.\",\n"
        "  \"time_series_analysis\": {\n"
        "    \"global\": {\n"
        "      \"trend\": \"ascendente|descendente|estable|volátil\",\n"
        "      \"anomalies\": [ { \"date\": \"YYYY-MM-DD\", \"type\": \"spike|drop\", \"metric\": \"mentions|sentiment\", \"note\": \"breve explicación\" } ],\n"
        "      \"sentiment_vs_volume\": \"breve explicación del cruce\"\n"
        "    },\n"
        "    \"by_category\": [\n"
        "      { \"category\": \"Nombre\", \"trend\": \"...\", \"anomalies\": [ ... ], \"sentiment_vs_volume\": \"...\" }\n"
        "    ]\n"
        "  },\n"
        "  \"key_findings\": [\n"
        "    \"Un hallazgo importante sobre el mercado (basado en series y KPIs).\",\n"
        "    \"Otro hallazgo relevante sobre la competencia (SOV, picos, etc.).\",\n"
        "    \"Un tercer hallazgo sobre la percepción del cliente.\"\n"
        "  ],\n"
        "  \"opportunities\": [ { \"opportunity\": \"Descripción\", \"impact\": \"Alto|Medio|Bajo\" } ],\n"
        "  \"risks\": [ { \"risk\": \"Descripción\", \"mitigation\": \"Acción\" } ],\n"
        "  \"recommendations\": [ \"Acción 1\", \"Acción 2\", \"Acción 3\" ]\n"
        "}"
    )


def get_strategic_summary_prompt(insights_json: dict) -> str:
    executive_summary = insights_json.get("executive_summary", "")
    key_findings = insights_json.get("key_findings", [])
    findings_text = "\n- ".join(key_findings) if key_findings else ""
    return f"""
    **ROL:** Eres un Analista de Inteligencia de Mercado Senior.
    **TAREA:** Redacta la sección "Resumen Ejecutivo y Hallazgos Principales" de un informe estratégico.

    **EXECUTIVE SUMMARY (base):**
    {executive_summary}

    **HALLAZGOS PRINCIPALES:**
    - {findings_text}

    **INSTRUCCIÓN:**
    Redacta entre 2 y 3 párrafos claros y concisos que sinteticen el estado del mercado, la competencia y la percepción del cliente, utilizando el executive summary y los hallazgos como base. Evita repetir literalmente las viñetas; integra y sintetiza.
    """


def get_strategic_plan_prompt(insights_json: dict) -> str:
    opportunities = insights_json.get("opportunities", [])
    risks = insights_json.get("risks", [])
    recommendations = insights_json.get("recommendations", [])
    opp_text = json.dumps(opportunities, ensure_ascii=False, indent=2)
    risks_text = json.dumps(risks, ensure_ascii=False, indent=2)
    recs_text = "\n- ".join(recommendations) if recommendations else ""
    return f"""
    **ROL:** Eres un Estratega de Negocio.
    **TAREA:** Redacta la sección "Plan de Acción Estratégico" conectando oportunidades, riesgos y recomendaciones de forma coherente y priorizada.

    **OPORTUNIDADES (base):**
    ```json
    {opp_text}
    ```

    **RIESGOS (base):**
    ```json
    {risks_text}
    ```

    **RECOMENDACIONES (base):**
    - {recs_text}

    **INSTRUCCIÓN:**
    Propón un plan de acción priorizado (corto/medio plazo), asignando recomendaciones a oportunidades específicas y contemplando mitigaciones para los riesgos. Escribe 2-3 párrafos con lenguaje claro y accionable.
    DEVUELVE SOLO TEXTO CORRIDO en prosa. NO devuelvas JSON, ni listas con guiones, ni encabezados markdown.
    """


def get_executive_summary_prompt(aggregated_data: dict) -> str:
    """
    Prompt robusto para el Resumen Ejecutivo.
    Usa fallbacks para evitar valores 0 o ausentes y deriva competidores / SOV
    desde estructuras disponibles cuando falten claves concretas.
    """
    kpis = aggregated_data.get('kpis', {}) or {}
    client_name = aggregated_data.get('client_name', 'Nuestra marca')

    # Totales básicos
    total_mentions = int(kpis.get('total_mentions') or 0)

    # Sentimiento promedio: preferir average_sentiment; fallback a sentiment_avg
    avg_sent = (
        kpis.get('average_sentiment')
        if kpis.get('average_sentiment') is not None
        else kpis.get('sentiment_avg', 0.0)
    )
    try:
        avg_sent = float(avg_sent or 0.0)
    except Exception:
        avg_sent = 0.0

    # SOV total: preferir share_of_voice; fallback a sov
    sov_total = (
        kpis.get('share_of_voice')
        if kpis.get('share_of_voice') is not None
        else kpis.get('sov', 0.0)
    )
    try:
        sov_total = float(sov_total or 0.0)
    except Exception:
        sov_total = 0.0

    # Tabla de SOV para derivar competidores cuando no haya lista explícita
    sov_table = kpis.get('sov_table') or []  # esperado como [(brand, value), ...]

    # Competidores explícitos o derivados desde la tabla, excluyendo la marca cliente
    competitors = aggregated_data.get('market_competitors') or []
    if not competitors and isinstance(sov_table, list):
        try:
            # Obtener nombre de marca si estuviera en KPIs
            client_brand = kpis.get('brand_name') or aggregated_data.get('client_name')
            competitors = [str(b) for b, _ in sov_table if str(b) != str(client_brand)]
        except Exception:
            competitors = []

    # SOV por categoría: preferir bloque externo aggregated.sov.current
    sov_by_cat = {}
    try:
        sov_block = aggregated_data.get('sov') or {}
        current = sov_block.get('current') or {}
        sov_by_cat = current.get('sov_by_category') or {}
    except Exception:
        sov_by_cat = {}
    if not sov_by_cat:
        sov_by_cat = kpis.get('sov_by_category') or {}

    # Menciones por competidor: preferir aggregated.sov.current.competitor_mentions; fallback a tabla
    competitor_mentions = {}
    try:
        curr = (aggregated_data.get('sov') or {}).get('current') or {}
        competitor_mentions = curr.get('competitor_mentions') or {}
    except Exception:
        competitor_mentions = {}
    if not competitor_mentions and isinstance(sov_table, list):
        try:
            competitor_mentions = {str(b): float(v) for b, v in sov_table}
        except Exception:
            competitor_mentions = {}

    comp_json = json.dumps(competitor_mentions, ensure_ascii=False, indent=2)
    sov_cat_json = json.dumps(sov_by_cat, ensure_ascii=False, indent=2)

    return f"""
    **ROL:** Eres un Chief Insights Officer.
    **TAREA:** Redacta un RESUMEN EJECUTIVO (2-3 párrafos) claro y accionable.

    **DATOS CLAVE:**
    - Marca analizada: {client_name}
    - Competidores monitoreados: {', '.join(competitors) if competitors else 'N/D'}
    - Menciones totales (periodo): {total_mentions}
    - Sentimiento promedio (escala -1 a 1): {avg_sent:.2f}
    - Share of Voice total (cliente vs. competidores): {sov_total:.2f}%

    **Desglose de SOV por categoría (cliente/total):**
    ```json
    {sov_cat_json}
    ```

    **Menciones de competidores (conteo o % aproximado):**
    ```json
    {comp_json}
    ```

    **INSTRUCCIÓN:**
    - Sintetiza qué pasó en el periodo, qué temas destacaron y cómo quedó la competencia (usa SOV para evidenciarlo).
    - Indica implicaciones estratégicas para el negocio (no tácticas de bajo nivel).
    - Sé concreto, evita jerga, y prioriza claridad.
    """


def get_competitive_analysis_prompt(aggregated_data: dict) -> str:
    """Prompt robusto para la sección de competencia.

    Usa siempre datos reales disponibles en el objeto agregado. Si faltan claves
    en `kpis`, toma fallbacks desde:
      - `kpis.sov_table` para construir competidores y reparto de share.
      - `sov.current.sov_by_category` y `sov.current.competitor_mentions`.
    Así evitamos casos donde sale "0%" o "sin competidores" pese a existir datos.
    """
    kpis = aggregated_data.get('kpis', {}) or {}
    client_name = aggregated_data.get('client_name', 'Nuestra marca')

    # 1) Competidores y reparto base
    competitors: list[str] = []
    # a) si viene una lista explícita
    if isinstance(aggregated_data.get('market_competitors'), list):
        competitors = [str(x) for x in aggregated_data.get('market_competitors') if x]
    # b) derivar de la tabla de SOV si está disponible
    sov_table = kpis.get('sov_table') or []
    if not competitors and isinstance(sov_table, list):
        try:
            # tabla como [(brand, value_pct), ...]
            competitors = [str(b) for b, _ in sov_table if isinstance(b, str)]
        except Exception:
            pass

    # 2) SOV total del cliente; si no viene share_of_voice usa kpis.sov o 0.0
    sov_total = (
        kpis.get('share_of_voice')
        or kpis.get('sov')
        or 0.0
    )

    # 3) SOV por categoría: preferir bloque de `sov.current`
    sov_by_cat = {}
    sov_block = aggregated_data.get('sov') or {}
    try:
        current = sov_block.get('current') or {}
        by_cat = current.get('sov_by_category') or {}
        # convertir a forma compacta cliente/total si viene expandido
        for cat, entry in (by_cat.items() if isinstance(by_cat, dict) else []):
            if isinstance(entry, dict):
                sov_by_cat[str(cat)] = {
                    'client': int(entry.get('client', 0)),
                    'total': int(entry.get('total', 0)),
                }
    except Exception:
        pass
    # fallback a lo que pueda venir en kpis
    if not sov_by_cat:
        sov_by_cat = kpis.get('sov_by_category') or {}

    # 4) Menciones por competidor: preferir `sov.current.competitor_mentions`
    competitor_mentions = {}
    try:
        curr = (aggregated_data.get('sov') or {}).get('current') or {}
        competitor_mentions = curr.get('competitor_mentions') or {}
    except Exception:
        competitor_mentions = {}

    # Como último recurso, derivar menciones aproximadas de la tabla de SOV
    if not competitor_mentions and isinstance(sov_table, list):
        try:
            competitor_mentions = {str(b): float(v) for b, v in sov_table}
        except Exception:
            competitor_mentions = {}

    comp_json = json.dumps(competitor_mentions, ensure_ascii=False, indent=2)
    sov_cat_json = json.dumps(sov_by_cat, ensure_ascii=False, indent=2)

    return (
        f"""
    **ROL:** Eres un Analista de Competencia.
    **TAREA:** Redacta la sección "Análisis Competitivo" del informe.

    **Contexto:** Cliente = {client_name}. Competidores = {', '.join(competitors) if competitors else 'N/D'}.
    - SOV total del cliente: {float(sov_total):.2f}%
    - SOV por categoría (cliente/total):
    ```json
    {sov_cat_json}
    ```
    - Menciones por competidor:
    ```json
    {comp_json}
    ```

    **INSTRUCCIÓN:**
    - Identifica quién lidera la conversación total y por tema; resalta brechas (>10pp).
    - Señala categorías donde el cliente está subrepresentado y oportunidades para ganar share.
    - Concluye con 3 bullets de movimientos competitivos recomendados.
    """
    )


def get_deep_dive_analysis_prompt(category_name: str, kpis: dict, client_name: str) -> str:
    cat_kpis = kpis.get('sentiment_by_category', {}).get(category_name, {})

    sentiment_avg = cat_kpis.get('average', 0.0)
    distribution = cat_kpis.get('distribution', {})
    key_topics = cat_kpis.get('key_topics', {})

    dist_text = json.dumps(distribution, indent=2, ensure_ascii=False)
    topics_text = json.dumps(key_topics, indent=2, ensure_ascii=False)

    return f"""
    **ROL:** Eres un Analista de Inteligencia de Mercado experto en el sector de {client_name}.
    **TAREA:** Escribe el análisis para la sección "{category_name}" de un informe. Tu análisis debe ser una minería de datos, conectando los datos cuantitativos con conclusiones cualitativas y accionables.

    **DATOS CUANTITATIVOS DE ESTA SECCIÓN:**
    - Sentimiento Promedio de la Sección: {sentiment_avg:.2f} (comparado con el general de {kpis.get('average_sentiment', 0.0):.2f})
    - Distribución del Sentimiento:
    {dist_text}
    - Temas Clave (y su frecuencia):
    {topics_text}

    **INSTRUCCIONES:**
    1.  **Interpreta el Sentimiento:** ¿Es el sentimiento de esta categoría significativamente diferente del promedio general? ¿Qué nos dice la distribución (ej. es polarizado, mayormente neutral, etc.)?
    2.  **Conecta con los Temas:** ¿Cómo se relacionan los temas más frecuentes con el sentimiento observado? (ej. 'El sentimiento negativo está impulsado principalmente por conversaciones sobre 'precios'').
    3.  **Extrae un Insight Accionable:** Basado en esta conexión entre datos y temas, ¿cuál es la principal conclusión o recomendación para {client_name}?

    **FORMATO:** Redacta un párrafo de análisis denso, profesional y directo. No listes los datos, intégralos en tu narrativa.
    """


def get_correlation_interpretation_prompt(aggregated_data: dict, correlation_data: dict) -> str:
    data_json = json.dumps(correlation_data, indent=2, ensure_ascii=False)
    client_name = aggregated_data.get('client_name', 'Nuestra marca')
    return f"""
    **ROL:** Eres un Analista Principal de Insights con enfoque causal.
    **TAREA:** Interpreta las correlaciones transversales entre categorías y KPIs y extrae 3-5 insights potentes y accionables para {client_name}.

    **DATOS DE CORRELACIÓN (base):**
    ```json
    {data_json}
    ```

    **INSTRUCCIONES:**
    - Prioriza relaciones con mayor "strength" y soporte cuantitativo.
    - Formula hipótesis plausibles y cómo validarlas (dato/experimento).
    - Concreta implicaciones y una recomendación por insight.
    - Responde en 2-3 párrafos, profesional y claro.
    """


def get_trends_anomalies_prompt(aggregated_data: dict) -> str:
    trends = aggregated_data.get('trends', {})
    prev = aggregated_data.get('previous_period', {})
    trends_json = json.dumps(trends, indent=2, ensure_ascii=False)
    return f"""
    **ROL:** Eres un Analista de Tendencias.
    **TAREA:** Redacta la sección "Tendencias y Señales Emergentes" del informe, explicando cambios significativos entre periodos.

    **PERIODO ANTERIOR:** {prev.get('start_date', 'N/D')} a {prev.get('end_date', 'N/D')}
    **CAMBIOS (trends JSON):**
    ```json
    {trends_json}
    ```

    **INSTRUCCIONES:**
    - Explica las variaciones de sentimiento (total y por categoría) y SOV por categoría.
    - Destaca competidores que ganaron/perdieron share y por qué podría estar pasando.
    - Enumera 3-5 tópicos emergentes y valora si son moda o tendencia (con criterios).
    - Cierra con 3 recomendaciones tácticas inmediatas y 2 estratégicas.
    """


def get_agent_insights_summary_prompt(data: dict) -> str:
    """
    Genera un prompt para resumir el análisis de insights de agentes.
    Espera una clave 'agent_insights' con buckets normalizados.
    """
    import json
    agent_json = json.dumps(data.get("agent_insights", {}), ensure_ascii=False, indent=2)
    return f"""
    **Rol y Objetivo:**
    Actúa como Analista Jefe de Estrategias de Agentes. Tu misión es analizar el seguimiento de insights del Answer Engine y presentar los hallazgos clave.

    **Estructura Requerida:**
    1.  **Rendimiento de Agentes (3 bullets):** Tres puntos que resuman el desempeño general de los agentes (calidad de oportunidades/risks/tendencias, señales repetidas, y cobertura de temas).
    2.  **Insights Destacados:** Un párrafo corto que describa los insights más relevantes identificados (oportunidades y riesgos más fuertes) con referencia a su impacto.
    3.  **Recomendaciones:** Dos recomendaciones basadas en el análisis para próximas decisiones.

    **Datos para el Análisis (JSON):**
    ```json
    {agent_json}
    ```

    **Instrucción Final:**
    Genera el resumen siguiendo estrictamente la estructura y el tono descritos. Sé claro y ejecutivo.
    """


def get_deep_dive_mentions_prompt(topic: str, mentions: list[str]) -> str:
    """
    Prompt para analizar en profundidad menciones textuales sobre un tema.
    Devuelve JSON estricto con: sintesis_del_hallazgo, causa_raiz y citas_destacadas.
    """
    import json
    corpus = "\n\n".join([m.strip()[:4000] for m in mentions if isinstance(m, str)])
    return (
        "Actúa como Analista Cualitativo Senior. Analiza las menciones textuales del tema especificado "
        "y devuelve EXCLUSIVAMENTE un JSON con las claves pedidas. Sé sintético y profesional. "
        f"\n\nTema: {topic}\n"
        "Menciones (texto literal, truncado si es largo):\n"
        f"{corpus}\n\n"
        "Formato de salida (JSON estricto):\n"
        "{\n"
        "  \"sintesis_del_hallazgo\": \"3-5 frases con el hallazgo principal y su impacto para negocio.\",\n"
        "  \"causa_raiz\": \"Hipótesis clara de la causa raíz basada en patrones en las menciones.\",\n"
        "  \"citas_destacadas\": [\"cita 1\", \"cita 2\", \"cita 3\"]\n"
        "}"
    )


def get_cluster_analyst_prompt(cluster: dict) -> str:
    """Prompt para Analista de Clusters (Nivel 1). Devuelve JSON con topic_name y key_points."""
    import json
    examples = [
        {
            "id": m.get("id"),
            "summary": (m.get("summary") or "")[:300],
            "sentiment": m.get("sentiment"),
        }
        for m in cluster.get("example_mentions", [])
    ]
    payload = {
        "examples": examples,
        "volume": cluster.get("count", 0),
        "avg_sentiment": cluster.get("avg_sentiment", 0.0),
        "top_sources": cluster.get("top_sources", []),
    }
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""
Eres un analista especializado en nombrar y resumir temas de conversación.
Con base en los EJEMPLOS representativos del cluster y sus métricas, devuelve SOLO este JSON:
{{"topic_name": "...", "key_points": ["...", "..."]}}

Datos de entrada (JSON):
{data_json}
"""


def get_clusters_synthesizer_prompt(clusters: list[dict]) -> str:
    """Prompt para Sintetizador Estratégico (Nivel 2) sobre múltiples clusters."""
    import json
    brief = [
        {
            "topic_name": c.get("topic_name", "(sin nombre)"),
            "volume": c.get("volume", 0),
            "sentiment": c.get("sentiment", 0.0),
        }
        for c in clusters
    ]
    data_json = json.dumps(brief, ensure_ascii=False)
    return f"""
Eres un estratega de mercado senior. Con el resumen de los principales temas (clusters), devuelve SOLO este JSON:
{{
  "meta_narrativas": ["...", "..."],
  "oportunidad_principal": "...",
  "riesgo_inminente": "...",
  "plan_estrategico": ["acción 1", "acción 2"]
}}

Datos de entrada (JSON):
{data_json}
"""