import json
from typing import Dict, List, Any

# Constante para el nombre de cliente por defecto.
DEFAULT_CLIENT_NAME = "Cliente"

# Bloque de Reglas de Rigor Analítico para todos los prompts.
ANALYTICAL_RIGOR_FOOTER = (
    "\n\n**REGLAS DE RIGOR ANALÍTICO (OBLIGATORIAS):**\n"
    "1.  **Cero Alucinaciones:** Basa el 100% de tus afirmaciones y conclusiones exclusivamente en los datos proporcionados. No inventes información.\n"
    "2.  **Trazabilidad de Datos:** Cuando uses una métrica, cítala entre paréntesis. Ejemplo: 'El sentimiento promedio cayó a -0.25 (KPIs Globales)'.\n"

"3.  **Declaración de Incertidumbre:** Si los datos son insuficientes, decláralo. Califica la confianza de tus hipótesis (Alta, Media, Baja).\n"
    "4.  **Prohibición de Generalidades:** Evita recomendaciones vacías. Cada recomendación debe ser una acción concreta derivada del análisis."
)

def get_detailed_category_prompt(
    category_name: str, category_data: List[Dict[str, Any]], all_summaries: List[str], kpis: Dict[str, Any]
) -> str:
    data_json = json.dumps(category_data, indent=2, ensure_ascii=False, default=str)
    summaries_text = "\n- ".join(list(set(all_summaries)))
    _sov_cat = kpis.get('sov_by_category', {}).get(category_name, {"client": 0, "total": 0})
    _client = float(_sov_cat.get('client', 0))
    _total = float(_sov_cat.get('total', 0))
    _sov_pct = (_client / max(1, _total) * 100.0) if _total > 0 else 0.0

    prompt = f"""
    **ROL:** Eres un Director de Inteligencia de Mercados, con mas de 40 años en el sector especializado en análisis causal y competitivo.

    **TAREA:** Elaborar un diagnóstico estratégico para la categoría "{category_name}". Explica la dinámica de mercado subyacente que genera los datos con rigor analítico y causal.

    **CONTEXTO ANALÍTICO:**
    - **Categoría de Análisis:** "{category_name}"
    - **KPIs de Referencia (Mercado General):**
      - Menciones Totales: {kpis.get('total_mentions', 'Dato no disponible')}
      - Sentimiento Promedio General: {kpis.get('average_sentiment', 0.0):.2f}
    - **KPIs Específicos de la Categoría:**
      - Share of Voice (SOV): {_sov_pct:.2f}%

    **EVIDENCIA (Datos brutos):**
    ```json
    {data_json}
    ```

    **CONTEXTO (Temas globales):**
    - {summaries_text}

    **METODOLOGÍA DE ANÁLISIS:**
    1.  **Análisis Causal (El "Porqué"):** Explica la razón detrás de las métricas. ¿Por qué el sentimiento es el que es? Fundamenta tu hipótesis causal.
    2.  **Diagnóstico Competitivo:** Evalúa el SOV. ¿Es una posición dominante, de retador o marginal? ¿Quién lidera la narrativa y con qué mensaje? Identifica una vulnerabilidad explotable del líder.
    3.  **Insight Estratégico y Recomendación:** Concluye con UN insight potente y UNA recomendación estratégica, específica y accionable.

    **FORMATO DE SALIDA:**
    Un párrafo de análisis denso y ejecutivo (150-200 palabras).
    """
    return prompt + ANALYTICAL_RIGOR_FOOTER

def get_main_analyst_prompt(aggregated_data: Dict[str, Any], global_mentions_corpus: List[str]) -> str:
    data_json = json.dumps(aggregated_data, ensure_ascii=False, indent=2, default=str)
    corpus = "\n\n".join([str(m)[:4000] for m in (global_mentions_corpus or []) if isinstance(m, str)])
    
    # Se define la estructura JSON fuera del f-string para evitar conflictos de comillas
    json_format_structure = """
    {
      "headline": "Titular estratégico que resuma la principal tensión u oportunidad del mercado.",
      "evaluacion_general": "Diagnóstico ejecutivo del periodo. ¿Ganamos o perdimos terreno y por qué? ¿Cuál es la fuerza dominante que ha marcado este periodo?",
      "analisis_profundo": "Análisis detallado de las dinámicas clave. Conecta series temporales con temas emergentes y acciones de la competencia. Formula hipótesis causales claras.",
      "analisis_competencia": "Lectura estratégica del panorama competitivo. ¿Quién gana la narrativa y dónde están sus vulnerabilidades?",
      "analisis_mercado": "Análisis de las corrientes de fondo del mercado. ¿Qué necesidades no cubiertas emergen del corpus? ¿Qué tendencias son pasajeras y cuáles estructurales?",
      "cualitativo_global": {
        "sintesis_del_hallazgo": "El insight cualitativo más profundo extraído del corpus.",
        "causa_raiz": "Tu hipótesis principal sobre la causa raíz del comportamiento observado.",
        "citas_destacadas": ["Cita que encapsule la emoción principal", "Cita que revele una necesidad latente"]
      },
      "deep_dive_temas": ["Tema 1: El más urgente", "Tema 2: El de mayor potencial"]
    }
    """

    prompt = (
        f"**ROL:** Actúa como un Chief Strategy Officer (CSO). Tu especialidad es sintetizar inteligencia de mercado para formular la narrativa estratégica del próximo trimestre.\n\n"
        f"**MISIÓN:** Analiza la totalidad de los datos para construir un 'Dossier Estratégico Ejecutivo' en formato JSON. Eres un intérprete y un estratega.\n\n"
        f"**MARCO DE ANÁLISIS ESTRATÉGICO:**\n"
        f"1.  **Síntesis Holística:** Conecta los puntos entre datos cuantitativos y cualitativos.\n"
        f"2.  **Extracción de 'Second-Order Insights':** Ve más allá de lo que el dato dice y enfócate en lo que *significa* para el negocio.\n"
        f"3.  **Validación Cualitativa:** Usa el CORPUS GLOBAL como evidencia principal para cada afirmación.\n"
        f"4.  **Identificación de Oportunidades/Amenazas:** Culmina identificando las mayores oportunidades o amenazas.\n\n"
        f"**DATOS CUANTITATIVOS (JSON):**\n{data_json}\n\n"
        f"**CORPUS GLOBAL DE MENCIONES (Evidencia Cualitativa Primaria):\n{corpus}\n\n"
        f"**FORMATO DE SALIDA (JSON ESTRICTO):**\n{json_format_structure}"
    )
    return prompt + ANALYTICAL_RIGOR_FOOTER

def get_insight_extraction_prompt(aggregated_data: Dict[str, Any]) -> str:
    data_json = json.dumps(aggregated_data, ensure_ascii=False, default=str)
    client_name = aggregated_data.get('client_name') or (aggregated_data.get('kpis', {}) or {}).get('brand_name') or DEFAULT_CLIENT_NAME
    
    # Se define la estructura JSON fuera del f-string
    json_format_structure = """
    {
      "executive_summary": "El insight más crítico y su implicación de negocio en 2 frases.",
      "assumptions": ["Suposición clave 1", "Suposición clave 2"],
      "limitations": ["Limitación de dato 1", "Limitación de dato 2"],
      "time_series_analysis": {
        "global": {
          "trend_description": "Descripción cuantificada de la tendencia y volatilidad.",
          "anomalies": [ { "date": "YYYY-MM-DD", "metric": "menciones|sentimiento", "magnitude": "+30% o -0.5", "causal_hypothesis": "Hipótesis explicativa", "confidence": "Alta|Media|Baja" } ],
          "sentiment_vs_volume_correlation": "Descripción de la correlación observada."
        }
      },
      "key_findings": ["Hallazgo estratégico 1", "Hallazgo estratégico 2", "Hallazgo estratégico 3"],
      "opportunities": [ { "opportunity": "Descripción", "impact": "Alto|Medio|Bajo", "probability": "Alta|Media|Baja", "effort": "Bajo|Medio|Alto", "confidence": "Alta|Media|Baja", "evidence_ids": ["id1","id2"] } ],
      "risks": [ { "risk": "Descripción", "mitigation": "Acción específica", "impact": "Alto|Medio|Bajo", "probability": "Alta|Media|Baja", "evidence_ids": ["id1"] } ],
      "recommendations": [ "Acción medible 1", "Acción medible 2" ],
      "recommendations_structured": [ { "action": "Acción concreta", "metric": "KPI medible", "owner": "Equipo/Persona", "due": "Q4 2025", "impact": "Alto|Medio|Bajo", "effort": "Bajo|Medio|Alto", "probability": "Alta|Media|Baja", "confidence": "Alta|Media|Baja", "evidence_ids": ["id1","id2"] } ]
    }
    """

    prompt = (
        f"**ROL:** Eres un Científico de Datos especializado en análisis causal. Tu tarea es procesar los datos de mercado y transformarlos en un JSON estructurado de insights.\n\n"
        f"**MARCA OBJETIVO:** '{client_name}'.\n\n"
        f"**DATOS BRUTOS DE ENTRADA:**\n"
        f"```json\n{data_json}\n```\n\n"
        f"**METODOLOGÍA DE ANÁLISIS (OBLIGATORIA):**\n"
        f"1.  **Análisis de Series Temporales:**\n"
        f"            - **Tendencia y Volatilidad:** Cuantifica la tendencia y califica su volatilidad (Baja/Media/Alta).\n"
        f"            - **Anomalías y Puntos de Inflexión:** Identifica picos/valles, provee fecha, métrica, magnitud e hipótesis causal con nivel de confianza.\n"
        f"2.  **Diagnóstico Estratégico:**\n"
        f"            - **Hallazgos Clave:** Transforma datos en 3 conclusiones críticas.\n"
        f"            - **Matriz Oportunidad/Riesgo:** Evalúa cada oportunidad y riesgo por su Impacto y Probabilidad.\n"
        f"            - **Recomendaciones Medibles:** Deriva acciones concretas y medibles.\n\n"
        f"**FORMATO DE SALIDA (JSON ESTRICTO):**\n{json_format_structure}"
    )
    return prompt + ANALYTICAL_RIGOR_FOOTER

def get_challenger_critique_prompt(
    data: Dict[str, Any], *, client_name: str | None = None
) -> str:
    """
    "Challenger" crítico: cuestiona, depura y fortalece las oportunidades/risks/recomendaciones.
    Devuelve JSON con sufijo _refined. Puede aprovechar un catálogo de evidencias.
    """
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    brand = client_name or data.get("client_name") or DEFAULT_CLIENT_NAME
    return (
        f"""
        **ROL:** Actúas como un 'Challenger' escéptico y orientado a resultados para {brand}.

        **TAREA:** Revisar críticamente oportunidades, riesgos y recomendaciones. Elimina vaguedades, exige evidencia y convierte ideas genéricas en acciones SMART.

        **DATOS DE ENTRADA (JSON):**
        ```json
        {payload}
        ```

        **REGLAS:**
        1. Elimina elementos que no tengan vínculo claro con datos o evidencias.
        2. Exige especificidad: cada recomendación debe tener action, metric, owner y due.
        3. Mantén y añade campos: impact, effort, probability, confidence, evidence_ids.
        4. Si hay "evidence_catalog", prioriza evidence_ids que soporten cada punto.
        5. Prohíbe campañas de marketing genéricas. Si una recomendación es una "campaña" o acciones tipo "SEM/ads/marketing", reescríbela como una acción operativa concreta (académica, producto, operaciones, partnerships, empleabilidad) con owner, KPI y due.

        **SALIDA (JSON ESTRICTO, SOLO JSON):**
        {{
          "opportunities_refined": [
            {{"opportunity": "...", "impact": "Alto", "probability": "Alta", "effort": "Medio", "confidence": "Media", "evidence_ids": ["..."]}}
          ],
          "risks_refined": [
            {{"risk": "...", "mitigation": "...", "impact": "Alto", "probability": "Media", "evidence_ids": ["..."]}}
          ],
          "recommendations_refined": [
            {{"action": "...", "metric": "...", "owner": "...", "due": "Q4 2025", "impact": "Alto", "effort": "Medio", "probability": "Alta", "confidence": "Alta", "evidence_ids": ["..."]}}
          ]
        }}
        """
    ) + ANALYTICAL_RIGOR_FOOTER

def get_strategic_summary_prompt(insights_json: Dict[str, Any], *, client_name: str) -> str:
    executive_summary = insights_json.get("executive_summary", "")
    key_findings = insights_json.get("key_findings", [])
    findings_text = "\n- ".join(f'"{finding}"' for finding in key_findings) if key_findings else ""

    # Contexto ampliado opcional
    ctx = insights_json.get("context", {}) if isinstance(insights_json, dict) else {}
    try:
        ctx_json = json.dumps({
            "competitive_analysis": ctx.get("competitive_analysis"),
            "trends": ctx.get("trends"),
            "correlations": ctx.get("correlations"),
            "action_plan": ctx.get("action_plan"),
            "sov_by_category": ctx.get("sov_by_category"),
            "kpis": ctx.get("kpis"),
        }, ensure_ascii=False, indent=2)
    except Exception:
        ctx_json = json.dumps({}, ensure_ascii=False)

    prompt = f"""
    **ROL:** Eres un Comunicador Estratégico y Asesor de C-Level. Tu habilidad es transformar datos complejos en una narrativa clara, concisa y persuasiva.

    **TAREA:** Redactar la sección "Resumen Ejecutivo y Hallazgos Principales" para la marca **{client_name}**.

    **MATERIAL DE BASE:**
    - **Síntesis Ejecutiva Preliminar:** "{executive_summary}"
    - **Hallazgos Clave Aislados:**
      - {findings_text}
    - **Contexto Estructurado (usar e integrar):**
    ```json
    {ctx_json}
    ```

    **INSTRUCCIONES DE COMUNICACIÓN ESTRATÉGICA:**
    1.  **Crea una Narrativa:** No enumeres. Teje una historia coherente que integre KPIs, SOV por categoría, análisis competitivo, tendencias, correlaciones y el plan de acción.
    2.  **Enfoque '¿Y qué?':** Transforma cada dato en una implicación estratégica para el negocio.
    3.  **Lenguaje de Decisión:** Utiliza un lenguaje activo y directo para facilitar la toma de decisiones.
    4.  **Consistencia con el Dossier:** Asegúrate de cubrir los mismos apartados que aparecen en el PDF, evitando contradicciones entre secciones.
    5.  **Regla:** No menciones carencias de desglose de SOV por categoría. Si hay datos limitados, expresa el grado de confianza sin afirmar que faltan apartados.

    **FORMATO DE SALIDA:**
    Entre 10 y 15 párrafos de prosa ejecutiva, inmediatamente comprensible para un directivo.
    """
    return prompt + ANALYTICAL_RIGOR_FOOTER

def get_strategic_plan_prompt(insights_json: Dict[str, Any], *, client_name: str | None = None) -> str:
    """
    Nivel Élite: Exige un plan de acción riguroso, priorizado y justificado.
    """
    opportunities = insights_json.get("opportunities", [])
    risks = insights_json.get("risks", [])
    recommendations = insights_json.get("recommendations", [])
    opp_text = json.dumps(opportunities, ensure_ascii=False, indent=2)
    risks_text = json.dumps(risks, ensure_ascii=False, indent=2)
    recs_text = "\n- ".join(recommendations) if recommendations else ""
    
    prompt = f"""
    **ROL:** Eres un Director de Estrategia y Operaciones. Tu responsabilidad es convertir insights en un plan de acción ejecutable, priorizado y con una lógica de negocio impecable.

    **TAREA:** Elaborar la sección "Plan de Acción Estratégico".
    {('**MARCA:** ' + client_name + ' — Todas las acciones deben fortalecer la posición de esta marca.' ) if client_name else ''}

    **INPUTS (Análisis Previo):**
    - **Oportunidades:**
    ```json
    {opp_text}
    ```
    - **Riesgos:**
    ```json
    {risks_text}
    ```
    - **Recomendaciones Preliminares:**
    - {recs_text}

    **METODOLOGÍA DE PLANIFICACIÓN ESTRATÉGICA:**
    1.  **Priorización Basada en Impacto:** Prioriza las recomendaciones en función de qué oportunidades capitalizan y qué riesgos de alto impacto mitigan. Usa un marco implícito de impacto vs. esfuerzo.
    2.  **Conexión Lógica:** Justifica *por qué* cada acción es la respuesta correcta al dato analizado.
    3.  **Especificidad y Medición:** Transforma recomendaciones genéricas en iniciativas concretas. Define el "qué" y el "porqué".
    4.  **Secuenciación:** Organiza las acciones en una secuencia lógica (ej. corto/medio plazo).

    **RESTRICCIONES (OBLIGATORIAS):**
    -  No propongas campañas de marketing (ni "campañas", ni "SEM/ads", ni aumentos de presupuesto publicitario).
    -  Favorece acciones operativas y de producto/experiencia académica: empleabilidad, partnerships con empresas, currículum, procesos de admisiones/soporte, calidad docente, acuerdos y certificaciones.
    -  Incluye al menos 2 acciones operativas no de marketing (ejemplos: "Implementar programa de orientación y empleabilidad con 4 talleres/mes", "Firmar 5 convenios de prácticas remuneradas con estudios/productoras en 90 días").

    **FORMATO DE SALIDA:**
    Un texto en prosa, articulado y profesional. Estructúralo en 10 y 15 párrafos que presenten el plan de forma coherente. NO devuelvas JSON ni listas.
    """
    return prompt + ANALYTICAL_RIGOR_FOOTER

def get_executive_summary_prompt(aggregated_data: Dict[str, Any]) -> str:
    """
    Nivel Élite: Eleva el rol a Chief Strategy Officer, demandando una visión de futuro.
    """
    kpis = aggregated_data.get('kpis', {}) or {}
    client_name = aggregated_data.get('client_name', DEFAULT_CLIENT_NAME)
    total_mentions = int(kpis.get('total_mentions') or 0)
    avg_sent = float(kpis.get('average_sentiment') or kpis.get('sentiment_avg') or 0.0)
    sov_total = float(kpis.get('share_of_voice') or kpis.get('sov') or 0.0)
    sov_table = kpis.get('sov_table') or []
    competitors = aggregated_data.get('market_competitors') or [str(b) for b, _ in sov_table if str(b) != client_name]
    sov_by_cat = (aggregated_data.get('sov', {}).get('current', {}).get('sov_by_category') or kpis.get('sov_by_category') or {})
    competitor_mentions = (aggregated_data.get('sov', {}).get('current', {}).get('competitor_mentions') or {str(b): float(v) for b, v in sov_table})
    comp_json = json.dumps(competitor_mentions, ensure_ascii=False, indent=2)
    sov_cat_json = json.dumps(sov_by_cat, ensure_ascii=False, indent=2)

    prompt = f"""
    **ROL:** Eres el Chief Strategy Officer (CSO). Tu audiencia es el Consejo de Administración. Esperan de ti una visión aguda, sintética y orientada al futuro.

    **TAREA:** Redactar el Resumen Ejecutivo Estratégico del trimestre.

    **DATOS CLAVE DEL PERIODO:**
    - Marca: {client_name}
    - Competidores Clave: {', '.join(competitors) if competitors else 'N/D'}
    - Volumen de Conversación: {total_mentions} menciones
    - Reputación (Sentimiento Promedio): {avg_sent:.2f}
    - Posición Competitiva (SOV Total): {sov_total:.2f}%

    **Desglose de SOV por Arena Competitiva (Categoría):**
    ```json
    {sov_cat_json}
    ```

    **Pulso de la Competencia (Menciones):**
    ```json
    {comp_json}
    ```

    **DIRECTRICES ESTRATÉGICAS:**
    1.  **Diagnóstico (Pasado):** ¿Cuál es la historia principal que cuentan los datos? ¿Ganamos o perdimos? ¿En qué campos de batalla (categorías)? Sé directo pero si necesitas analizar un tema en especifico porque encuentras algo interesante hazlo.
    2.  **Implicación (Presente):** ¿Qué significa este diagnóstico para nuestra posición actual? ¿Qué fortalezas consolidar y qué debilidades son críticas?
    3.  **Prescripción (Futuro):** ¿Cuál es la recomendación estratégica número uno que el Consejo debe considerar? Debe ser una acción audaz que responda al análisis.

    **FORMATO:** 10 y 15 párrafos densos en información, con un lenguaje de alto nivel enfocado en el impacto de negocio y en la visión de futuro.
    """
    return prompt + ANALYTICAL_RIGOR_FOOTER

def get_competitive_analysis_prompt(aggregated_data: Dict[str, Any]) -> str:
    """
    Nivel Élite: Centrado en la explotación de vulnerabilidades competitivas.
    Soporta perspectiva explícita: {client_name} analiza al {focus_competitor}.
    """
    kpis_client = aggregated_data.get('kpis', {}) or {}
    client_name = aggregated_data.get('client_name', DEFAULT_CLIENT_NAME)
    focus = aggregated_data.get('focus_competitor')
    comp_kpis = aggregated_data.get('competitor_kpis') or {}

    # Datos generales del mercado (para contexto)
    sov_total_client = float(kpis_client.get('share_of_voice') or kpis_client.get('sov') or 0.0)
    sov_table = kpis_client.get('sov_table') or []

    # SOV por categoría: permitir bundle específico del competidor
    sov_by_cat_focus = aggregated_data.get('sov_by_category_focus') or (
        aggregated_data.get('sov', {}).get('current', {}).get('sov_by_category')
    ) or kpis_client.get('sov_by_category') or {}

    # Pulso de la competencia (si viene del bundle general)
    competitor_mentions = (
        aggregated_data.get('sov', {}).get('current', {}).get('competitor_mentions')
        or {str(b): float(v) for b, v in sov_table}
    )

    comp_json = json.dumps(competitor_mentions, ensure_ascii=False, indent=2)
    sov_cat_json = json.dumps(sov_by_cat_focus, ensure_ascii=False, indent=2)
    comp_kpis_json = json.dumps(comp_kpis, ensure_ascii=False, indent=2)

    prompt = (
        f"""
    **ROL:** Eres un Analista de Inteligencia Competitiva, experto en identificar y explotar las debilidades del adversario.

    **PERSPECTIVA:** {client_name} analiza al competidor focal {focus or 'N/D'}. No confundas roles: {client_name} = nuestra marca; {focus or 'N/D'} = competidor a evaluar.

    **DOSSIER DE INTELIGENCIA:**
    - Cliente: {client_name}
    - Competidor focal: {focus or 'N/D'}
    - SOV del Cliente (Total): {sov_total_client:.2f}%
    - KPIs del Competidor Focal (usar SOV extendido si está disponible):
    ```json
    {comp_kpis_json}
    ```
    - Territorios del Competidor (SOV por categoría):
    ```json
    {sov_cat_json}
    ```
    - Actividad del mercado por marca (menciones):
    ```json
    {comp_json}
    ```

    **PROTOCOLO DE ANÁLISIS:**
    1.  **Identificar Fortalezas y Flancos del Competidor Focal:** ¿Dónde concentra cuota (SOV%) y con qué sentimiento? Señala 1-2 flancos explotables.
    2.  **Guerrilla por Categoría:** Detecta nichos donde {client_name} pueda ganar terreno rápidamente frente a {focus or 'el competidor'}.
    3.  **Estrategias de Flanqueo (3-5 bullets):** Acciones específicas para {client_name} directamente orientadas a arrebatar cuota y mejorar percepción.

    **FORMATO:** Texto ejecutivo orientado a decisión. No narres sobre el cliente equivocado.
    """
    )
    return prompt + ANALYTICAL_RIGOR_FOOTER

def get_deep_dive_analysis_prompt(category_name: str, kpis: Dict[str, Any], client_name: str) -> str:
    """
    Nivel Élite: Exige una inmersión profunda en la psicología del consumidor de la categoría.
    """
    cat_kpis = kpis.get('sentiment_by_category', {}).get(category_name, {})
    sentiment_avg = cat_kpis.get('average', 0.0)
    distribution = cat_kpis.get('distribution', {})
    key_topics = cat_kpis.get('key_topics', {})
    dist_text = json.dumps(distribution, indent=2, ensure_ascii=False)
    topics_text = json.dumps(key_topics, indent=2, ensure_ascii=False)

    prompt = f"""
    **ROL:** Eres un Investigador Cualitativo Senior con experiencia en etnografía digital, analizando para {client_name}.

    **TAREA:** Realizar un "Análisis Psicológico Profundo" de la categoría "{category_name}". Tu objetivo es ir más allá de los datos para entender las motivaciones, frustraciones y necesidades no expresadas del consumidor.

    **DATOS CUANTITATIVOS PARA CONTEXTUALIZAR:**
    - Sentimiento Promedio (Categoría vs. General): {sentiment_avg:.2f} vs. {kpis.get('average_sentiment', 0.0):.2f}
    - Distribución del Sentimiento:
    {dist_text}
    - Temas de Conversación (Frecuencia):
    {topics_text}

    **GUÍA DE ANÁLISIS PROFUNDO:**
    1.  **El Significado del Sentimiento:** ¿Qué emoción subyace a la métrica de sentimiento? Si es negativo, ¿es frustración, decepción, enfado? Si es positivo, ¿es satisfacción, entusiasmo, lealtad?
    2.  **La Tensión en los Temas:** ¿Qué conflicto o tensión revelan los temas clave? (ej. alta frecuencia de "precio" y "calidad" sugiere una tensión en torno al valor percibido).
    3.  **El Insight Oculto y la Oportunidad:** Conecta la emoción (sentimiento) con la tensión (temas) para desvelar una necesidad no satisfecha. A partir de este insight, formula una oportunidad de innovación para {client_name}.

    **FORMATO:** 3 párrafos de análisis narrativo que revele la psicología del consumidor y concluya con una oportunidad de innovación clara.
    """
    return prompt + ANALYTICAL_RIGOR_FOOTER

def get_correlation_interpretation_prompt(aggregated_data: Dict[str, Any], correlation_data: Dict[str, Any]) -> str:
    """
    Nivel Élite: Enfocado en la validación de hipótesis y la causalidad.
    """
    data_json = json.dumps(correlation_data, indent=2, ensure_ascii=False)
    client_name = aggregated_data.get('client_name', DEFAULT_CLIENT_NAME)
    
    prompt = f"""
    **ROL:** Científico de Datos experto en inferencia causal aplicada a marketing para {client_name}.

    **TAREA:** Redactar la sección "Correlaciones Transversales entre Categorías" combinando cuantitativo (Pearson r, matriz, lags) y lectura estratégica por categoría (SOV% vs Sentimiento). Odia las vaguedades: cada conclusión debe incluir un dato y una acción.

    **DATOS:**
    ```json
    {data_json}
    ```

    **GUÍA DE ANÁLISIS (OBLIGATORIA):**
    1.  **Resumen Ejecutivo (3 bullets):** Los 3 hallazgos más relevantes con su implicación.
    2.  **Matriz de Correlaciones (menciones, visibilidad, sentimiento):** Explica signo, magnitud (|r|) y n. Señala relaciones espurias potenciales.
    3.  **Correlación Temporal con Desfase (±7 días):** Indica el mejor lag volumen→sentimiento y cómo interpretarlo operativamente.
    4.  **Cuadrantes por Categoría (SOV% vs Sentimiento):** Lista Top 3 "OPORTUNIDAD" y Top 3 "RIESGO" con CTA concreto por cada una.
    5.  **Hipótesis Causales + Validación:** Para 2-3 relaciones fuertes, formula hipótesis, experimento A/B (métrica, tamaño muestral aproximado, duración) y criterio de éxito.
    6.  **Limitaciones y Próximos Pasos:** Qué no se puede concluir aún y qué analizar a continuación.

    **FORMATO:** Texto ejecutivo con subtítulos (como arriba) y bullets donde sea útil. Cita r y n siempre que menciones una correlación.
    """
    return prompt + ANALYTICAL_RIGOR_FOOTER

def get_trends_anomalies_prompt(aggregated_data: Dict[str, Any]) -> str:
    """
    Nivel Élite: Análisis de tendencias con enfoque en la velocidad del cambio (momentum).
    """
    trends = aggregated_data.get('trends', {})
    prev = aggregated_data.get('previous_period', {})
    trends_json = json.dumps(trends, indent=2, ensure_ascii=False)
    
    prompt = f"""
    **ROL:** Eres un Analista de Tendencias y Futuros (Futures Analyst).

    **TAREA:** Redactar la sección "Tendencias, Anomalías y Momentum del Mercado".

    **DATOS DE CAMBIO (Periodo Actual vs. {prev.get('start_date', 'N/D')} - {prev.get('end_date', 'N/D')}):**
    ```json
    {trends_json}
    ```

    **FRAMEWORK DE ANÁLISIS DE TENDENCIAS:**
    1.  **Magnitud y Velocidad:** No solo identifiques qué ha cambiado, sino la magnitud y la velocidad del cambio. ¿Es un cambio lento o una disrupción repentina?
    2.  **Análisis de 'Drivers':** ¿Qué fuerzas están impulsando estas tendencias? ¿Son movimientos de la competencia, cambios en el comportamiento del consumidor, etc.?
    3.  **De Señal Débil a Tendencia Fuerte:** Para los tópicos emergentes, evalúa su potencial. ¿Es una "señal débil" (pasajera) o una "tendencia fuerte" (estructural)? Justifica tu evaluación.
    4.  **Implicaciones Estratégicas y Tácticas:** Concluye con 2 recomendaciones estratégicas (a 12-18 meses) y 3 tácticas inmediatas (a 30-60 días).

    **FORMATO:** Un análisis ejecutivo que siga el framework descrito con detalle de todo lo que esta pasando.
    """
    return prompt + ANALYTICAL_RIGOR_FOOTER

def get_agent_insights_summary_prompt(data: Dict[str, Any]) -> str:
    """
    Nivel Élite: Meta-análisis sobre la calidad de los insights generados por el sistema.
    """
    agent_json = json.dumps(data.get("agent_insights", {}), ensure_ascii=False, indent=2)
    
    prompt = f"""
    **ROL:** Eres un Auditor de Sistemas de Inteligencia Artificial. Tu misión es realizar un meta-análisis de los insights generados por el "Answer Engine" para evaluar su rendimiento y proponer mejoras.

    **ESTRUCTURA DEL INFORME DE AUDITORÍA:**
    1.  **Diagnóstico del Rendimiento del Agente (3 bullets):**
        -   **Calidad y Especificidad:** Evalúa si los insights son específicos y accionables o genéricos.
        -   **Cobertura y Puntos Ciegos:** Analiza si los insights cubren todos los temas estratégicos o si hay áreas importantes sin insights.
        -   **Novedad vs. Redundancia:** Determina el ratio de insights novedosos frente a señales repetidas.
    2.  **Síntesis de Insights de Mayor Valor:** Destaca el insight más valioso y explica por qué es estratégicamente relevante.
    3.  **Recomendaciones para la Optimización del Agente:** Propón dos recomendaciones para mejorar la calidad de los insights en el futuro.

    **DATOS PARA EL ANÁLISIS (JSON):**
    ```json
    {agent_json}
    ```

    **INSTRUCCIÓN FINAL:**
    Genera el informe de auditoría siguiendo la estructura y el tono crítico y constructivo.
    """
    return prompt + ANALYTICAL_RIGOR_FOOTER

def get_deep_dive_mentions_prompt(topic: str, mentions: List[str]) -> str:
    """
    Nivel Élite: Análisis cualitativo para descubrir 'jobs to be done' del consumidor.
    """
    corpus = "\n\n".join([m.strip()[:4000] for m in mentions if isinstance(m, str)])
    
    prompt = (
        "**ROL:** Actúa como un Investigador de Diseño (Design Researcher) aplicando el framework 'Jobs to be Done'. Tu objetivo no es entender lo que los clientes dicen, sino lo que intentan *lograr*.\n\n"
        f"**TEMA A INVESTIGAR:** {topic}\n\n"
        "**CORPUS DE MENCIONES (Datos de Campo):**\n"
        f"{corpus}\n\n"
        "**METODOLOGÍA DE ANÁLISIS 'JOBS TO BE DONE':**\n"
        "1.  **Síntesis del Hallazgo:** ¿Cuál es el 'trabajo' (job) funcional, social o emocional que los clientes intentan hacer cuando hablan de este tema? Resume el progreso que buscan.\n"
        "2.  **Causa Raíz ('Forces of Progress'):** ¿Qué 'fuerzas' los impulsan a buscar una nueva solución (frustraciones, problemas) y qué 'fuerzas' los frenan (ansiedades, hábitos)?\n"
        "3.  **Citas Destacadas (Evidencia):** Extrae 3 citas que revelen claramente el 'trabajo a realizar' o las 'fuerzas' en juego.\n\n"
        "**FORMATO DE SALIDA (JSON ESTRICTO):**\n"
        "{\n"
        '  "sintesis_del_hallazgo": "Descripción del Job to be Done principal que emerge de las menciones.",\n'
        '  "causa_raiz": "Análisis de las fuerzas de empuje y freno que experimenta el cliente.",\n'
        '  "citas_destacadas": ["Cita que revela el progreso deseado", "Cita que muestra una frustración", "Cita que expone una ansiedad"]\n'
        "}"
    )
    return prompt + ANALYTICAL_RIGOR_FOOTER

def get_cluster_analyst_prompt(cluster: Dict[str, Any]) -> str:
    """
    Nivel Élite: Demanda un nombramiento y resumen que capture la esencia del cluster.
    """
    examples = [{"id": m.get("id"), "summary": (m.get("summary") or "")[:300], "sentiment": m.get("sentiment")} for m in cluster.get("example_mentions", [])]
    payload = {"examples": examples, "volume": cluster.get("count", 0), "avg_sentiment": cluster.get("avg_sentiment", 0.0), "top_sources": cluster.get("top_sources", [])}
    data_json = json.dumps(payload, ensure_ascii=False)
    
    prompt = f"""
    **ROL:** Eres un Analista Cualitativo experto en síntesis y conceptualización.

    **TAREA:** Analiza los siguientes datos de un cluster de conversación y destila su esencia.

    **DATOS DEL CLUSTER:**
    ```json
    {data_json}
    ```

    **INSTRUCCIONES:**
    1.  **Nombra el Tema:** Crea un `topic_name` corto y evocador que capture la idea central o la tensión del cluster. Evita nombres genéricos.
    2.  **Sintetiza los Puntos Clave:** Extrae 2-3 `key_points` que no sean un resumen, sino que representen las ideas, emociones o preguntas fundamentales que definen esta conversación.

    **FORMATO DE SALIDA (JSON ESTRICTO):**
    {{"topic_name": "Nombre evocador del tema", "key_points": ["Insight principal 1", "Insight principal 2"]}}
    """
    return prompt + ANALYTICAL_RIGOR_FOOTER

def get_clusters_synthesizer_prompt(clusters: List[Dict[str, Any]]) -> str:
    """
    Nivel Élite: Demanda la creación de una tesis estratégica a partir de los clusters.
    """
    brief = [{"topic_name": c.get("topic_name", "(sin nombre)"), "volume": c.get("volume", 0), "sentiment": c.get("sentiment", 0.0)} for c in clusters]
    data_json = json.dumps(brief, ensure_ascii=False)
    
    prompt = f"""
    **ROL:** Eres un Estratega de Marca y Mercado.

    **TAREA:** Sintetiza el siguiente resumen de temas de conversación (clusters) en una tesis estratégica coherente.

    **DATOS DE ENTRADA (Resumen de Clusters):**
    ```json
    {data_json}
    ```

    **MARCO DE SÍNTESIS ESTRATÉGICA:**
    1.  **Meta-Narrativas:** Identifica 2-3 narrativas o temas transversales que conecten varios de los clusters individuales. ¿Cuál es la historia más grande que se está contando?
    2.  **Oportunidad Principal:** ¿Cuál es la mayor oportunidad de mercado sin explotar que revelan estos clusters en su conjunto?
    3.  **Riesgo Inminente:** ¿Cuál es la amenaza más significativa o el riesgo más urgente que se desprende del análisis agregado?
    4.  **Plan Estratégico:** Propón 2-3 iniciativas estratégicas de alto nivel que respondan a las meta-narrativas, capitalicen la oportunidad y mitiguen el riesgo.

    **FORMATO DE SALIDA (JSON ESTRICTO):**
    {{
      "meta_narrativas": ["Narrativa transversal 1", "Narrativa transversal 2"],
      "oportunidad_principal": "Descripción de la oportunidad de mercado sin explotar.",
      "riesgo_inminente": "Descripción de la amenaza o riesgo más significativo.",
      "plan_estrategico": ["Iniciativa estratégica 1", "Iniciativa estratégica 2"]
    }}
    """
    return prompt + ANALYTICAL_RIGOR_FOOTER