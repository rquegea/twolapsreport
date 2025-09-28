import json

# Bloque común de reglas para rigor analítico y anti-alucinación
ANTI_HALLUCINATION_FOOTER = (
    "\n\n**REGLAS DE RIGOR (Obligatorias):**\n"
    "- Usa exclusivamente los datos proporcionados. No inventes cifras ni entidades.\n"
    "- Si falta un dato, escribe 'Dato no disponible'.\n"
    "- Cita la fuente del dato entre paréntesis (ej. 'KPIs', 'SOV por categoría').\n"
    "- Declara supuestos y posibles confusores; marca confianza de hipótesis: Alta|Media|Baja.\n"
)

DEFAULT_CLIENT_NAME = "Nuestra marca"

def get_detailed_category_prompt(category_name: str, category_data: list, all_summaries: list, kpis: dict) -> str:
    """
    Nivel Doctorado (Generalista): Exige un análisis causal y predictivo para la categoría, aplicable a cualquier mercado.
    """
    data_json = json.dumps(category_data, indent=2, ensure_ascii=False, default=str)
    summaries_text = "\n- ".join(list(set(all_summaries)))

    _sov_cat = kpis.get('sov_by_category', {}).get(category_name, {"client": 0, "total": 0})
    _client = float(_sov_cat.get('client', 0))
    _total = float(_sov_cat.get('total', 0))
    _sov_pct = (_client / _total * 100.0) if _total > 0 else 0.0

    prompt = f"""
    **ROL:** Eres un Director de Investigación de Mercados con un doctorado en comportamiento del consumidor y dinámicas de mercado. Tu análisis debe ser incisivo, profundo y contraintuitivo.

    **TAREA:** Realizar un diagnóstico estratégico para la categoría "{category_name}". No te limites a describir los datos; tu objetivo es desvelar la dinámica subyacente del mercado, identificar las causas raíz y formular una hipótesis estratégica validable.

    **CONTEXTO Analítico:**
    - **Categoría de Análisis:** "{category_name}"
    - **KPIs de Referencia (Mercado General):**
      - Menciones Totales: {kpis.get('total_mentions', 'N/A')}
      - Sentimiento Promedio General: {kpis.get('average_sentiment', 0.0):.2f}
    - **KPIs Específicos de esta Categoría:**
      - Share of Voice (SOV) en la Categoría: {_sov_pct:.2f}%

    **DATOS BRUTOS DE LA CATEGORÍA (para tu análisis fundamental):**
    ```json
    {data_json}
    ```

    **SÍNTESIS DE TEMAS GLOBALES (para identificar correlaciones y anomalías):**
    - {summaries_text}

    **INSTRUCCIONES DE ANÁLISIS AVANZADO:**
    1.  **Análisis Causal (El "Porqué"):** No te limites a decir "el sentimiento es bajo". Explica *por qué*. ¿Está correlacionado con menciones a competidores específicos? ¿Coincide con el lanzamiento de un producto? ¿Refleja una preocupación latente en el mercado (ej. precio, calidad, servicio al cliente)? Utiliza los datos para fundamentar tu hipótesis.
    2.  **Diagnóstico Competitivo:** Evalúa el SOV. Un {_sov_pct:.2f}% ¿es una posición de liderazgo, de desafío o de nicho? ¿Quién domina la conversación y con qué narrativa? Identifica una vulnerabilidad clave del líder en esta categoría que pueda ser explotada.
    3.  **Insight Estratégico y Recomendación Accionable:** Basado en tu análisis, destila UN insight principal y potente. A continuación, propón UNA recomendación estratégica concreta y original, no una generalidad.
        - **Ejemplo de mala recomendación:** "Mejorar el contenido en esta área".
        - **Ejemplo de buena recomendación:** "Lanzar una micro-campaña de 'thought leadership' enfocada en [NECESIDAD DETECTADA], utilizando [FORMATO ESPECÍFICO] para contrarrestar la narrativa de [ATRIBUTO] del competidor X, que actualmente domina el 60% de la conversación en esta categoría".

    **FORMATO DE SALIDA:**
    Redacta un párrafo de análisis denso y ejecutivo (aproximadamente 150-200 palabras) que siga la estructura lógica de tus instrucciones: diagnóstico causal, evaluación competitiva y conclusión con una recomendación estratégica de alto impacto.
    """
    return prompt


def get_main_analyst_prompt(aggregated_data: dict, global_mentions_corpus: list[str]) -> str:
    """
    Nivel Doctorado (Generalista): Demanda la creación de una narrativa estratégica unificada a partir de datos dispersos.
    """
    data_json = json.dumps(aggregated_data, ensure_ascii=False, indent=2, default=str)
    corpus = "\n\n".join([str(m)[:4000] for m in (global_mentions_corpus or []) if isinstance(m, str)])
    return (
        "**ROL:** Actúa como un Chief Strategy Officer (CSO). Tu especialidad es la síntesis de inteligencia de mercado cuantitativa y cualitativa para formular la narrativa estratégica que guiará las decisiones del próximo trimestre."
        "Tu audiencia es el comité de dirección; valora la claridad, la audacia y la justificación basada en datos.\n\n"
        "**MISIÓN:** Analiza la totalidad de los datos (KPIs, series temporales, SOV, temas y el corpus de menciones textuales) para construir un 'Dossier Estratégico Ejecutivo'. No eres un mero reportero de datos, eres un intérprete y un estratega. Tu entregable final debe ser un único objeto JSON que contenga esta narrativa estratégica.\n\n"
        "**MARCO DE ANÁLISIS ESTRATÉGICO:**\n"
        "1.  **Síntesis Holística:** Ve más allá de los datos individuales. Conecta los puntos: ¿cómo un pico en el volumen de menciones (dato cuantitativo) se explica por las conversaciones específicas encontradas en el corpus (dato cualitativo)? ¿Una caída en el sentimiento se correlaciona con la campaña de un competidor?\n"
        "2.  **Extracción de 'Second-Order Insights':** El primer insight es lo que el dato dice. El segundo es lo que el dato *significa* para el negocio. Enfócate en el segundo.\n"
        "3.  **Validación Cualitativa:** Usa el CORPUS GLOBAL como evidencia principal. Cada afirmación estratégica debe estar, implícita o explícitamente, respaldada por la voz del cliente extraída del corpus. Extrae citas que no solo sean representativas, sino que encapsulen la tensión o la emoción del momento.\n"
        "4.  **Identificación de Oportunidades y Amenazas Estratégicas:** Tu análisis debe culminar en la identificación de 2-3 temas que representen las mayores oportunidades o amenazas existenciales para la marca en el corto y medio plazo.\n\n"
        "**DATOS CUANTITATIVOS (JSON):**\n" + data_json + "\n\n"
        "**CORPUS GLOBAL DE MENCIONES (Evidencia Cualitativa Primaria):\n" + corpus + "\n\n"
        "**FORMATO DE SALIDA (JSON ESTRICTO - SIN EXCEPCIONES):\n"
        "{\n"
        '  "headline": "Titular estratégico que resuma la principal tensión u oportunidad del mercado en una frase impactante.",\n'
        '  "evaluacion_general": "Diagnóstico ejecutivo del periodo. ¿Estamos ganando o perdiendo terreno y por qué? ¿Cuál es la fuerza dominante (interna o externa) que ha marcado este periodo?",\n'
        '  "analisis_profundo": "Análisis detallado de las dinámicas clave. Conecta las series temporales con los temas emergentes y las acciones de la competencia. Formula hipótesis causales claras y argumentadas.",\n'
        '  "analisis_competencia": "Lectura estratégica del panorama competitivo. ¿Quién está ganando la narrativa y en qué frentes? ¿Dónde se encuentran las vulnerabilidades de nuestros competidores que podemos explotar?",\n'
        '  "analisis_mercado": "Análisis de las corrientes de fondo del mercado. ¿Qué necesidades no cubiertas o frustraciones emergen del corpus? ¿Qué tendencias son pasajeras y cuáles estructurales?",\n'
        '  "cualitativo_global": {\n'
        '    "sintesis_del_hallazgo": "El insight cualitativo más profundo extraído del corpus, redactado en 3-5 frases.",\n'
        '    "causa_raiz": "Tu hipótesis principal sobre la causa raíz del comportamiento observado en el mercado, basada en la evidencia del corpus.",\n'
        '    "citas_destacadas": ["Cita que encapsule la emoción principal", "Cita que revele una necesidad latente", "Cita que ilustre la percepción de un competidor"]\n'
        "  },\n"
        '  "deep_dive_temas": ["Tema 1: El más urgente/crítico", "Tema 2: El de mayor potencial a largo plazo"]\n'
        "}"
    )


def get_insight_extraction_prompt(aggregated_data: dict) -> str:
    """
    Nivel Doctorado (Generalista): Impone un marco de análisis riguroso para la extracción de insights.
    """
    data_json = json.dumps(aggregated_data, ensure_ascii=False, default=str)
    client_name = aggregated_data.get('client_name') or (aggregated_data.get('kpis', {}) or {}).get('brand_name') or DEFAULT_CLIENT_NAME
    return (
        f"**ROL:** Eres un Científico de Datos especializado en modelado predictivo y análisis causal. Tu tarea es deconstruir los datos de mercado para extraer insights estructurados y accionables, listos para ser consumidos por una IA estratega.\n\n"
        f"**MARCA OBJETIVO:** '{client_name}'. Todas las conclusiones deben gravitar en torno a esta marca.\n\n"
        "**DATOS DE ENTRADA:**\n"
        "```json\n" + data_json + "\n```\n\n"
        "**METODOLOGÍA DE ANÁLISIS (Obligatoria):**\n"
        "1.  **Análisis de Series Temporales:**\n"
        "-   **Tendencia y Volatilidad:** No solo describas la tendencia (ascendente/descendente), cuantifícala (ej. 'tendencia ascendente moderada de +0.15 en sentimiento') y califica su volatilidad.\n"
        "-   **Detección de Anomalías y Puntos de Inflexión:** Identifica los picos y valles más significativos. Para cada uno, especifica la fecha, la métrica, la magnitud del cambio (ej. '+30% en volumen') y la hipótesis causal más probable.\n"
        "-   **Análisis de Correlación (Sentimiento vs. Volumen):** Determina la naturaleza de la relación. ¿Es una correlación positiva (más ruido, mejor sentimiento), negativa (crisis) o no hay correlación aparente?\n"
        "2.  **Diagnóstico Estratégico:**\n"
        "-   **Hallazgos Clave:** Transforma los datos en tres conclusiones estratégicas fundamentales. Cada hallazgo debe responder a la pregunta: '¿Qué es lo más importante que el negocio necesita saber sobre el mercado, la competencia y sus propios clientes?'\n"
        "-   **Matriz de Oportunidades/Riesgos:** Identifica oportunidades y riesgos. Para cada uno, evalúa su **Impacto** (Alto, Medio, Bajo) y su **Probabilidad** (Alta, Media, Baja).\n"
        "-   **Recomendaciones Tácticas:** Deriva un conjunto de acciones concretas, directas y medibles a partir de tu análisis. Prohíbe las generalidades.\n\n"
        "**FORMATO DE SALIDA (JSON ESTRICTO - SIN TEXTO ADICIONAL):**\n\n"
        "{\n"
        '  "executive_summary": "Una síntesis de 2-3 frases que capture el insight más crítico y su implicación de negocio.",\n'
        '  "time_series_analysis": {\n'
        '    "global": {\n'
        '      "trend": "Descripción cuantificada de la tendencia y volatilidad.",\n'
        '      "anomalies": [ { "date": "YYYY-MM-DD", "type": "pico|valle", "metric": "menciones|sentimiento", "magnitude": "ej. +30% o -0.5", "causal_hypothesis": "Hipótesis explicativa" } ],\n'
        '      "sentiment_vs_volume_correlation": "Descripción de la correlación observada."\n'
        "    }\n"
        "  },\n"
        '  "key_findings": [\n'
        '    "Hallazgo 1 sobre el mercado.",\n'
        '    "Hallazgo 2 sobre la competencia.",\n'
        '    "Hallazgo 3 sobre la percepción de la marca."\n'
        "  ],\n"
        '  "opportunities": [ { "opportunity": "Descripción de la oportunidad", "impact": "Alto|Medio|Bajo", "probability": "Alta|Media|Baja" } ],\n'
        '  "risks": [ { "risk": "Descripción del riesgo", "mitigation": "Acción específica de mitigación", "impact": "Alto|Medio|Bajo", "probability": "Alta|Media|Baja" } ],\n'
        '  "recommendations": [ "Acción medible 1", "Acción medible 2", "Acción medible 3" ]\n'
        "}"
    ) + ANTI_HALLUCINATION_FOOTER


def get_strategic_summary_prompt(insights_json: dict, *, client_name: str) -> str:
    """
    Nivel Doctorado (Generalista): Enfocado en la redacción de un narrativo persuasivo y ejecutivo.
    """
    executive_summary = insights_json.get("executive_summary", "")
    key_findings = insights_json.get("key_findings", [])
    findings_text = "\n- ".join(key_findings) if key_findings else ""
    return f"""
    **ROL:** Eres un Comunicador Estratégico y Asesor de C-Level. Tu habilidad es transformar datos complejos en una narrativa clara, concisa y persuasiva que inspire a la acción.

    **TAREA:** Redactar la sección "Resumen Ejecutivo y Hallazgos Principales" de un informe estratégico para la marca **{client_name}**.

    **MATERIAL DE BASE:**
    - **Síntesis Ejecutiva Preliminar:** {executive_summary}
    - **Hallazgos Clave Aislados:**
      - {findings_text}

    **INSTRUCCIONES DE COMUNICACIÓN ESTRATÉGICA:**
    1.  **Crea una Narrativa, no una Lista:** No enumeres los hallazgos. Teje una historia coherente. Empieza con la conclusión más importante (el `executive_summary`) y luego utiliza los `key_findings` como los pilares argumentales que la sostienen.
    2.  **Enfoque '¿Y qué?':** Para cada hallazgo, responde implícitamente a la pregunta '¿Y qué significa esto para el negocio?'. Transforma cada punto de datos en una implicación estratégica.
    3.  **Lenguaje de Decisión:** Utiliza un lenguaje activo y directo. Evita la voz pasiva y la jerga técnica. El objetivo es facilitar la toma de decisiones, no demostrar la complejidad del análisis.
    4.  **Foco Absoluto en {client_name}:** Todo el análisis debe estar centrado en {client_name}. La competencia solo se menciona para dar contexto a la posición de nuestra marca.

    **FORMATO DE SALIDA:**
    Entre 2 y 3 párrafos de prosa ejecutiva. El texto debe fluir lógicamente y ser inmediatamente comprensible para un directivo con poco tiempo.
    """ + ANTI_HALLUCINATION_FOOTER

def get_strategic_plan_prompt(insights_json: dict, *, client_name: str | None = None) -> str:
    """
    Nivel Doctorado (Generalista): Exige un plan de acción riguroso, priorizado y justificado.
    """
    opportunities = insights_json.get("opportunities", [])
    risks = insights_json.get("risks", [])
    recommendations = insights_json.get("recommendations", [])
    opp_text = json.dumps(opportunities, ensure_ascii=False, indent=2)
    risks_text = json.dumps(risks, ensure_ascii=False, indent=2)
    recs_text = "\n- ".join(recommendations) if recommendations else ""
    return f"""
    **ROL:** Eres un Director de Estrategia y Operaciones. Tu responsabilidad es convertir los insights en un plan de acción ejecutable, priorizado y con una lógica de negocio impecable.

    **TAREA:** Elaborar la sección "Plan de Acción Estratégico" del informe.
    {('**MARCA:** ' + client_name + ' — Todas las acciones deben estar orientadas a fortalecer la posición de esta marca.' ) if client_name else ''}

    **INPUTS (Análisis Previo):**
    - **Oportunidades Identificadas:**
    ```json
    {opp_text}
    ```
    - **Riesgos Detectados:**
    ```json
    {risks_text}
    ```
    - **Recomendaciones Preliminares:**
    - {recs_text}

    **METODOLOGÍA DE PLANIFICACIÓN ESTRATÉGICA:**
    1.  **Priorización Basada en Impacto:** No todas las acciones son iguales. Prioriza las recomendaciones en función de qué oportunidades capitalizan y qué riesgos de alto impacto mitigan. Utiliza un marco implícito de impacto vs. esfuerzo.
    2.  **Conexión Lógica:** Cada acción propuesta debe estar explícitamente vinculada a un insight (una oportunidad o un riesgo). Justifica *por qué* cada acción es la respuesta correcta al dato analizado.
    3.  **Especificidad y Medición:** Transforma las recomendaciones genéricas en iniciativas concretas. Define el "qué" y el "porqué". Evita a toda costa las banalidades.
        - **PROHIBIDO:** "Mejorar la visibilidad", "Analizar a la competencia", "Aumentar el presupuesto".
        - **REQUERIDO:** "Lanzar una iniciativa de 'thought leadership' sobre [Tema X] para capturar el 15% del SOV en la categoría [Categoría Y] en los próximos 6 meses, mitigando el riesgo de irrelevancia frente al competidor Z".
    4.  **Secuenciación:** Organiza las acciones en una secuencia lógica (ej. corto, medio, largo plazo o por áreas funcionales: Marketing, Producto, etc.).

    **FORMATO DE SALIDA:**
    Un texto en prosa, articulado y profesional. Estructúralo en 2-3 párrafos que presenten el plan de forma coherente. NO devuelvas JSON, listas con guiones o encabezados markdown.
    """ + ANTI_HALLUCINATION_FOOTER

def get_executive_summary_prompt(aggregated_data: dict) -> str:
    """
    Nivel Doctorado (Generalista): Eleva el rol a Chief Strategy Officer, demandando una visión de futuro.
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

    return f"""
    **ROL:** Eres el Chief Strategy Officer (CSO) de la empresa. Tu audiencia es el Consejo de Administración. Esperan de ti una visión aguda, sintética y, sobre todo, orientada al futuro.

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

    **DIRECTRICES ESTRATÉGICAS PARA TU ANÁLISIS:**
    1.  **Diagnóstico (Pasado):** ¿Cuál es la historia principal que cuentan los datos de este periodo? ¿Ganamos o perdimos? ¿En qué campos de batalla (categorías)? Sé directo y concluyente.
    2.  **Implicación (Presente):** ¿Qué significa este diagnóstico para nuestra posición actual en el mercado? ¿Qué fortalezas debemos consolidar y qué debilidades son críticas?
    3.  **Prescripción (Futuro):** ¿Cuál es la recomendación estratégica número uno que el Consejo debe considerar? Debe ser una acción audaz que responda directamente al análisis.

    **FORMATO:** Redacta 2-3 párrafos densos en información, con un lenguaje claro y de alto nivel. Evita la jerga de marketing y enfócate en el impacto de negocio.
    """ + ANTI_HALLUCINATION_FOOTER

def get_competitive_analysis_prompt(aggregated_data: dict) -> str:
    """
    Nivel Doctorado (Generalista): Centrado en la explotación de vulnerabilidades competitivas.
    """
    kpis = aggregated_data.get('kpis', {}) or {}
    client_name = aggregated_data.get('client_name', DEFAULT_CLIENT_NAME)
    sov_total = float(kpis.get('share_of_voice') or kpis.get('sov') or 0.0)
    sov_table = kpis.get('sov_table') or []
    competitors = aggregated_data.get('market_competitors') or [str(b) for b, _ in sov_table if str(b) != client_name]
    sov_by_cat = (aggregated_data.get('sov', {}).get('current', {}).get('sov_by_category') or kpis.get('sov_by_category') or {})
    competitor_mentions = (aggregated_data.get('sov', {}).get('current', {}).get('competitor_mentions') or {str(b): float(v) for b, v in sov_table})
    comp_json = json.dumps(competitor_mentions, ensure_ascii=False, indent=2)
    sov_cat_json = json.dumps(sov_by_cat, ensure_ascii=False, indent=2)

    return (
        f"""
    **ROL:** Eres un Analista de Inteligencia Competitiva, experto en identificar y explotar las debilidades del adversario.

    **TAREA:** Redactar un informe de "Análisis Competitivo y Estrategias de Ataque" para {client_name}.

    **DOSSIER DE INTELIGENCIA:**
    - Cliente: {client_name}
    - Adversarios: {', '.join(competitors) if competitors else 'N/D'}
    - SOV del Cliente: {sov_total:.2f}%
    - Mapa de Territorios (SOV por categoría):
    ```json
    {sov_cat_json}
    ```
    - Actividad del Adversario (Menciones):
    ```json
    {comp_json}
    ```

    **PROTOCOLO DE ANÁLISIS:**
    1.  **Identificar al Líder y sus Flancos Débiles:** ¿Quién domina la conversación general? Más importante aún, ¿en qué categoría temática su dominio es más frágil o su sentimiento es negativo a pesar del alto volumen? Ese es su flanco débil.
    2.  **Detectar "Movimientos de Guerrilla":** ¿Hay algún competidor más pequeño que esté ganando una cuota desproporcionada en una categoría nicho? Analiza su estrategia.
    3.  **Formular Estrategias de "Flanqueo":** Basado en lo anterior, define 3 movimientos estratégicos recomendados para {client_name}. No te limites a "atacar", piensa en cómo "flanquear": ganar donde el competidor no está mirando o no es fuerte. Cada recomendación debe ser un bullet point claro y accionable.

    **FORMATO:** Un análisis conciso que identifique al líder, sus debilidades y proponga 3 estrategias de flanqueo claras.
    """ + ANTI_HALLUCINATION_FOOTER
    )


def get_deep_dive_analysis_prompt(category_name: str, kpis: dict, client_name: str) -> str:
    """
    Nivel Doctorado (Generalista): Exige una inmersión profunda en la psicología del consumidor de la categoría.
    """
    cat_kpis = kpis.get('sentiment_by_category', {}).get(category_name, {})
    sentiment_avg = cat_kpis.get('average', 0.0)
    distribution = cat_kpis.get('distribution', {})
    key_topics = cat_kpis.get('key_topics', {})
    dist_text = json.dumps(distribution, indent=2, ensure_ascii=False)
    topics_text = json.dumps(key_topics, indent=2, ensure_ascii=False)

    return f"""
    **ROL:** Eres un Investigador Cualitativo Senior con experiencia en etnografía digital, analizando para {client_name}.

    **TAREA:** Realizar un "Análisis Psicológico Profundo" de la categoría "{category_name}". Tu objetivo es ir más allá de los datos para entender las motivaciones, frustraciones y necesidades no expresadas del consumidor.

    **DATOS CUANTITATIVOS PARA CONTEXTUALIZAR:**
    - Sentimiento Promedio (Categoría vs. General): {sentiment_avg:.2f} vs. {kpis.get('average_sentiment', 0.0):.2f}
    - Distribución del Sentimiento:
    {dist_text}
    - Temas de Conversación (Frecuencia):
    {topics_text}

    **GUÍA DE ANÁLISIS PROFUNDO:**
    1.  **El Significado del Sentimiento:** ¿Qué emoción subyace a la métrica de sentimiento? Si es negativo, ¿es frustración, decepción, enfado? Si es positivo, ¿es satisfacción, entusiasmo, lealtad? La distribución (polarizada, neutral) es una pista clave.
    2.  **La Tensión en los Temas:** ¿Qué conflicto o tensión revelan los temas clave? Por ejemplo, una alta frecuencia de "precio" y "calidad" sugiere una tensión en torno al valor percibido.
    3.  **El Insight Oculto y la Oportunidad:** Conecta la emoción (sentimiento) con la tensión (temas) para desvelar una necesidad no satisfecha del consumidor. A partir de este insight, formula una oportunidad de innovación para {client_name} (de producto, comunicación o servicio).

    **FORMATO:** Un párrafo de análisis narrativo que revele la psicología del consumidor en esta categoría y concluya con una oportunidad de innovación clara.
    """ + ANTI_HALLUCINATION_FOOTER

def get_correlation_interpretation_prompt(aggregated_data: dict, correlation_data: dict) -> str:
    """
    Nivel Doctorado (Generalista): Enfocado en la validación de hipótesis y la causalidad.
    """
    data_json = json.dumps(correlation_data, indent=2, ensure_ascii=False)
    client_name = aggregated_data.get('client_name', DEFAULT_CLIENT_NAME)
    return f"""
    **ROL:** Eres un Científico de Datos especializado en inferencia causal.

    **TAREA:** Interpretar los datos de correlación para {client_name}, distinguir causalidad de casualidad y proponer experimentos para validar tus hipótesis.

    **DATOS DE CORRELACIÓN:**
    ```json
    {data_json}
    ```

    **METODOLOGÍA DE ANÁLISIS CAUSAL:**
    1.  **Identificar Correlaciones Significativas:** Filtra y enfócate en las 2-3 relaciones con la mayor fuerza ("strength").
    2.  **Formular Hipótesis Causales:** Para cada correlación, propón una hipótesis de relación causa-efecto. (Ej. "Un aumento en las menciones de la categoría 'Eventos' CAUSA un aumento en el sentimiento general").
    3.  **Proponer un Plan de Validación:** Para cada hipótesis, diseña un "experimento" o un análisis de datos que podría validarla o refutarla. (Ej. "Para validar la hipótesis, analizar si el sentimiento de los asistentes a eventos específicos es significativamente mayor que el de la media durante las 24h posteriores al evento").
    4.  **Implicación de Negocio:** Si la hipótesis causal fuera cierta, ¿cuál sería la implicación estratégica para {client_name}?

    **FORMATO:** Redacta 2-3 párrafos. Cada párrafo debe presentar una correlación, la hipótesis causal, el plan de validación y la implicación estratégica.
    """ + ANTI_HALLUCINATION_FOOTER


def get_trends_anomalies_prompt(aggregated_data: dict) -> str:
    """
    Nivel Doctorado (Generalista): Análisis de tendencias con enfoque en la velocidad del cambio (momentum).
    """
    trends = aggregated_data.get('trends', {})
    prev = aggregated_data.get('previous_period', {})
    trends_json = json.dumps(trends, indent=2, ensure_ascii=False)
    return f"""
    **ROL:** Eres un Analista de Tendencias y Futuros (Futures Analyst).

    **TAREA:** Redactar la sección "Tendencias, Anomalías y Momentum del Mercado".

    **DATOS DE CAMBIO (Periodo Actual vs. {prev.get('start_date', 'N/D')} - {prev.get('end_date', 'N/D')}):**
    ```json
    {trends_json}
    ```

    **FRAMEWORK DE ANÁLISIS DE TENDENCIAS:**
    1.  **Magnitud y Velocidad:** No solo identifiques qué ha cambiado, sino la magnitud y la velocidad del cambio. ¿Es un cambio lento y gradual o una disrupción repentina?
    2.  **Análisis de 'Drivers':** ¿Qué fuerzas están impulsando estas tendencias? ¿Son movimientos de la competencia, cambios en el comportamiento del consumidor, factores macroeconómicos?
    3.  **De Señal Débil a Tendencia Fuerte:** Para los tópicos emergentes, evalúa su potencial. ¿Es una "señal débil" (novedad pasajera) o tiene el potencial de convertirse en una "tendencia fuerte" (cambio estructural)? Justifica tu evaluación.
    4.  **Implicaciones Estratégicas y Tácticas:** Concluye con 2 recomendaciones estratégicas (a 12-18 meses) y 3 tácticas inmediatas (a 30-60 días) basadas en tu análisis.

    **FORMATO:** Un análisis ejecutivo que siga el framework descrito.
    """ + ANTI_HALLUCINATION_FOOTER

def get_agent_insights_summary_prompt(data: dict) -> str:
    """
    Nivel Doctorado (Generalista): Meta-análisis sobre la calidad de los insights generados.
    """
    agent_json = json.dumps(data.get("agent_insights", {}), ensure_ascii=False, indent=2)
    return f"""
    **Rol y Objetivo:**
    Actúa como un Auditor de Sistemas de Inteligencia Artificial. Tu misión es realizar un meta-análisis de los insights generados por el "Answer Engine" para evaluar su rendimiento, identificar sesgos y proponer mejoras.

    **Estructura Requerida del Informe de Auditoría:**
    1.  **Diagnóstico del Rendimiento del Agente (3 bullets):**
        -   **Calidad y Especificidad:** Evalúa si los insights (oportunidades, riesgos) son específicos y accionables o genéricos y banales.
        -   **Cobertura y Puntos Ciegos:** Analiza si los insights cubren de manera equilibrada todos los temas estratégicos o si hay "puntos ciegos" (áreas importantes sin insights).
        -   **Novedad vs. Redundancia:** Determina el ratio de insights novedosos frente a señales repetidas o redundantes.
    2.  **Síntesis de Insights de Mayor Valor:** Destaca el insight (oportunidad o riesgo) más valioso detectado por el agente y explica por qué es estratégicamente relevante.
    3.  **Recomendaciones para la Optimización del Agente:** Propón dos recomendaciones para mejorar la calidad de los insights en el futuro (ej. "Refinar los prompts de 'riesgos' para que se enfoquen en amenazas competitivas y no solo en quejas de clientes").

    **Datos para el Análisis (JSON):**
    ```json
    {agent_json}
    ```

    **Instrucción Final:**
    Genera el informe de auditoría siguiendo estrictamente la estructura y el tono crítico y constructivo descritos.
    """ + ANTI_HALLUCINATION_FOOTER

def get_deep_dive_mentions_prompt(topic: str, mentions: list[str]) -> str:
    """
    Nivel Doctorado (Generalista): Análisis cualitativo para descubrir 'jobs to be done'.
    """
    corpus = "\n\n".join([m.strip()[:4000] for m in mentions if isinstance(m, str)])
    return (
        "**ROL:** Actúa como un Investigador de Diseño (Design Researcher) aplicando el framework 'Jobs to be Done'. Tu objetivo no es entender lo que los clientes dicen, sino lo que intentan *lograr*.\n\n"
        f"**TEMA A INVESTIGAR:** {topic}\n\n"
        "**CORPUS DE MENCIONES (Datos de Campo):**\n"
        f"{corpus}\n\n"
        "**METODOLOGÍA DE ANÁLISIS 'JOBS TO BE DONE':**\n"
        "1.  **Síntesis del Hallazgo:** ¿Cuál es el 'trabajo' (job) funcional, social o emocional que los clientes intentan hacer cuando hablan de este tema? Resume el progreso que buscan en sus vidas.\n"
        "2.  **Causa Raíz ('Forces of Progress'):** ¿Qué 'fuerzas' los impulsan a buscar una nueva solución (frustraciones, problemas con la solución actual) y qué 'fuerzas' los frenan (ansiedades, hábitos)?\n"
        "3.  **Citas Destacadas (Evidencia):** Extrae 3 citas que no solo sean representativas, sino que revelen claramente el 'trabajo a realizar' o las 'fuerzas' en juego.\n\n"
        "**FORMATO DE SALIDA (JSON ESTRICTO):**\n"
        "{\n"
        '  "sintesis_del_hallazgo": "Descripción del Job to be Done principal que emerge de las menciones.",\n'
        '  "causa_raiz": "Análisis de las fuerzas de empuje y freno que experimenta el cliente.",\n'
        '  "citas_destacadas": ["Cita que revela el progreso deseado", "Cita que muestra una frustración", "Cita que expone una ansiedad"]\n'
        "}"
    ) + ANTI_HALLUCINATION_FOOTER

def get_cluster_analyst_prompt(cluster: dict) -> str:
    """
    Nivel Doctorado (Generalista): Demanda un nombramiento y resumen que capture la esencia del cluster.
    """
    examples = [{"id": m.get("id"), "summary": (m.get("summary") or "")[:300], "sentiment": m.get("sentiment")} for m in cluster.get("example_mentions", [])]
    payload = {"examples": examples, "volume": cluster.get("count", 0), "avg_sentiment": cluster.get("avg_sentiment", 0.0), "top_sources": cluster.get("top_sources", [])}
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""
    **ROL:** Eres un Analista Cualitativo experto en síntesis y conceptualización.

    **TAREA:** Analiza los siguientes datos de un cluster de conversación y destila su esencia.

    **DATOS DEL CLUSTER:**
    ```json
    {data_json}
    ```

    **INSTRUCCIONES:**
    1.  **Nombra el Tema:** Crea un `topic_name` que sea corto, evocador y que capture la idea central o la tensión del cluster. Evita nombres genéricos.
    2.  **Sintetiza los Puntos Clave:** Extrae 2-3 `key_points` que no sean un resumen de los ejemplos, sino que representen las ideas, emociones o preguntas fundamentales que definen este cluster de conversación.

    **FORMATO DE SALIDA (JSON ESTRICTO):**
    {{"topic_name": "Nombre evocador del tema", "key_points": ["Insight principal 1", "Insight principal 2"]}}
    """ + ANTI_HALLUCINATION_FOOTER


def get_clusters_synthesizer_prompt(clusters: list[dict]) -> str:
    """
    Nivel Doctorado (Generalista): Demanda la creación de una tesis estratégica a partir de los clusters.
    """
    brief = [{"topic_name": c.get("topic_name", "(sin nombre)"), "volume": c.get("volume", 0), "sentiment": c.get("sentiment", 0.0)} for c in clusters]
    data_json = json.dumps(brief, ensure_ascii=False)
    return f"""
    **ROL:** Eres un Estratega de Marca y Mercado.

    **TAREA:** Sintetiza el siguiente resumen de temas de conversación (clusters) en una tesis estratégica coherente.

    **DATOS DE ENTRADA (Resumen de Clusters):**
    ```json
    {data_json}
    ```

    **MARCO DE SÍNTESIS ESTRATÉGICA:**
    1.  **Meta-Narrativas:** Identifica 2-3 narrativas o temas transversales que conecten varios de los clusters individuales. ¿Cuál es la historia más grande que se está contando a través de estos temas?
    2.  **Oportunidad Principal:** ¿Cuál es la mayor oportunidad de mercado sin explotar que revelan estos clusters en su conjunto?
    3.  **Riesgo Inminente:** ¿Cuál es la amenaza más significativa o el riesgo más urgente que se desprende del análisis agregado?
    4.  **Plan Estratégico:** Propón 2-3 iniciativas estratégicas de alto nivel que respondan directamente a las meta-narrativas, capitalicen la oportunidad principal y mitiguen el riesgo inminente.

    **FORMATO DE SALIDA (JSON ESTRICTO):**
    {{
      "meta_narrativas": ["Narrativa transversal 1", "Narrativa transversal 2"],
      "oportunidad_principal": "Descripción de la oportunidad de mercado sin explotar.",
      "riesgo_inminente": "Descripción de la amenaza o riesgo más significativo.",
      "plan_estrategico": ["Iniciativa estratégica 1", "Iniciativa estratégica 2"]
    }}
    """ + ANTI_HALLUCINATION_FOOTER