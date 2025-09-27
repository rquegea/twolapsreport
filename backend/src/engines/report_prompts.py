from typing import Any
import json
from . import prompts as catalog


def get_executive_summary_prompt(data: dict[str, Any]):
    """
    Genera un prompt para crear un resumen ejecutivo de alto nivel.
    - Evita KeyError si falta 'summary' en los datos agregados.
    - Compone un bloque de contexto compacto a partir de KPIs y tablas si no hay 'summary'.
    """
    try:
        brand = (
            (data or {}).get("brand")
            or (data or {}).get("client_name")
            or (data.get("kpis", {}) if isinstance(data, dict) else {}).get("brand_name")
            or "La Marca"
        )

        # Construir una representación segura del contenido para el análisis
        raw_summary = (data or {}).get("summary") if isinstance(data, dict) else None
        if isinstance(raw_summary, str) and raw_summary.strip():
            summary_block = raw_summary.strip()
        else:
            # Seleccionar solo lo más relevante para no enviar todo el objeto enorme
            minimal_context = {
                "brand": brand,
                "period": {
                    "start": (data or {}).get("start_date"),
                    "end": (data or {}).get("end_date"),
                },
                "kpis": (data or {}).get("kpis"),
                "time_series": (data or {}).get("time_series"),
                "competitor_ranking": (data or {}).get("competitor_ranking"),
                "top_positive_themes": (data or {}).get("top_positive_themes"),
                "top_negative_themes": (data or {}).get("top_negative_themes"),
                "emerging_trends": (data or {}).get("emerging_trends"),
                "top_opportunities": (data or {}).get("top_opportunities"),
                "top_risks": (data or {}).get("top_risks"),
                "key_quotes": (data or {}).get("key_quotes"),
            }
            summary_block = json.dumps(minimal_context, ensure_ascii=False)

        return f"""
    **Rol y Objetivo:**
    Actúa como Director de Insights y Estrategia en una consultora de élite como McKinsey o BCG. Tu audiencia es el C-Suite (CEO, CMO) de '{brand}'. Tu misión es destilar todos los datos proporcionados en un resumen ejecutivo que sea a la vez informativo y prescriptivo. No te limites a describir los datos; interprétalos para contar una historia convincente sobre la posición de la marca en el mercado.

    **Estructura Requerida:**
    1.  **Headline (Titular Impactante):** Un titular de una sola frase que resuma el hallazgo más crítico del periodo. Debe ser audaz y directo.
        * *Ejemplo de buen titular:* "A pesar de una percepción de marca positiva, {brand} se enfrenta a una brecha de visibilidad relevante frente a su principal competidor."
        * *Ejemplo de mal titular:* "Resumen de datos del periodo."

    2.  **Hallazgos Clave (3 bullets):** Tres puntos concisos que resuman las conclusiones más importantes. Cada punto debe ser una frase completa que combine un dato con una implicación de negocio.

    3.  **Evaluación General y Conclusión:** Un párrafo corto (2-3 frases) que ofrezca un diagnóstico general y una conclusión estratégica clara. ¿Cuál es la situación general y qué se debe hacer al respecto?

    **Datos para el Análisis:**
    {summary_block}

    **Instrucción Final:**
    Genera el resumen ejecutivo siguiendo estrictamente la estructura y el tono descritos. Sé implacable en tu análisis y no tengas miedo de señalar tanto las oportunidades como las amenazas críticas.
    """
    except Exception:
        # Fallback muy defensivo
        safe = json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else str(data)
        return f"Resumen ejecutivo para contexto: {safe}"


def get_deep_dive_analysis_prompt(data: dict[str, Any]):
    """Analista de Datos Senior: Análisis granular y correlaciones.
    Incluye extractos (si hay) para ilustrar los hallazgos.
    """
    corpus_note = ""
    if isinstance(data, dict) and data.get("corpus_for_llm"):
        corpus_note = f"\n\n**EXTRACTOS TEXTUALES (muestra):** {json.dumps(data.get('corpus_for_llm'), ensure_ascii=False)}\n"
    return f"""
    **ROL:** Eres un Analista de Datos Senior especializado en inteligencia de mercado. Tu misión es realizar un 'deep dive' en los datos granulares para descubrir insights ocultos.
    **TAREA:** Analiza las tablas desglosadas y la comparativa de sentimiento para extraer conclusiones profundas.
    **DATOS:** {json.dumps(data, indent=2, ensure_ascii=False)}{corpus_note}
    **RESPONDE ÚNICAMENTE CON ESTE JSON:**
    {{
      "deep_dive_analysis": {{
        "title": "Análisis Profundo de Visibilidad y Sentimiento",
        "visibility_by_model": "Lectura con apoyo en datos y, si procede, breve cita",
        "visibility_by_topic": "Lectura con apoyo en datos y, si procede, breve cita",
        "sentiment_comparison": "Lectura con apoyo en datos y, si procede, breve cita"
      }}
    }}
    """


def get_recommendations_prompt(data: dict[str, Any]):
    """Estratega y Planificador: convierte insights en un plan de acción SMART.
    Puede usar citas para reforzar el porqué de cada acción.
    """
    corpus_note = ""
    if isinstance(data, dict) and data.get("corpus_for_llm"):
        corpus_note = f"\n\n**EXTRACTOS TEXTUALES (muestra):** {json.dumps(data.get('corpus_for_llm'), ensure_ascii=False)}\n"
    return f"""
    **ROL:** Eres un Consultor de Estrategia Senior de BCG/Bain. Tu trabajo es destilar todo el análisis previo en un plan de acción ejecutivo, claro y medible.

    **TAREA:** Define un Plan de Acción Estratégico con formato SMART.

    **DATOS COMPLETOS DEL ANÁLISIS:** {json.dumps(data, indent=2, ensure_ascii=False)}{corpus_note}

    **RESPONDE ÚNICAMENTE CON ESTE FORMATO JSON:**
    {{
      "strategic_action_plan": {{
        "title": "Plan de Acción Estratégico",
        "market_outlook": "Breve pronóstico basado en datos y señales cualitativas.",
        "strategic_recommendations": [
          {{
            "recommendation": "Recomendación 1",
            "details": "Justificación basada en datos y extracto si procede.",
            "kpis": "1-2 KPIs",
            "timeline": "Plazo",
            "priority": "Alta"
          }}
        ]
      }}
    }}
    """


def get_methodology_prompt():
    """Generador de texto para el apéndice."""
    return """
    **ROL:** Eres un experto en metodología de datos para una consultora.
    **TAREA:** Redacta un texto breve y claro para el apéndice de un informe que explique la metodología utilizada.
    **RESPONDE ÚNICAMENTE CON ESTE JSON:**
    {{
        "methodology": {{
            "title": "Apéndice: Metodología",
            "text": "El presente informe se basa en un análisis cuantitativo y cualitativo..."
        }}
    }}
    """


def get_correlation_anomalies_prompt(data: dict[str, Any]):
    """
    Genera un prompt para identificar correlaciones y anomalías entre diferentes temas.
    Es robusto si faltan claves: usa eventos de correlación y la matriz tema-sentimiento si están disponibles.
    """
    try:
        context = {
            "topic_sentiment_correlations": (data or {}).get("topic_sentiment_correlations"),
            "correlation_events": (data or {}).get("correlation_events"),
        }
        corr_json = json.dumps(context, ensure_ascii=False)
        return f"""
    **Rol y Objetivo:**
    Actúas como un Científico de Datos especializado en descubrir insights ocultos. Tu misión es analizar las correlaciones entre los temas y el sentimiento en diferentes contextos. Busca patrones inesperados, anomalías y relaciones causa-efecto que no son obvias a primera vista. El objetivo es generar una hipótesis estratégica única.

    **Tarea Específica:**
    1.  Observa los datos de sentimiento por tema y las citas destacadas.
    2.  Identifica una correlación o una anomalía que te parezca significativa. Por ejemplo:
        * ¿Un tema que es muy positivo en general se vuelve negativo cuando se asocia a un competidor específico?
        * ¿Hay dos temas aparentemente no relacionados que siempre aparecen juntos con un sentimiento similar?
        * ¿Una preocupación de los padres (sentimiento negativo) está directamente correlacionada con la conversación sobre el coste (sentimiento neutro)?
    3.  Formula una **hipótesis estratégica** basada en tu hallazgo.
    4.  Propón **una acción de contenido o de marketing** para validar o explotar esta hipótesis.

    **Datos para el Análisis:**
    {corr_json}

    **Instrucción Final:**
    No te limites a describir los datos. Tu valor reside en ir más allá de lo evidente y proponer una idea original basada en las conexiones que descubras.
    """
    except Exception:
        return "Analiza posibles correlaciones entre temas y sentimiento; devuelve una hipótesis y una acción." 


def get_competitive_analysis_prompt(data: dict[str, Any]):
    """
    Genera un prompt para el análisis del panorama competitivo con valores por defecto seguros.
    """
    brand = (data or {}).get("brand") or (data or {}).get("client_name") or "La Marca"
    leader_brand = (data or {}).get("sov_leader_name") or (data or {}).get("leader_brand") or "Líder"
    sov_summary = json.dumps((data or {}).get("sov_summary") or (data or {}).get("kpis") or {}, ensure_ascii=False)
    granular_sov = json.dumps((data or {}).get("sov_by_topic") or (data or {}).get("kpis", {}).get("sov_by_category") or {}, ensure_ascii=False)
    sentiment_summary = json.dumps((data or {}).get("sentiment_comparison") or {}, ensure_ascii=False)
    key_mentions = json.dumps((data or {}).get("content_competitive") or (data or {}).get("key_mentions") or [], ensure_ascii=False)
    return f"""
    **Rol y Objetivo:**
    Eres un Analista Senior de Inteligencia Competitiva. Tu tarea es analizar el posicionamiento de '{brand}' frente a sus competidores clave, basándote en los datos de Share of Voice (SOV) y sentimiento. Tu análisis debe ser agudo, identificando no solo quién lidera, sino por qué y dónde residen las oportunidades.

    **Estructura Requerida:**
    1.  **Posicionamiento General:** Un párrafo que describa la posición de '{brand}' en el mercado. ¿Es un líder, un retador, un nicho? Usa los datos de SOV para justificar tu afirmación.
    2.  **Análisis del Líder ({leader_brand}):** Describe por qué el líder domina la conversación. ¿En qué temas o narrativas es más fuerte? ¿Cuál es la implicación estratégica para '{brand}'?
    3.  **Oportunidades frente a Debilidades de Competidores:** Analiza la tabla de 'Debilidades Detectadas' y propón cómo '{brand}' puede explotarlas. Sé específico en tus recomendaciones.

    **Datos para el Análisis:**
    - SOV General: {sov_summary}
    - SOV por Tema: {granular_sov}
    - Sentimiento por Marca: {sentiment_summary}
    - Debilidades de competidores: {key_mentions}

    **Instrucción Final:**
    Tu análisis debe concluir con una recomendación clara sobre dónde enfocar los esfuerzos para ganar cuota de conversación al líder.
    """

