"""
======================================================================
           CATÁLOGO CENTRAL DE PROMPTS PARA ANÁLISIS DE IA
======================================================================
"""

# --- ANÁLISIS DE COMPETENCIA ---

IDENTIFY_COMPETITOR_WEAKNESSES = """
Eres un estratega de marketing. Analiza las siguientes menciones sobre '{competitor_name}'.
Identifica 3 debilidades clave y sugiere una acción de contenido para cada una.
MENCIONES: {mentions_text}
RESPUESTA JSON: {{"debilidades": [{{"punto": str, "accion": str}}]}}
"""


# --- ANÁLISIS DE MERCADO ---

SUMMARIZE_MARKET_TRENDS = """
A partir de estas menciones del mercado, identifica las 3 principales tendencias emergentes.
MENCIONES: {market_mentions}
TENDENCIAS: lista con 3 ítems, breve explicación.
"""


# --- ANÁLISIS DE RIESGOS ---

DETECT_POTENTIAL_RISKS = """
Analiza el siguiente texto en busca de posibles riesgos reputacionales, financieros o de mercado.
TEXTO: "{text}"
Devuelve una lista de riesgos con tipo y breve justificación.
"""


# --- ANÁLISIS DE SENTIMIENTO ---

SENTIMENT_ANALYSIS_JSON = """
Eres un analista experto en sentimiento en ESPAÑOL.

INSTRUCCIONES CLAVE (escala de −1 a 1):
- Muy positivo: elogios explícitos, resultados excelentes, mejora clara → 0.6 a 1.0
- Positivo: valoración favorable, utilidad, ventajas → 0.2 a 0.6
- Neutral: información factual sin valoración → −0.2 a 0.2
- Negativo: críticas, problemas, quejas, retrocesos → −0.6 a −0.2
- Muy negativo: rechazo fuerte, fracaso, daño → −1.0 a −0.6

IMPORTANTE: Palabras o contextos como "incertidumbre", "indecisión", "preocupación",
"riesgo", "caída", "descenso", "empeora", "duda", "temor", "crítica" deben reducir el score
y NUNCA devolver un valor positivo. Si el tono es ambiguo pero con preocupación/indecisión,
clasifícalo como negativo leve (≈ −0.2 a −0.4) o neutral.

EJEMPLOS RÁPIDOS:
- "Los estudiantes están indecisos y aumenta la incertidumbre" → sentiment ≈ −0.3
- "Resultados récord y gran satisfacción" → sentiment ≈ 0.7
- "Se anuncian cambios sin indicar impacto" → sentiment ≈ 0.0

Devuelve SOLO este JSON exacto (sin texto adicional):
{{"sentiment": 0.8, "emotion": "alegría", "confidence": 0.9}}

Donde:
- sentiment: número entre -1 (muy negativo) y 1 (muy positivo)
- emotion: alegría, tristeza, enojo, miedo, sorpresa, neutral
- confidence: número entre 0 y 1

Texto a analizar:
{text}
"""


