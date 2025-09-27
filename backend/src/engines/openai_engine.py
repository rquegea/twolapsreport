# src/engines/openai_engine.py
"""
Wrapper de utilidades para la OpenAI Python >= 1.0.
Expone dos funciones:

    • fetch_response()    → texto “crudo” del modelo
    • extract_insights()  → JSON rico para dashboards

Todas las llamadas usan la nueva sintaxis v1 (`client.chat.completions.create`).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Tuple

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
import httpx

# ───────────────────────── Config ──────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DEFAULT_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "25"))
MAX_INPUT_CHARS = int(os.getenv("OPENAI_MAX_INPUT_CHARS", "60000"))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def _truncate_input(text: str) -> str:
    """Recorta el contenido de entrada si excede el máximo permitido para evitar errores de contexto.
    Mantiene el inicio y el final (head/tail) para preservar señales.
    """
    try:
        if not isinstance(text, str):
            return text
        if len(text) <= MAX_INPUT_CHARS:
            return text
        head = text[: MAX_INPUT_CHARS // 2]
        tail = text[-(MAX_INPUT_CHARS // 2) :]
        logger.warning("✂️ Truncando prompt de %s a %s caracteres para ajustarse al límite.", len(text), MAX_INPUT_CHARS)
        return head + "\n\n[...contenido truncado por longitud...]\n\n" + tail
    except Exception:
        return text[:MAX_INPUT_CHARS]


# ─────────────────── Funciones del Engine ──────────────────
def fetch_response(
    prompt: str,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 1_024,
) -> str:
    """
    Envía un prompt y devuelve la respuesta textual del modelo.
    Usa gpt-4o-mini por defecto por ser rápido y económico.
    """
    try:
        user_content = prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)
        if not user_content or user_content.strip() == "":
            logger.error("fetch_response: prompt vacío o None; evitando llamada al modelo.")
            return ""
        user_content = _truncate_input(user_content)
        res = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un analista especializado en educación superior y captación de alumnos. "
                        "Conoces a 'The Core School' en Madrid: escuela superior de entretenimiento y artes audiovisuales "
                        "(cine, videojuegos, animación, producción). Prioriza exactitud, JSON válido y contexto de negocio."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=DEFAULT_TIMEOUT,
        )
        answer: str = res.choices[0].message.content.strip()
        return answer
    except (OpenAIError, httpx.TimeoutException, TimeoutError) as exc:
        logger.exception("❌ OpenAI API error en fetch_response: %s", exc)
        return ""


def extract_insights(text: str) -> Dict[str, Any]:
    """
    Analiza el CONTENIDO y devuelve un JSON listo para la tabla `insights`.
    Utiliza un modelo más potente (gpt-4o) para asegurar alta calidad en el análisis.
    """
    prompt = f"""
Contexto de Negocio: Estás analizando datos para 'The Core School', escuela superior en Madrid, especializada en
entretenimiento y artes audiovisuales (cine, videojuegos, animación, producción). Interpreta las consultas pensando en
captación de alumnos, reputación de marca y posicionamiento en el sector educativo.

Eres un **analista senior de inteligencia de mercado**. Tu tarea es destilar el siguiente contenido en insights estructurados.

**INSTRUCCIONES:**
1.  Lee el CONTENIDO y extrae los insights clave.
2.  Para "opportunities" y "risks", añade una clave "impact" con valor "Alto", "Medio" o "Bajo". Basa el impacto en la urgencia, el coste potencial o el beneficio mencionado.
3.  Sé conciso y directo.

**FORMATO JSON ESTRICTO (SOLO EL JSON):**
{{
  "brands": [{{"name": "...", "mentions": <int>, "sentiment_avg": <float>}}],
  "competitors": ["...", "..."],
  "opportunities": [{{"opportunity": "Descripción de la oportunidad.", "impact": "Alto"}}, {{"...": "..."}}],
  "risks": [{{"risk": "Descripción del riesgo.", "impact": "Bajo"}}, {{"...": "..."}}],
  "pain_points": ["...", "..."],
  "trends": ["...", "..."],
  "quotes": ["...", "..."],
  "top_themes": ["...", "..."],
  "topic_frequency": {{}},
  "source_mentions": {{}},
  "calls_to_action": ["...", "..."],
  "audience_targeting": ["...", "..."],
  "products_or_features": ["...", "..."]
}}

No añadas texto fuera del JSON.
----------
CONTENIDO:
{text}
----------
"""
    # Usamos gpt-4o explícitamente para la máxima calidad en el análisis
    raw = fetch_response(_truncate_input(prompt), model="gpt-4o", temperature=0.2, max_tokens=2048)
    try:
        # Intenta limpiar la respuesta si viene en un bloque de código markdown
        if raw.startswith("```json"):
            raw = raw[7:-3].strip()
        data: Dict[str, Any] = json.loads(raw)
        return data
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error("❌ Error extrayendo insights: %s\nRespuesta del modelo: %s", exc, raw)
        return {}


def fetch_response_with_metadata(
    prompt: str,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 1_024,
) -> Tuple[str, Dict[str, Any]]:
    """
    Igual que fetch_response pero además devuelve metadatos útiles para observabilidad.
    """
    try:
        user_content = prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)
        if not user_content or user_content.strip() == "":
            logger.error("fetch_response_with_metadata: prompt vacío o None; evitando llamada al modelo.")
            return "", {
                "model_name": model,
                "api_status_code": 400,
                "engine_request_id": None,
                "input_tokens": None,
                "output_tokens": None,
                "price_usd": None,
                "error_category": "client_error",
                "error": "empty_prompt",
            }
        user_content = _truncate_input(user_content)
        res = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Eres un asistente útil. Sigue exactamente las instrucciones del usuario."},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=DEFAULT_TIMEOUT,
        )
        answer = res.choices[0].message.content.strip()
        usage = getattr(res, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage else None
        request_id = getattr(res, "id", None)
        meta: Dict[str, Any] = {
            "model_name": model,
            "api_status_code": 200,
            "engine_request_id": request_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        # Estimación simple de coste (ajusta precios según tu contrato)
        if input_tokens is not None and output_tokens is not None:
            # Ejemplo: 0.000005 por token in, 0.000015 por token out
            meta["price_usd"] = round(input_tokens * 0.000005 + output_tokens * 0.000015, 6)
        return answer, meta
    except (OpenAIError, httpx.TimeoutException, TimeoutError) as exc:
        logger.exception("❌ OpenAI API error en fetch_response_with_metadata: %s", exc)
        return "", {
            "model_name": model,
            "api_status_code": getattr(getattr(exc, "response", None), "status_code", None) or 500,
            "engine_request_id": None,
            "input_tokens": None,
            "output_tokens": None,
            "price_usd": None,
            "error_category": "timeout" if isinstance(exc, (httpx.TimeoutException, TimeoutError)) else "api_error",
        }