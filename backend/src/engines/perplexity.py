import os
import requests
from typing import Tuple, Dict, Any
from dotenv import load_dotenv

load_dotenv()

PPLX_KEY = os.getenv("PERPLEXITY_API_KEY")
API_URL  = "https://api.perplexity.ai/chat/completions"   #  ← ESTO Faltaba
HEADERS  = {
    "Authorization": f"Bearer {PPLX_KEY}",
    "Content-Type": "application/json"
}

def fetch_perplexity_response(query: str) -> str:
    body = {
        "model": "sonar-reasoning",                # modelo disponible en cuentas free
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7
    }
    resp = requests.post(API_URL, headers=HEADERS, json=body)
    if resp.status_code != 200:
        print("🔴 PPLX error detail:", resp.text)  # mostrará la causa exacta
        resp.raise_for_status()

    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def fetch_perplexity_with_metadata(query: str) -> Tuple[str, Dict[str, Any]]:
    body = {
        "model": "sonar-reasoning",
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7
    }
    resp = requests.post(API_URL, headers=HEADERS, json=body)
    meta: Dict[str, Any] = {
        "model_name": body["model"],
        "api_status_code": resp.status_code,
        "engine_request_id": resp.headers.get("x-request-id") or resp.headers.get("x-requestid") or None,
        "input_tokens": None,
        "output_tokens": None,
        "price_usd": None,
    }
    if resp.status_code != 200:
        return "", {**meta, "error_category": "api_error"}
    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage") or {}
    meta.update({
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
    })
    # Si publican pricing por token, se puede estimar aquí
    return text, meta
