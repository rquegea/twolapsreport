from __future__ import annotations

from typing import Dict, List, Any


def _score_item(item: Dict[str, Any]) -> float:
    """
    Heuristic score by impact and frequency.
    Expects keys: impact, count/frequency.
    """
    impact = (item.get("impact") or "medio").lower()
    base = {"alto": 3.0, "medio": 2.0, "bajo": 1.0}.get(impact, 1.5)
    freq = float(item.get("count") or item.get("frequency") or 1.0)
    return base * (1.0 + min(freq, 100.0) / 25.0)


def normalize_agent_payloads(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Normalize a list of rows from `insights.payload` into canonical buckets.
    """
    out: Dict[str, List[Dict[str, Any]]] = {
        "opportunities": [],
        "risks": [],
        "trends": [],
        "quotes": [],
        "ctas": [],
    }
    for r in rows:
        payload = r.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        for bucket in out.keys():
            items = payload.get(bucket) or []
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict):
                        out[bucket].append(it)
                    elif isinstance(it, str):
                        out[bucket].append({"text": it})
    return out


def summarize_agent_insights(rows: List[Dict[str, Any]], limit_per_bucket: int = 50) -> Dict[str, Any]:
    """
    Top-N per bucket plus meta counts.
    """
    buckets = normalize_agent_payloads(rows)
    ranked: Dict[str, List[Dict[str, Any]]] = {}
    for name, items in buckets.items():
        if name in ("opportunities", "risks", "trends"):
            ranked[name] = sorted(items, key=_score_item, reverse=True)[:limit_per_bucket]
        else:
            ranked[name] = items[:limit_per_bucket]

    meta = {
        "counts": {k: len(v) for k, v in buckets.items()},
    }
    return {"buckets": ranked, "meta": meta}
