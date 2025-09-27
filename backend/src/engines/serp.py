import os
from urllib.parse import urlparse
from serpapi import GoogleSearch
from typing import List, Dict, Any, Tuple

def get_search_results(query: str) -> str:
    params = {
        "q": query,
        "api_key": os.getenv("SERPAPI_KEY")
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    answers = []
    for result in results.get("organic_results", []):
        answers.append(result.get("snippet", ""))

    return "\n".join(answers[:3]) if answers else "No results found."


def get_search_results_structured(query: str, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    params = {
        "q": query,
        "api_key": os.getenv("SERPAPI_KEY")
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    organic = results.get("organic_results", [])
    structured: List[Dict[str, Any]] = []
    snippets: List[str] = []
    for idx, item in enumerate(organic[:top_k], start=1):
        link = item.get("link") or ""
        domain = urlparse(link).netloc if link else None
        entry = {
            "rank": idx,
            "title": item.get("title"),
            "url": link,
            "domain": domain,
            "snippet": item.get("snippet"),
        }
        structured.append(entry)
        if item.get("snippet"):
            snippets.append(f"Fuente: {domain or ''} | Título: {item.get('title','')}\nResumen: {item.get('snippet','')}")

    text = "\n\n".join(snippets) if snippets else "No results found."
    return text, structured
