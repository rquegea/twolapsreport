import os
import json
import requests


def main():
    api = os.environ.get("API_URL", "http://localhost:5050")
    brand = os.environ.get("BRAND") or None
    dry_run = os.environ.get("DRY_RUN", "true").lower() in ("1", "true", "yes")
    limit = os.environ.get("LIMIT")
    payload = {"dry_run": dry_run}
    if brand:
        payload["brand"] = brand
    if limit:
        payload["limit"] = int(limit)

    resp = requests.post(f"{api}/api/prompts/recategorize", json=payload)
    resp.raise_for_status()
    data = resp.json()
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


