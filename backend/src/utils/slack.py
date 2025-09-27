import os
import requests

def send_slack_alert(query, sentiment, response):
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("⚠️ SLACK_WEBHOOK_URL no está definido en el entorno")
        return

    message = f"""
🚨 *Alerta de Sentimiento Negativo*
*Query:* {query}
*Sentimiento:* {sentiment}
*Respuesta:* {response}
"""
    payload = {"text": message}
    try:
        requests.post(webhook_url, json=payload)
        print("📣 Alerta enviada a Slack")
    except Exception as e:
        print(f"❌ Error al enviar alerta a Slack: {e}")

