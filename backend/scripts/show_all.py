import psycopg2
import os
import json
from tabulate import tabulate

conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=int(os.getenv("POSTGRES_PORT", 5433)),
    database=os.getenv("POSTGRES_DB", "ai_visibility"),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD", "postgres"),
)
cur = conn.cursor()

print("\n📋 Tablas en la base de datos:")
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
for row in cur.fetchall():
    print(" -", row[0])

# --- Queries ---
print("\n📌 Contenido de 'queries':")
cur.execute("SELECT id, query, brand, topic, language FROM queries ORDER BY id")
queries = cur.fetchall()
print(tabulate(queries, headers=["ID", "Query", "Brand", "Topic", "Lang"], tablefmt="grid"))

# --- Mentions ---
print("\n🗂 Contenido de 'mentions':")
cur.execute("""
    SELECT id, query_id, engine, sentiment, emotion, confidence_score, created_at, response
    FROM mentions
    ORDER BY created_at DESC
""")
mentions = cur.fetchall()
for m in mentions:
    print("\n--- Mention ---")
    print(f"ID: {m[0]} | Query: {m[1]} | Engine: {m[2]} | Sent: {m[3]:.2f} | Emo: {m[4]} | Conf: {m[5]:.2f} | Time: {m[6]}")
    print(f"Response:\n{m[7]}\n")

# --- Insights ---
print("\n💡 Contenido de 'insights':")
cur.execute("SELECT id, query_id, payload FROM insights ORDER BY id DESC")
insights = cur.fetchall()
for ins in insights:
    print(f"\nInsight ID: {ins[0]} | Query: {ins[1]}")
    try:
        payload = json.loads(ins[2])
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception:
        print(ins[2])

cur.close()
conn.close()
