import os
from dotenv import load_dotenv
import psycopg2

# Carga las variables del .env
load_dotenv()

DB_HOST = os.getenv("DB_HOST") or "localhost"
DB_PORT = os.getenv("DB_PORT") or "5432"
DB_NAME = os.getenv("DB_NAME") or "ai_visibility"
DB_USER = os.getenv("DB_USER") or "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD") or "postgres"

schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")

with open(schema_path, "r") as f:
    schema = f.read()

# Conexión usando las variables del entorno
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

cur = conn.cursor()
cur.execute(schema)
conn.commit()
cur.close()
conn.close()

print("✅ Base de datos inicializada correctamente.")
