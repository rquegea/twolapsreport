import psycopg2
import os

# Conexión a la base de datos (asegúrate de que tu .env está configurado)
conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=int(os.getenv("POSTGRES_PORT", 5433)),
    database=os.getenv("POSTGRES_DB", "ai_visibility"),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD", "postgres"),
)
cur = conn.cursor()

print("🗑️  Eliminando todos los datos de las tablas...")

# Eliminar datos de las tablas dependientes primero
cur.execute("DELETE FROM mentions;")
cur.execute("DELETE FROM insights;")

# Eliminar datos de la tabla principal
cur.execute("DELETE FROM queries;")

conn.commit()

print("✅ Base de datos limpiada correctamente.")

cur.close()
conn.close()