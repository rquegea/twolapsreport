import psycopg2
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# --- NUEVAS QUERIES SOLICITADAS ---
"""
QUERIES ahora incluye categoría de negocio; guardamos la misma en topic para compatibilidad.
"""
QUERIES = [
    # === Análisis Estratégicos y de Mercado ===
    # Categoría: Análisis de Mercado
    ("¿Qué escuelas superiores o centros privados en España ofrecen programas para escribir guiones y formación audiovisual práctica?", "The Core School", "Análisis de Mercado", "es"),
    ("¿Qué escuelas especializadas en entretenimiento existen en España, aparte de las universidades públicas?", "The Core School", "Análisis de Mercado", "es"),
    ("¿Cuáles son los mejores centros de formación para estudiar desarrollo de videojuegos en España con un enfoque en la industria?", "The Core School", "Análisis de Mercado", "es"),
    ("¿Dónde puedo estudiar cine o televisión en Madrid en escuelas especializadas con un enfoque práctico?", "The Core School", "Análisis de Mercado", "es"),
    ("¿Qué formación especializada existe en España si me gusta el cine y la televisión, más allá de los grados universitarios tradicionales?", "The Core School", "Análisis de Mercado", "es"),
    ("¿Dónde estudiar carreras creativas con enfoque profesional y conexión con empresas en España?", "The Core School", "Análisis de Mercado", "es"),
    ("¿Dónde ofrecen formación profesional para la creación de guiones en Madrid?", "The Core School", "Análisis de Mercado", "es"),
    ("¿Dónde estudiar guion para series en España en escuelas con enfoque práctico?", "The Core School", "Análisis de Mercado", "es"),
    ("¿Cuáles son las mejores escuelas de postproducción y VFX en España?", "The Core School", "Análisis de Mercado", "es"),
    ("¿Qué escuelas de audiovisual en España tienen los mejores acuerdos con productoras y cadenas de televisión?", "The Core School", "Análisis de Mercado", "es"),
    ("¿Qué centros de formación audiovisual tienen la mejor reputación profesional en España, excluyendo universidades públicas?", "The Core School", "Análisis de Mercado", "es"),

    # Categoría: Análisis de Competencia
    ("¿Cuál es el mejor máster en dirección de cine en Europa, enfocándose en escuelas con conexión directa con la industria?", "The Core School", "Análisis de Competencia", "es"),
    ("¿Qué máster online para creación de contenido o para ser influencer es el mejor en España?", "The Core School", "Análisis de Competencia", "es"),
    ("Comparativa de precios de grados en comunicación audiovisual en escuelas privadas de España.", "The Core School", "Análisis de Competencia", "es"),
    ("Análisis comparativo entre las mejores escuelas de formación audiovisual de España", "The Core School", "Análisis de Competencia", "es"),
    ("¿Cuáles son las mejores escuelas en Madrid para formarse en producción de televisión?", "The Core School", "Análisis de Competencia", "es"),
    ("¿Qué alternativas de formación (FP, bootcaps, academias online) consideran los jóvenes en España para el sector audiovisual?", "The Core School", "Análisis de Competencia", "es"),
    ("¿Cuáles son las mejores escuelas en Madrid para formarse en producción de televisión, del estilo de The Core School?", "The Core School", "Análisis de Competencia", "es"),
    ("¿Cómo evoluciona el 'share of voice' y el sentimiento online de The Core School frente a sus competidores en España?", "The Core School", "Análisis de Competencia", "es"),

    # Categoría: Análisis de Oportunidades
    ("¿Qué trabajos en el sector audiovisual tienen buen sueldo y qué formación especializada se necesita?", "The Core School", "Análisis de Oportunidades", "es"),
    ("¿Qué salidas laborales reales ofrece la formación en comunicación audiovisual de escuelas privadas en España?", "The Core School", "Análisis de Oportunidades", "es"),
    ("¿Es difícil encontrar trabajo en la industria del cine en España si estudio en una escuela especializada en lugar de una universidad pública?", "The Core School", "Análisis de Oportunidades", "es"),
    ("¿Qué estilos de trabajo rechazan los jóvenes en España y qué carreras audiovisuales perciben con mayor libertad y mejor sueldo?", "The Core School", "Análisis de Oportunidades", "es"),
    ("¿Qué argumentos (casos de éxito, salarios, empleo) son más persuasivos para los padres en España sobre estudiar carreras audiovisuales?", "The Core School", "Análisis de Oportunidades", "es"),

    # Categoría: Análisis de Riesgos
    ("¿Qué escuelas superiores privadas de cine en España ofrecen becas o ayudas al estudio?", "The Core School", "Análisis de Riesgos", "es"),
    ("¿Cuáles son las preocupaciones de los padres en España sobre las carreras en el sector audiovisual y qué fuentes consultan para informarse?", "The Core School", "Análisis de Riesgos", "es"),

    # Categoría: Análisis de Marketing y Estrategia
    ("¿Qué canales digitales son más efectivos para llegar a jóvenes interesados en audiovisual en España?", "The Core School", "Análisis de Marketing y Estrategia", "es"),
    ("¿Qué intereses en el sector audiovisual y producción de contenidos muestran los jóvenes indecisos en España?", "The Core School", "Análisis de Marketing y Estrategia", "es"),
    ("¿Qué 'triggers' o referentes motivan a los jóvenes en España a interesarse por carreras en el sector audiovisual y qué emociones asocian a ello?", "The Core School", "Análisis de Marketing y Estrategia", "es"),
    ("¿Qué motivaciones llevan a los jóvenes en España a preferir carreras creativas en audiovisual frente a estudios tradicionales?", "The Core School", "Análisis de Marketing y Estrategia", "es"),
    ("¿Cómo perciben los jóvenes en España la industria audiovisual en términos de prestigio, empleabilidad e innovación?", "The Core School", "Análisis de Marketing y Estrategia", "es"),
]

def insert_thecore_queries():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5433)),
        database=os.getenv("POSTGRES_DB", "ai_visibility"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres")
    )
    cur = conn.cursor()

    # BORRAR COMPLETAMENTE todas las queries anteriores
    print("🗑️ Eliminando TODAS las queries anteriores...")
    
    # Primero borrar menciones e insights relacionados (para evitar errores de foreign key)
    cur.execute("DELETE FROM mentions;")
    cur.execute("DELETE FROM insights;")
    # Borrar citations solo si existe
    try:
        cur.execute("DELETE FROM citations;")
    except psycopg2.errors.UndefinedTable:
        conn.rollback()
        print("ℹ️ Tabla 'citations' no existe. Continuando…")
    
    # Ahora borrar todas las queries
    cur.execute("DELETE FROM queries;")
    
    print("✅ Base de datos limpiada completamente.")
    
    # Añadir columnas si no existen
    try:
        cur.execute("ALTER TABLE queries ADD COLUMN language TEXT DEFAULT 'en';")
        conn.commit()
        print("✅ Columna 'language' añadida.")
    except psycopg2.errors.DuplicateColumn:
        conn.rollback()
        print("ℹ️ La columna 'language' ya existía.")
    try:
        cur.execute("ALTER TABLE queries ADD COLUMN category TEXT;")
        conn.commit()
        print("✅ Columna 'category' añadida.")
    except psycopg2.errors.DuplicateColumn:
        conn.rollback()
        print("ℹ️ La columna 'category' ya existía.")

    # Insertar las queries de The Core
    print(f"🎯 Insertando las {len(QUERIES)} queries de The Core School...")
    for i, (query, brand, category, lang) in enumerate(QUERIES, 1):
        # Evitar duplicados de forma portable (sin ON CONFLICT)
        cur.execute("SELECT 1 FROM queries WHERE query = %s", (query,))
        exists = cur.fetchone() is not None
        if not exists:
            cur.execute(
                """
                INSERT INTO queries (query, brand, topic, category, language, enabled)
                VALUES (%s, %s, %s, %s, %s, TRUE)
                """,
                (query, brand, category, category, lang)
            )
            print(f"   {i:2d}. {query[:80]}... (+)")
        else:
            print(f"   {i:2d}. {query[:80]}... (skip)")

    conn.commit()
    print(f"✅ Insertadas las {len(QUERIES)} queries de The Core School correctamente.\n")

    # Mostrar queries activas
    print("📌 Queries ACTIVAS en la base de datos:")
    cur.execute("""
        SELECT id, brand, COALESCE(category, topic) AS category, LEFT(query, 80) as query_preview 
        FROM queries 
        WHERE enabled = TRUE 
        ORDER BY id DESC;
    """)
    active_queries = cur.fetchall()
    
    for row in active_queries:
        print(f"   ID {row[0]:2d}: [{row[1]}] {row[2]} - {row[3]}...")

    # Verificar conteo total
    cur.execute("SELECT COUNT(*) FROM queries WHERE enabled = TRUE")
    total_active = cur.fetchone()[0]
    print(f"\n📊 Total de queries activas: {total_active}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    print("🚀 CONFIGURANDO QUERIES DE THE CORE SCHOOL")
    print("=" * 60)
    insert_thecore_queries()
    
    print("\n🎬 ¡Listo! Ahora puedes:")
    print("1. Ejecutar el scheduler: python -c \"from src.scheduler.poll import main; main(loop_once=True)\"")
    print("2. Ver el frontend en: http://localhost:3000")
    print("3. Verificar datos: python scripts/show_all.py")