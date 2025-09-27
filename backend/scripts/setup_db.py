#!/usr/bin/env python3
"""
Script para inicializar la base de datos aplicando el esquema SQL.

Funciones principales:
- Detecta credenciales desde variables de entorno (POSTGRES_* o DB_*).
- (Opcional) Crea la base de datos si no existe.
- Aplica el archivo de esquema `src/db/schema.sql`.

Uso:
  python backend/scripts/setup_db.py --verbose --create-db-if-missing
  python backend/scripts/setup_db.py --schema /ruta/a/schema.sql
"""

import argparse
import os
import sys
from contextlib import closing

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    # Permitir que el script funcione incluso si no está instalado dotenv
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False

import psycopg2
import psycopg2.extensions


def env(*names: str, default: str | int | None = None):
    for n in names:
        v = os.getenv(n)
        if v is not None and v != "":
            return v
    return default


def build_db_cfg():
    load_dotenv()
    return {
        "host": env("POSTGRES_HOST", "DB_HOST", default="localhost"),
        "port": int(env("POSTGRES_PORT", "DB_PORT", default=5433)),
        "dbname": env("POSTGRES_DB", "DB_NAME", default="ai_visibility"),
        "user": env("POSTGRES_USER", "DB_USER", default="postgres"),
        "password": env("POSTGRES_PASSWORD", "DB_PASSWORD", default="postgres"),
    }


def connect_db(dbname: str | None = None, cfg: dict | None = None):
    if cfg is None:
        cfg = build_db_cfg()
    dsn = cfg.copy()
    if dbname is not None:
        dsn["dbname"] = dbname
    return psycopg2.connect(**dsn)  # type: ignore[arg-type]


def ensure_database_exists(target_db: str, verbose: bool = False):
    cfg = build_db_cfg()
    # Conectamos a la BD "postgres" para poder crear otras BDs
    admin_db = "postgres" if cfg["dbname"] == target_db else cfg["dbname"]
    try:
        with closing(connect_db(dbname=admin_db, cfg=cfg)) as conn:
            conn.set_session(autocommit=True)
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
                exists = cur.fetchone() is not None
                if not exists:
                    if verbose:
                        print(f"Creando base de datos '{target_db}'...")
                    cur.execute(f"CREATE DATABASE {psycopg2.extensions.quote_ident(target_db, cur)}")
                elif verbose:
                    print(f"BD '{target_db}' ya existe. Continuando…")
    except Exception as e:
        raise SystemExit(f"Error al verificar/crear la BD '{target_db}': {e}")


def apply_schema(schema_path: str, verbose: bool = False):
    if not os.path.isfile(schema_path):
        raise SystemExit(f"No se encontró el schema en: {schema_path}")
    cfg = build_db_cfg()
    if verbose:
        safe = {k: ("***" if k == "password" else v) for k, v in cfg.items()}
        print(f"Conectando con: {safe}")
        print(f"Aplicando esquema desde: {schema_path}")

    sql = open(schema_path, "r", encoding="utf-8").read()
    with closing(connect_db(cfg=cfg)) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    if verbose:
        print("✅ Esquema aplicado correctamente.")


def default_schema_path() -> str:
    # Este script vive en backend/scripts/ — subimos un nivel y buscamos src/db/schema.sql
    here = os.path.dirname(__file__)
    backend_root = os.path.dirname(here)
    return os.path.join(backend_root, "src", "db", "schema.sql")


def main():
    parser = argparse.ArgumentParser(description="Inicializa la BD aplicando schema.sql")
    parser.add_argument("--schema", default=default_schema_path(), help="Ruta al archivo schema.sql")
    parser.add_argument("--create-db-if-missing", action="store_true", help="Crea la BD si no existe")
    parser.add_argument("--verbose", action="store_true", help="Mensajes detallados")
    args = parser.parse_args()

    cfg = build_db_cfg()
    if args.create_db_if_missing:
        ensure_database_exists(str(cfg["dbname"]), verbose=args.verbose)

    apply_schema(args.schema, verbose=args.verbose)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

