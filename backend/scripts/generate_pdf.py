#!/usr/bin/env python3
"""
CLI para generar PDFs del informe desde terminal.

Ejemplos:
  python backend/scripts/generate_pdf.py --project-id 1 --output backend/files/reporte.pdf
  python backend/scripts/generate_pdf.py --project-id 1 --start 2025-09-01 --end 2025-09-26 --mode full --output backend/files/reporte_full.pdf
  python backend/scripts/generate_pdf.py --mode skeleton --company "Mi Empresa" --output backend/files/estructura.pdf

Modos disponibles:
- skeleton: genera estructura en blanco (portada+índice+secciones)
- hybrid: genera un informe híbrido (rápido) con KPIs + gráficos clave
- full: genera el informe completo (más costoso; usa agentes y prompts)
"""

import argparse
import os
import sys

# Carga de .env si está disponible
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Ajustar sys.path para importar el paquete 'src' (vive en backend/src)
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genera PDFs de informes")
    p.add_argument("--project-id", type=int, help="ID del mercado/proyecto (si no, se preguntará)")
    p.add_argument("--brand", dest="client_brand", help="Nombre exacto del cliente/marca (si no, se preguntará)")
    p.add_argument("--start", dest="start_date", help="Fecha inicio YYYY-MM-DD")
    p.add_argument("--end", dest="end_date", help="Fecha fin YYYY-MM-DD")
    p.add_argument("--mode", choices=["skeleton", "hybrid", "full"], default="hybrid", help="Modo de generación")
    p.add_argument("--company", help="Nombre de la empresa (solo skeleton)")
    p.add_argument("--output", required=True, help="Ruta de salida del PDF")
    p.add_argument("--max-rows", type=int, default=5000, help="Límite de filas para clustering (full)")
    p.add_argument("--no-insights-json", action="store_true", help="No guardar JSON de insights (full)")
    p.add_argument("--non-interactive", action="store_true", help="No preguntar por mercado/cliente (requiere --project-id y/o --brand)")
    return p.parse_args()


def _interactive_select_market_and_brand() -> tuple[int, str]:
    # Evitar dependencias pesadas fuera de este bloque
    from src.reports.aggregator import get_session
    from sqlalchemy import text
    sess = get_session()
    try:
        # Listar mercados por COALESCE(project_id, id)
        rows = sess.execute(text(
            """
            SELECT COALESCE(project_id, id) AS pid,
                   MIN(COALESCE(brand, topic, 'Unknown')) AS main_brand,
                   COUNT(*) AS qn
            FROM queries
            GROUP BY COALESCE(project_id, id)
            ORDER BY pid ASC
            """
        )).all()
        if not rows:
            raise SystemExit("No hay mercados en la tabla 'queries'. Inserta primero queries.")
        print("Selecciona un mercado:")
        for i, (pid, main_brand, qn) in enumerate(rows, 1):
            print(f"  {i}. Mercado #{int(pid)} — Marca principal: {str(main_brand)} ({int(qn)} queries)")
        while True:
            sel = input("Número de mercado: ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(rows):
                pid = int(rows[int(sel) - 1][0])
                break
            print("Entrada inválida. Intenta de nuevo.")

        # Listar marcas disponibles dentro del mercado elegido
        brands = sess.execute(text(
            """
            SELECT DISTINCT COALESCE(brand, topic, 'Unknown') AS b
            FROM queries
            WHERE COALESCE(project_id, id) = :pid
            ORDER BY 1
            """
        ), {"pid": pid}).all()
        brand_list = [str(r[0]) for r in brands]
        print("Selecciona la marca/cliente:")
        for i, b in enumerate(brand_list, 1):
            print(f"  {i}. {b}")
        while True:
            selb = input("Número de marca: ").strip()
            if selb.isdigit() and 1 <= int(selb) <= len(brand_list):
                brand = brand_list[int(selb) - 1]
                break
            print("Entrada inválida. Intenta de nuevo.")
        return pid, brand
    finally:
        sess.close()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def main() -> int:
    args = parse_args()
    out_path = os.path.abspath(args.output)
    ensure_parent_dir(out_path)

    try:
        if args.mode == "skeleton":
            # Import perezoso: evita dependencias pesadas
            from src.reports import pdf_writer  # type: ignore
            company = args.company or os.getenv("COMPANY_NAME", "Empresa")
            pdf_bytes = pdf_writer.build_empty_structure_pdf(company)
        elif args.mode == "hybrid":
            from src.reports import generator as report_generator
            from src.reports import aggregator

            project_id = args.project_id
            client_brand = args.client_brand
            if not args.non_interactive and (project_id is None or not client_brand):
                pid_sel, brand_sel = _interactive_select_market_and_brand()
                project_id = project_id or pid_sel
                client_brand = client_brand or brand_sel

            if project_id is None:
                raise SystemExit("Debes proporcionar --project-id o seleccionar un mercado interactivamente.")

            full_data = aggregator.get_full_report_data(
                int(project_id), start_date=args.start_date, end_date=args.end_date, client_brand=client_brand
            )
            pdf_bytes = report_generator.generate_hybrid_report(full_data)
        else:  # full
            from src.reports import generator as report_generator  # type: ignore
            project_id = args.project_id
            client_brand = args.client_brand
            if not args.non_interactive and project_id is None:
                pid_sel, brand_sel = _interactive_select_market_and_brand()
                project_id = pid_sel
                if not client_brand:
                    client_brand = brand_sel
            if project_id is None:
                raise SystemExit("Debes proporcionar --project-id o seleccionar un mercado interactivamente.")
            bytes_pdf = report_generator.generate_report(
                int(project_id),
                save_insights_json=(not args.no_insights_json),
                start_date=args.start_date,
                end_date=args.end_date,
                client_brand=client_brand,
            )
            pdf_bytes = bytes_pdf

        # Guardar PDF
        with open(out_path, "wb") as f:
            f.write(pdf_bytes or b"")
        print(f"✅ PDF generado: {out_path}")
        return 0
    except Exception as e:
        import sys as _sys
        print(f"❌ Error generando PDF: {e}", file=_sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
