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
    p.add_argument("--project-id", type=int, default=1, help="ID del proyecto/consulta en la BD")
    p.add_argument("--start", dest="start_date", help="Fecha inicio YYYY-MM-DD")
    p.add_argument("--end", dest="end_date", help="Fecha fin YYYY-MM-DD")
    p.add_argument("--mode", choices=["skeleton", "hybrid", "full"], default="hybrid", help="Modo de generación")
    p.add_argument("--company", help="Nombre de la empresa (solo skeleton)")
    p.add_argument("--output", required=True, help="Ruta de salida del PDF")
    p.add_argument("--max-rows", type=int, default=5000, help="Límite de filas para clustering (full)")
    p.add_argument("--no-insights-json", action="store_true", help="No guardar JSON de insights (full)")
    return p.parse_args()


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
            from src.reports import generator as report_generator  # type: ignore
            full_data = report_generator.get_full_report_data(
                args.project_id, start_date=args.start_date, end_date=args.end_date, max_rows=2000
            )
            pdf_bytes = report_generator.generate_hybrid_report(full_data)
        else:  # full
            from src.reports import generator as report_generator  # type: ignore
            bytes_pdf = report_generator.generate_report(
                args.project_id,
                save_insights_json=(not args.no_insights_json),
                start_date=args.start_date,
                end_date=args.end_date,
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
