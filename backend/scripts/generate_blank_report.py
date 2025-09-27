import argparse
from pathlib import Path

from src.reports.pdf_writer import build_empty_structure_pdf


def main():
    parser = argparse.ArgumentParser(description="Genera un PDF en blanco con la estructura del informe")
    parser.add_argument("company", help="Nombre de la empresa")
    parser.add_argument("--output", "-o", help="Ruta de salida del PDF", default="informe_estructura.pdf")
    args = parser.parse_args()

    pdf_bytes = build_empty_structure_pdf(args.company)
    out_path = Path(args.output)
    out_path.write_bytes(pdf_bytes)
    print(f"PDF generado en: {out_path.resolve()}")


if __name__ == "__main__":
    main()


