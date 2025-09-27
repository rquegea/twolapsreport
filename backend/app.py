# backend/app.py (API mínima enfocada en informes)

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
from datetime import datetime, timedelta, timezone
import io

# Importar solo del módulo de reports necesario para generar PDFs
from src.reports.generator import generate_report as generate_pdf_report


app = Flask(__name__)
CORS(app)


@app.get('/health')
def health() -> tuple:
    return jsonify({"status": "ok"}), 200


@app.post('/api/reports/generate')
def generate_report_endpoint():
    try:
        payload = request.get_json() or {}

        # Aceptar market/project, client_brand, start_date y end_date
        project_id = payload.get('project_id') or payload.get('market_id')
        brand_name = payload.get('brand') or payload.get('client_brand') or os.getenv('DEFAULT_BRAND')
        # Validación estricta de formato YYYY-MM-DD
        def _parse_ymd(value: str) -> datetime:
            return datetime.strptime(value, '%Y-%m-%d').replace(tzinfo=timezone.utc)

        if payload.get('start_date'):
            try:
                start_dt = _parse_ymd(str(payload['start_date'])).date()
            except Exception:
                return jsonify({"error": "start_date debe tener formato YYYY-MM-DD"}), 400
        else:
            start_dt = (datetime.now(timezone.utc) - timedelta(days=30)).date()

        if payload.get('end_date'):
            try:
                end_dt = _parse_ymd(str(payload['end_date'])).date()
            except Exception:
                return jsonify({"error": "end_date debe tener formato YYYY-MM-DD"}), 400
        else:
            end_dt = datetime.now(timezone.utc).date()

        if start_dt > end_dt:
            return jsonify({"error": "start_date no puede ser posterior a end_date"}), 400

        # Resolver project_id (mercado) si solo llega brand
        if not project_id and brand_name:
            # Lookup sencillo vía SQLAlchemy dentro de aggregator (no modificamos src/):
            from sqlalchemy import text
            from src.reports.aggregator import get_session
            sess = get_session()
            try:
                row = sess.execute(text("SELECT COALESCE(project_id, id) FROM queries WHERE COALESCE(brand, topic) = :b ORDER BY id ASC LIMIT 1"), {"b": brand_name}).first()
                project_id = int(row[0]) if row and row[0] is not None else None
            finally:
                sess.close()

        if not project_id:
            return jsonify({"error": "Debe proporcionar 'project_id' o 'brand'."}), 400

        # Generar PDF principal (con fechas y filtro de cliente)
        pdf_bytes = generate_pdf_report(
            int(project_id),
            start_date=start_dt.strftime('%Y-%m-%d'),
            end_date=end_dt.strftime('%Y-%m-%d'),
            client_brand=brand_name,
        )

        return send_file(
            io.BytesIO(pdf_bytes),
            as_attachment=True,
            download_name=f"Informe_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.pdf",
            mimetype='application/pdf',
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True, use_reloader=False)


