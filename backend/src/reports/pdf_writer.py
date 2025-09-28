from typing import Optional, List, Tuple, Dict
from fpdf import FPDF


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, "Informe Estratégico Geocore", 0, 1, "L")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", size=8)
        self.cell(0, 10, f"Página {self.page_no()}", 0, 0, "C")

    # --- Nueva tabla de Oportunidades Competitivas ---
    def write_competitive_opportunities_table(self, opportunities_data: List[Dict[str, str]]):
        if not opportunities_data:
            return
        self.add_page()
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Análisis de Contenido Competitivo y Oportunidades", 0, 1, "C")
        self.ln(3)

        headers = ["Competidor", "Debilidad Detectada", "#", "Acción de Contenido"]
        col_widths = [40, 60, 12, 78]

        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(230, 230, 230)
        for h, w in zip(headers, col_widths):
            self.cell(w, 7, _sanitize(h), 1, 0, 'C', True)
        self.ln()

        self.set_font("Helvetica", size=9)
        for row in opportunities_data:
            comp = _sanitize(str(row.get("Competidor", "")))
            topic = _sanitize(str(row.get("Debilidad Detectada", "")))
            count = _sanitize(str(row.get("#", "")))
            action = _sanitize(str(row.get("Acción de Contenido", "")))

            x, y = self.get_x(), self.get_y()
            h = 6
            # Competidor
            self.multi_cell(col_widths[0], h, comp, border=1)
            y1 = self.get_y()
            self.set_xy(x + col_widths[0], y)
            # Debilidad
            self.multi_cell(col_widths[1], h, topic, border=1)
            y2 = self.get_y()
            self.set_xy(x + col_widths[0] + col_widths[1], y)
            # Conteo
            self.multi_cell(col_widths[2], h, count, border=1, align='C')
            y3 = self.get_y()
            self.set_xy(x + col_widths[0] + col_widths[1] + col_widths[2], y)
            # Acción
            self.multi_cell(col_widths[3], h, action, border=1)
            y4 = self.get_y()
            # Ajustar a la mayor altura
            max_y = max(y1, y2, y3, y4)
            self.set_xy(x, max_y)

    def write_action_plan_table(self, items: List[Dict[str, str]]):
        if not items:
            return
        self.add_page()
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Plan de Acción (Tabla)", 0, 1, "L")
        self.ln(2)
        headers = ["Acción", "Impacto", "Esfuerzo", "Owner", "Due", "Prob.", "Conf."]
        widths = [70, 18, 20, 30, 20, 16, 16]
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(230, 230, 230)
        for h, w in zip(headers, widths):
            self.cell(w, 7, _sanitize(h), 1, 0, 'C', True)
        self.ln()
        self.set_font("Helvetica", size=9)
        for it in items[:20]:
            row = [
                _sanitize(str(it.get("action", ""))[:180]),
                _sanitize(str(it.get("impact", ""))),
                _sanitize(str(it.get("effort", ""))),
                _sanitize(str(it.get("owner", ""))),
                _sanitize(str(it.get("due", ""))),
                _sanitize(str(it.get("probability", ""))),
                _sanitize(str(it.get("confidence", ""))),
            ]
            x0, y0 = self.get_x(), self.get_y()
            self.multi_cell(widths[0], 6, row[0], border=1)
            y_end = self.get_y(); self.set_xy(x0 + widths[0], y0)
            for i in range(1, len(widths)):
                self.multi_cell(widths[i], 6, row[i], border=1, align='C')
                y_end = max(y_end, self.get_y())
                if i < len(widths) - 1:
                    self.set_xy(self.get_x(), y0)
            self.set_xy(x0, y_end)

    def write_evidence_annex(self, items: List[Dict[str, str]]):
        if not items:
            return
        self.add_page()
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Anexo de Evidencias (muestras anónimas)", 0, 1, "L")
        self.ln(2)
        self.set_font("Helvetica", size=10)
        for it in items[:60]:
            ident = _sanitize(str(it.get("id", "")))
            topic = _sanitize(str(it.get("topic", "")))
            snip = _sanitize(str(it.get("snippet", "")))
            self.multi_cell(0, 5, f"[{ident}] ({topic}) \n\u2014 {snip}")
            self.ln(1)


def _sanitize(text: str) -> str:
    """Convierte comillas curvas y otros caracteres Unicode comunes a ASCII apto para Helvetica.
    Elimina caracteres fuera de latin-1 para evitar FPDFUnicodeEncodingException.
    """
    if not isinstance(text, str):
        text = str(text)
    # Reemplazos comunes
    replacements = {
        "“": '"',  # U+201C
        "”": '"',  # U+201D
        "‘": "'",  # U+2018
        "’": "'",  # U+2019
        "–": "-",   # U+2013
        "—": "-",   # U+2014
        "…": "...", # U+2026
        "\xa0": " ",  # NBSP
        "•": "- ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Correcciones ortográficas simples
    try:
        text = text.replace("participació", "participación")
    except Exception:
        pass
    try:
        text = text.encode("latin-1", errors="ignore").decode("latin-1")
    except Exception:
        pass
    return text


def add_title(pdf: ReportPDF, text: str):
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _sanitize(text), 0, 1, "L")
    pdf.ln(2)


def add_paragraph(pdf: ReportPDF, text: str):
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, _sanitize(text))
    pdf.ln(1)


def add_image(pdf: ReportPDF, path: Optional[str], width: float = 180):
    if not path:
        return
    try:
        pdf.image(path, w=width)
        pdf.ln(3)
    except Exception:
        pass


def add_table(pdf: ReportPDF, rows: List[List[str]]):
    if not rows:
        return
    col_widths = [pdf.w / max(1, len(rows[0])) - 10 for _ in rows[0]]
    pdf.set_font("Helvetica", size=10)
    for r_idx, row in enumerate(rows):
        for i, cell in enumerate(row):
            pdf.multi_cell(col_widths[i], 6, _sanitize(str(cell)), border=1, ln=3, max_line_height=pdf.font_size)
        pdf.ln(0)
    pdf.ln(3)


def add_chart_with_caption(pdf: ReportPDF, path: Optional[str], *, caption: str | None = None, width: float = 180):
    if not path:
        return
    try:
        pdf.image(path, w=width)
        pdf.ln(1)
        if caption:
            pdf.set_font("Helvetica", size=8)
            pdf.multi_cell(0, 5, _sanitize(caption))
            pdf.ln(1)
    except Exception:
        pass


def add_agent_insights_section(pdf: ReportPDF, agent_summary: Dict[str, str], raw_buckets: Dict[str, list] | None = None):
    """
    Inserta la sección de "Insights de Agentes" con un resumen ejecutivo y, opcionalmente,
    un anexo con elementos destacados por bucket.
    """
    if not agent_summary:
        return
    add_title(pdf, "Insights del Answer Engine")
    summary_text = agent_summary.get("summary", "")
    if summary_text:
        add_paragraph(pdf, summary_text)
    # Opcional: listas breves de ejemplos
    if raw_buckets:
        def _add_bucket(name: str, title: str, max_items: int = 5):
            items = (raw_buckets.get(name) or [])[:max_items]
            if not items:
                return
            add_title(pdf, title)
            for it in items:
                text = it.get("text") or it.get("opportunity") or it.get("risk") or it.get("trend") or "-"
                add_paragraph(pdf, f"- {text}")
        _add_bucket("opportunities", "Oportunidades destacadas")
        _add_bucket("risks", "Riesgos destacados")
        _add_bucket("trends", "Tendencias destacadas")


def build_strategic_pdf(
    *,
    narrative: Dict,
    kpi_rows: List[List[str]],
    images: Dict[str, Optional[str]],
) -> bytes:
    """
    Genera el PDF final según la estructura narrativa solicitada:
    - Página 1: Dashboard Ejecutivo (headline, evaluación, KPIs, gráfico combinado)
    - Sección 1: Análisis Estratégicos y de Mercado (Competencia, Mercado)
    - Sección 2: Análisis Cuantitativo y Correlaciones (Tendencias, Analisis Profundo)
    - Anexo: Cualitativo Global (síntesis, causa raíz, citas)
    """
    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    # Página 1: Dashboard Ejecutivo
    headline = (narrative.get("headline") or "").strip()
    overall = (narrative.get("evaluacion_general") or "").strip()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Dashboard Ejecutivo", 0, 1, "L")
    if headline:
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 7, headline)
        pdf.ln(2)
    if overall:
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, overall)
        pdf.ln(2)
    if kpi_rows:
        add_title(pdf, "KPIs")
        add_table(pdf, kpi_rows)
    if images.get("combined_vis_sent"):
        add_title(pdf, "Evolución de Visibilidad y Sentimiento")
        add_image(pdf, images.get("combined_vis_sent"))

    # Sección 1: Análisis Estratégicos y de Mercado
    pdf.add_page()
    add_title(pdf, "Sección 1: Análisis Estratégicos y de Mercado")
    comp_text = (narrative.get("analisis_competencia") or "").strip()
    if comp_text:
        add_title(pdf, "Análisis de Competencia")
        add_paragraph(pdf, comp_text)
    if images.get("sov_pie"):
        add_image(pdf, images.get("sov_pie"))
    market_text = (narrative.get("analisis_mercado") or "").strip()
    if market_text:
        add_title(pdf, "Análisis de Mercado")
        add_paragraph(pdf, market_text)
    if images.get("top_topics"):
        add_image(pdf, images.get("top_topics"))

    # Sección 2: Análisis Cuantitativo y Correlaciones
    pdf.add_page()
    add_title(pdf, "Sección 2: Análisis Cuantitativo y Correlaciones")
    # Preferir Sentimiento por Categoría; si no existe, no mostrar gráfico (opción B)
    if images.get("sentiment_by_category"):
        add_title(pdf, "Tendencias: Sentimiento por categoría")
        add_image(pdf, images.get("sentiment_by_category"))
    deep_text = (narrative.get("analisis_profundo") or "").strip()
    if deep_text:
        add_title(pdf, "Correlaciones Transversales (Análisis Profundo)")
        add_paragraph(pdf, deep_text)

    # Anexo: Cualitativo Global
    annex = (narrative.get("cualitativo_global") or {}) if isinstance(narrative, dict) else {}
    if annex:
        pdf.add_page()
        add_title(pdf, "Anexo: Análisis Cualitativo Profundo (Global)")
        if annex.get("sintesis_del_hallazgo"):
            add_paragraph(pdf, f"Síntesis del hallazgo: {annex.get('sintesis_del_hallazgo')}")
        if annex.get("causa_raiz"):
            add_paragraph(pdf, f"Causa raíz: {annex.get('causa_raiz')}")
        citas = annex.get("citas_destacadas") or []
        if isinstance(citas, list) and citas:
            add_title(pdf, "Citas destacadas")
            for c in citas[:8]:
                add_paragraph(pdf, '"' + _sanitize(str(c)) + '"')

    # fpdf2 puede devolver bytearray cuando dest="S"; convertir de forma segura a bytes
    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    try:
        return bytes(str(out).encode("latin-1"))
    except Exception:
        return bytes(str(out).encode("utf-8", errors="ignore"))

def build_pdf(content: Dict) -> bytes:
    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)

    # Página 1: Dashboard Ejecutivo (2 columnas)
    pdf.add_page()
    add_title(pdf, content.get("title") or "Dashboard Ejecutivo")
    # Titular y bullets clave si existen
    strategic = content.get("strategic", {}) if isinstance(content.get("strategic"), dict) else {}
    headline = (strategic.get("headline") or "").strip()
    key_points = strategic.get("key_findings") or []
    if headline:
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 7, _sanitize(headline))
        pdf.ln(2)
    if key_points:
        pdf.set_font("Helvetica", size=10)
        for kp in key_points[:4]:
            add_paragraph(pdf, f"- {kp}")
    page_width = pdf.w - 2 * pdf.l_margin
    col_w = page_width / 2.0
    y0 = pdf.get_y()

    # Columna izquierda: KPIs
    pdf.set_xy(pdf.l_margin, y0)
    add_title(pdf, "KPIs Principales")
    add_table(pdf, content.get("kpi_rows") or [])

    # Columna derecha: evolución
    pdf.set_xy(pdf.l_margin + col_w, y0)
    add_title(pdf, "Evolución Sentimiento")
    img_trend = content.get("images", {}).get("sentiment_evolution")
    if img_trend:
        try:
            pdf.image(img_trend, w=col_w - 6)
        except Exception:
            pass

    # Página 2: Competencia (SOV + Top/Bottom)
    pdf.add_page()
    add_title(pdf, "Análisis Competitivo")
    # SOV Pie a la izquierda
    y1 = pdf.get_y()
    pdf.set_xy(pdf.l_margin, y1)
    img_sov = content.get("images", {}).get("sov_pie")
    if img_sov:
        try:
            pdf.image(img_sov, w=col_w - 6)
        except Exception:
            pass
    # Top/Bottom a la derecha
    pdf.set_xy(pdf.l_margin + col_w, y1)
    img_tb = content.get("images", {}).get("topics_top_bottom")
    if img_tb:
        try:
            pdf.image(img_tb, w=col_w - 6)
        except Exception:
            pass

    # Tabla detallada de oportunidades competitivas (si existe)
    opps = content.get("competitive_opportunities") or []
    if opps:
        pdf.write_competitive_opportunities_table(opps)

    # Página 3+: Clusters
    clusters = content.get("clusters") or []
    if clusters:
        pdf.add_page()
        add_title(pdf, "Principales Temas de Conversación")
        for i, c in enumerate(clusters):
            add_title(pdf, c.get("topic_name", "(sin nombre)"))
            for kp in (c.get("key_points") or [])[:3]:
                add_paragraph(pdf, f"- {kp}")
            examples = c.get("examples") or []
            for ex in examples[:3]:
                q = (ex.get("summary") or "").strip()
                if q:
                    add_paragraph(pdf, '"' + _sanitize(q) + '"')
            if (i + 1) % 3 == 0 and i < len(clusters) - 1:
                pdf.add_page()

    # Última: Síntesis y Plan de Acción
    syn = content.get("clusters_synthesis") or {}
    if syn:
        pdf.add_page()
        add_title(pdf, "Síntesis Estratégica y Plan de Acción")
        metas = syn.get("meta_narrativas") or []
        if metas:
            add_paragraph(pdf, "Meta-narrativas:")
            for m in metas[:6]:
                add_paragraph(pdf, f"- {m}")
        if syn.get("oportunidad_principal"):
            add_paragraph(pdf, f"Oportunidad principal: {syn['oportunidad_principal']}")
        if syn.get("riesgo_inminente"):
            add_paragraph(pdf, f"Riesgo inminente: {syn['riesgo_inminente']}")
        plan = syn.get("plan_estrategico") or []
        if plan:
            add_paragraph(pdf, "Plan estratégico:")
            for p in plan[:8]:
                add_paragraph(pdf, f"- {p}")

    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    try:
        return bytes(str(out).encode("latin-1"))
    except Exception:
        return bytes(str(out).encode("utf-8", errors="ignore"))



# --- NUEVO: Generador de estructura en blanco (portada + índice + secciones) ---
def build_empty_structure_pdf(company_name: str) -> bytes:
    """
    Genera un PDF con solo la estructura solicitada:
    - Portada con título "Marketing Intelligence" y nombre de empresa
    - Índice con todas las secciones
    - Páginas en blanco para cada sección con su encabezado
    """
    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)

    # 1) Portada
    pdf.add_page()
    pdf.ln(60)
    pdf.set_font("Helvetica", "B", 28)
    pdf.cell(0, 14, _sanitize("Marketing Intelligence"), 0, 1, "C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _sanitize(f"{company_name}"), 0, 1, "C")

    # 2) Índice
    pdf.add_page()
    add_title(pdf, "Índice")

    # Definir estructura solicitada
    parte1 = [
        "Share of Voice vs. Competencia",
        "Analisis de visibilidad (diaria)",
        "Evolución del sentimiento",
    ]
    parte2 = [
        "Informe Estratégico",
        "Resumen Ejecutivo",
        "Resumen Ejecutivo y Hallazgos Principales",
        "Tendencias y Señales Emergentes",
        "Análisis Competitivo",
        "Plan de Acción Estratégico",
        "Correlaciones Transversales entre Categorías",
    ]

    def _write_index_block(title: str, items: List[str]):
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, _sanitize(title), 0, 1, "L")
        pdf.set_font("Helvetica", size=11)
        for it in items:
            pdf.cell(0, 6, _sanitize(f"- {it}"), 0, 1, "L")
        pdf.ln(2)

    _write_index_block("Parte 1: Resumen Visual y KPIs", parte1)
    _write_index_block("Parte 2: Informe Estratégico", parte2)

    # 3) Secciones en blanco
    def _write_section_page(title: str):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, _sanitize(title), 0, 1, "L")
        pdf.ln(4)

    # Parte 1
    _write_section_page("Parte 1: Resumen Visual y KPIs")
    for s in parte1:
        _write_section_page(s)

    # Parte 2
    _write_section_page("Parte 2: Informe Estratégico")
    for s in parte2:
        _write_section_page(s)

    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    try:
        return bytes(str(out).encode("latin-1"))
    except Exception:
        return bytes(str(out).encode("utf-8", errors="ignore"))



# --- NUEVO: Estructura con contenido para Parte 1 ---
def build_skeleton_with_content(company_name: str, images: Dict[str, Optional[str]], strategic: Dict[str, str] | None = None) -> bytes:
    """
    Genera un PDF con la estructura (portada + índice + secciones) e inserta
    contenido visual en las secciones de la Parte 1 cuando haya imágenes disponibles.

    Secciones pobladas:
    - "Share of Voice vs. Competencia" <= images["part1_sov_donut"]
    - "Evolución del Sentimiento (diario)" <= images["part1_sentiment_line"]
    - "Serie temporal - Análisis de competencia" <= images["part1_visibility_ranking"]
    - "Resumen Ejecutivo de KPIs" <= images["part1_category_distribution"]
    """
    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)

    # 1) Portada
    pdf.add_page()
    pdf.ln(60)
    pdf.set_font("Helvetica", "B", 28)
    pdf.cell(0, 14, _sanitize("Marketing Intelligence"), 0, 1, "C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _sanitize(f"{company_name}"), 0, 1, "C")

    # 2) Índice
    pdf.add_page()
    add_title(pdf, "Índice")

    parte1 = [
        "Share of Voice vs. Competencia",
        "Analisis de visibilidad (diaria)",
        "Evolución del sentimiento",
    ]
    parte2 = [
        "Informe Estratégico",
        "Resumen Ejecutivo",
        "Resumen Ejecutivo y Hallazgos Principales",
        "Tendencias y Señales Emergentes",
        "Análisis Competitivo",
        "Plan de Acción Estratégico",
        "Correlaciones Transversales entre Categorías",
    ]

    def _write_index_block(title: str, items: List[str]):
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, _sanitize(title), 0, 1, "L")
        pdf.set_font("Helvetica", size=11)
        for it in items:
            pdf.cell(0, 6, _sanitize(f"- {it}"), 0, 1, "L")
        pdf.ln(2)

    _write_index_block("Parte 1: Resumen Visual y KPIs", parte1)
    _write_index_block("Parte 2: Informe Estratégico", parte2)

    # 2.1) Dashboard Ejecutivo (Qué cambió)
    try:
        pdf.add_page()
        add_title(pdf, "Dashboard Ejecutivo — Qué cambió")
        deltas = (strategic or {}).get("deltas") if isinstance(strategic, dict) else None
        k = (strategic or {}).get("kpis") if isinstance(strategic, dict) else None
        rows: List[List[str]] = []
        if isinstance(k, dict):
            sov = k.get("share_of_voice") or k.get("sov")
            sent = k.get("average_sentiment") or k.get("sentiment_avg")
            rows.append(["Share of Voice", f"{float(sov or 0.0):.1f}%", ""]) 
            rows.append(["Sentimiento", f"{float(sent or 0.0):.2f}", ""]) 
        if isinstance(deltas, dict):
            def _fmt_delta(x):
                try:
                    return ("▲" if float(x) > 0 else ("▼" if float(x) < 0 else "=")) + f" {float(x):+.2f}"
                except Exception:
                    return "N/D"
            # Sobrescribir tercera columna con MoM/WoW
            if rows:
                if deltas.get("sov") is not None:
                    rows[0][2] = _fmt_delta(deltas.get("sov"))
                if deltas.get("sentiment") is not None and len(rows) > 1:
                    rows[1][2] = _fmt_delta(deltas.get("sentiment"))
            # Visibilidad como fila adicional
            if deltas.get("visibility") is not None:
                rows.append(["Visibilidad (media)", "—", _fmt_delta(deltas.get("visibility"))])
        if rows:
            add_table(pdf, [["Métrica", "Valor", "Δ Previo"]] + rows)
    except Exception:
        pass

    # Ficha técnica
    try:
        pdf.add_page()
        add_title(pdf, "Ficha técnica")
        meta = strategic or {}
        k = (meta.get("kpis") or {}) if isinstance(meta, dict) else {}
        total_m = k.get("total_mentions")
        period = meta.get("period") or {}
        competitors = meta.get("competitors") or []
        ai_engines = meta.get("ai_engines") or ["OpenAI GPT-4o"]
        scope = meta.get("scope") or "Nacional"
        definitions = meta.get("definitions") or {
            "Sentimiento": "Promedio diario en [-1, 1] de menciones atribuibles a la marca.",
            "Mención": "Registro textual asociado a queries del mercado (prensa, redes o foros si el dataset lo contiene).",
            "Share of Voice (SOV)": "Porcentaje de la conversación del periodo atribuible a la marca frente al total con marca detectada.",
            "Visibilidad": "Serie diaria (0–100%) que indica el % de consultas del día donde la marca aparece al menos una vez.",
        }
        add_paragraph(pdf, f"- Tipo: Análisis de conversación y percepción de marca")
        add_paragraph(pdf, f"- Periodo: {period.get('start','N/D')} a {period.get('end','N/D')}")
        add_paragraph(pdf, f"- Ámbito: {scope}")
        add_paragraph(pdf, f"- Menciones totales: {total_m if total_m is not None else 'N/D'}")
        add_paragraph(pdf, f"- Competidores incluidos: {', '.join(competitors) if competitors else 'N/D'}")
        add_paragraph(pdf, f"- Motores de IA: {', '.join(ai_engines)}")
        add_title(pdf, "Definiciones")
        if isinstance(definitions, dict):
            for kdef, vdef in definitions.items():
                add_paragraph(pdf, f"- {kdef}: {vdef}")
    except Exception:
        pass

    # 3) Secciones: cabeceras + contenido de Parte 1 cuando aplique
    def _write_section_page(title: str):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, _sanitize(title), 0, 1, "L")
        pdf.ln(4)

    # Portada de Parte 1
    _write_section_page("Parte 1: Resumen Visual y KPIs")

    # Mapeo de secciones -> claves de imágenes
    section_to_image_keys: Dict[str, List[str]] = {
        "Share of Voice vs. Competencia": ["sov_pie", "part1_sov_donut"],
        "Analisis de visibilidad (diaria)": ["part1_visibility_line", "visibility_line"],
        "Evolución del sentimiento": ["part1_sentiment_line", "sentiment_evolution"],
    }

    for s in parte1:
        _write_section_page(s)
        if s == "Share of Voice vs. Competencia":
            add_image(pdf, images.get("sov_pie") or images.get("part1_sov_donut"), width=180)
            continue
        if s == "Analisis de visibilidad (diaria)":
            add_image(pdf, images.get("part1_visibility_line") or images.get("visibility_line"), width=180)
            continue
        if s == "Evolución del sentimiento":
            add_image(pdf, images.get("part1_sentiment_line") or images.get("sentiment_evolution"), width=180)
            continue

    # Parte 2 (cabeceras) + inserción de contenidos estratégicos si vienen
    _write_section_page("Parte 2: Informe Estratégico")
    for s in parte2:
        _write_section_page(s)
        if strategic:
            if s == "Plan de Acción Estratégico":
                # Tabla si existe
                items = (strategic.get("action_plan_table") or []) if isinstance(strategic.get("action_plan_table"), list) else []
                if items:
                    pdf.write_action_plan_table(items)  # type: ignore[attr-defined]
                # Texto de apoyo si existe
                text = (strategic.get("action_plan") or "").strip()
                if text:
                    add_paragraph(pdf, text)
            elif s == "Resumen Ejecutivo":
                text = (strategic.get("executive_summary") or "").strip()
                if text:
                    add_paragraph(pdf, text)
            elif s == "Resumen Ejecutivo y Hallazgos Principales":
                text = (strategic.get("summary_and_findings") or "").strip()
                if text:
                    add_paragraph(pdf, text)
            elif s == "Análisis Competitivo":
                text = (strategic.get("competitive_analysis") or "").strip()
                if text:
                    add_paragraph(pdf, text)
                # Si hay fichas por competidor, renderizarlas
                profiles = strategic.get("competitor_profiles") or []
                if isinstance(profiles, list) and profiles:
                    try:
                        add_title(pdf, "Fichas de Competidores")
                        add_competitor_profiles(pdf, profiles)  # type: ignore[attr-defined]
                    except Exception:
                        pass
            elif s == "Tendencias y Señales Emergentes":
                # Preferir gráfico de sentimiento por categoría para alinear visual con el texto
                try:
                    total_m = ((strategic.get("kpis") or {}).get("total_mentions") if isinstance(strategic.get("kpis"), dict) else None)
                    caption = f"Base: {total_m if total_m is not None else 'N/D'} menciones | Fuente: Geocore (DB) + motores IA"
                except Exception:
                    caption = "Fuente: Geocore (DB) + motores IA"
                img_sent_cat = images.get("sentiment_by_category")
                if img_sent_cat:
                    add_chart_with_caption(pdf, img_sent_cat, caption=caption, width=180)
                # Si no hay datos adecuados, no insertar gráfico (opción B)
                text = (strategic.get("trends") or "").strip()
                if text:
                    add_paragraph(pdf, text)
            elif s == "Correlaciones Transversales entre Categorías":
                # Insertar gráficos si existen
                try:
                    add_image(pdf, images.get("sov_sent_scatter"), width=170)
                except Exception:
                    pass
                try:
                    add_image(pdf, images.get("corr_heatmap"), width=120)
                except Exception:
                    pass
                try:
                    add_image(pdf, images.get("lag_corr"), width=170)
                except Exception:
                    pass
                text = (strategic.get("correlations") or "").strip()
                if text:
                    add_paragraph(pdf, text)
    # Evidencias (si vienen)
    try:
        ev = (strategic or {}).get("evidence_annex") if isinstance(strategic, dict) else None
        if isinstance(ev, list) and ev:
            pdf.write_evidence_annex(ev)  # type: ignore[attr-defined]
    except Exception:
        pass
    
    # NUEVO: Anexo — Análisis Detallado del SOV por Categoría
    try:
        pdf.add_page()
        add_title(pdf, "Anexo: Análisis Detallado — SOV por Categoría")
        if images.get("sov_by_category"):
            add_chart_with_caption(pdf, images.get("sov_by_category"), caption="SOV del cliente por categoría (% sobre total con marca).", width=180)
        if images.get("sov_by_category_delta"):
            add_chart_with_caption(pdf, images.get("sov_by_category_delta"), caption="Δ SOV por categoría vs periodo previo (p.p.).", width=180)
        cat_texts = (strategic or {}).get("category_analyses") or {}
        if isinstance(cat_texts, dict) and cat_texts:
            for cat, txt in list(cat_texts.items())[:10]:
                add_title(pdf, str(cat))
                add_paragraph(pdf, str(txt))
    except Exception:
        pass
    # Eliminado Parte 3 según solicitud

    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    try:
        return bytes(str(out).encode("latin-1"))
    except Exception:
        return bytes(str(out).encode("utf-8", errors="ignore"))


def build_skeleton_from_content(content: Dict) -> bytes:
    """Convenience wrapper que acepta un único diccionario con todo el contenido.
    Espera: { company_name: str, images: Dict[str,str|None], strategic: Dict[str,str] }
    """
    company_name = (content.get("company_name") or "Empresa") if isinstance(content, dict) else "Empresa"
    images = (content.get("images") or {}) if isinstance(content, dict) else {}
    strategic = (content.get("strategic") or {}) if isinstance(content, dict) else {}
    return build_skeleton_with_content(company_name, images, strategic)


# --- NUEVO: Render de fichas por competidor ---
def add_competitor_profiles(pdf: ReportPDF, profiles: list[dict]):
    if not isinstance(profiles, list) or not profiles:
        return
    for prof in profiles:
        try:
            name = str(prof.get("name") or "Competidor")
            k = prof.get("kpis") or {}
            top_topics = prof.get("top_topics") or []
            bottom_topics = prof.get("bottom_topics") or []
            narrative = (prof.get("narrative") or "").strip()
            imgs = prof.get("images") or {}

            pdf.add_page()
            add_title(pdf, f"Ficha del Competidor — {name}")

            # KPIs mini-tabla
            rows = [
                ["Métrica", "Valor"],
                ["Menciones", str(int(k.get("total_mentions", 0)))],
                ["Sentimiento medio", f"{float(k.get('sentiment_avg', 0.0) or 0.0):.2f}"],
                ["SOV", f"{float(k.get('sov', 0.0) or 0.0):.1f}%"],
                ["Visibilidad media", f"{float(k.get('visibility_avg', 0.0) or 0.0):.1f}%"],
            ]
            add_table(pdf, rows)

            # Dos columnas de gráficos si existen
            page_width = pdf.w - 2 * pdf.l_margin
            col_w = page_width / 2.0
            y0 = pdf.get_y()
            pdf.set_xy(pdf.l_margin, y0)
            if imgs.get("sentiment_line"):
                add_title(pdf, "Evolución del Sentimiento")
                try:
                    pdf.image(imgs.get("sentiment_line"), w=col_w - 6)
                except Exception:
                    pass
            pdf.set_xy(pdf.l_margin + col_w, y0)
            if imgs.get("visibility_line"):
                add_title(pdf, "Evolución de la Visibilidad")
                try:
                    pdf.image(imgs.get("visibility_line"), w=col_w - 6)
                except Exception:
                    pass

            # Narrativa y flancos
            pdf.ln(4)
            if narrative:
                add_title(pdf, "Lectura Estratégica")
                add_paragraph(pdf, narrative)
            flanks = prof.get("flanks") or []
            if isinstance(flanks, list) and flanks:
                add_title(pdf, "Flancos Explotables (con evidencia)")
                for f in flanks[:2]:
                    add_paragraph(pdf, f"- {str(f)}")
            # Top/Bottom temas
            if top_topics or bottom_topics:
                add_title(pdf, "Temas Clave")
                if top_topics:
                    add_paragraph(pdf, "Top (más favorables):")
                    for t, v in top_topics[:5]:
                        add_paragraph(pdf, f"- {t}: {float(v):.2f}")
                if bottom_topics:
                    add_paragraph(pdf, "Bottom (más desfavorables):")
                    for t, v in bottom_topics[:5]:
                        add_paragraph(pdf, f"- {t}: {float(v):.2f}")
        except Exception:
            continue
