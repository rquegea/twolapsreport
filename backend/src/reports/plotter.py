import os
import tempfile
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    from wordcloud import WordCloud
except Exception:  # pragma: no cover
    WordCloud = None  # type: ignore


def _tmp_path(prefix: str, suffix: str = ".png") -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return path


def plot_line_series(dates: List[str], values: List[float], *, title: str = "", ylabel: str = "", ylim: tuple | None = None, color: str = "#1f77b4") -> Optional[str]:
    if not dates or not values:
        return None
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 3))
    sns.lineplot(x=dates, y=values, marker="o", linewidth=1.8, color=color)
    try:
        step = max(1, len(dates) // 8)
        xticks_idx = list(range(0, len(dates), step))
        xticks_labels = [dates[i] for i in xticks_idx]
        plt.xticks(xticks_idx, xticks_labels, rotation=45, ha="right")
    except Exception:
        plt.xticks(rotation=45, ha="right")
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    out = _tmp_path("line_series_")
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_visibility_series(dates: List[str], values: List[float]) -> Optional[str]:
    """Gráfico de líneas (lineplot) de Puntuación de Visibilidad (0–100%)."""
    if not dates or not values:
        return None
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 3))
    sns.lineplot(x=dates, y=values, marker="o", linewidth=1.8, color="#000000")
    try:
        step = max(1, len(dates) // 8)
        xticks_idx = list(range(0, len(dates), step))
        xticks_labels = [dates[i] for i in xticks_idx]
        plt.xticks(xticks_idx, xticks_labels, rotation=45, ha="right")
    except Exception:
        plt.xticks(rotation=45, ha="right")
    plt.ylabel("Visibilidad (%)")
    plt.ylim(0, 100)
    plt.tight_layout()
    out = _tmp_path("visibility_series_")
    plt.savefig(out, dpi=160)
    plt.close()
    return out

def plot_sentiment_evolution(series: List[Tuple[str, float]]) -> Optional[str]:
    if not series:
        return None
    dates = [d for d, _ in series]
    values = [v for _, v in series]
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 3))
    sns.lineplot(x=dates, y=values, marker="o", linewidth=1.8, label="Sentimiento")
    # Reducir etiquetas en eje X para evitar solapes
    try:
        step = max(1, len(dates) // 8)
        xticks_idx = list(range(0, len(dates), step))
        xticks_labels = [dates[i] for i in xticks_idx]
        plt.xticks(xticks_idx, xticks_labels, rotation=45, ha="right")
    except Exception:
        plt.xticks(rotation=45, ha="right")
    plt.ylabel("Sentimiento")
    plt.legend(loc="best", fontsize=8)
    plt.ylim(-1.0, 1.0)
    plt.tight_layout()
    out = _tmp_path("sentiment_evolution_")
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_sentiment_by_category(cat_to_avg: Dict[str, float]) -> Optional[str]:
    if not cat_to_avg:
        return None
    items = sorted(cat_to_avg.items(), key=lambda x: x[1])
    cats = [k for k, _ in items]
    vals = [v for _, v in items]
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 3.6))
    sns.barplot(x=vals, y=cats, orient="h", palette="vlag")
    plt.xlim(-1.0, 1.0)
    plt.tight_layout()
    out = _tmp_path("sentiment_by_category_")
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_topics_top_bottom(top5: List[Tuple[str, float]], bottom5: List[Tuple[str, float]]) -> Optional[str]:
    labels = [t for t, _ in bottom5] + [t for t, _ in top5]
    vals = [v for _, v in bottom5] + [v for _, v in top5]
    if not labels:
        return None
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 3.6))
    sns.barplot(x=vals, y=labels, orient="h", palette=["#d9534f" for _ in bottom5] + ["#5cb85c" for _ in top5])
    plt.xlim(-1.0, 1.0)
    plt.tight_layout()
    out = _tmp_path("topics_top_bottom_")
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_sov_pie(sov_list: List[Tuple[str, float]]) -> Optional[str]:
    if not sov_list:
        return None
    # Asegurar múltiples marcas: filtrar elementos con tamaño > 0 y normalizar
    filtered = [(n, float(v)) for n, v in sov_list if v is not None]
    if not filtered:
        return None
    total = sum(v for _, v in filtered) or 1.0
    labels = [n for n, _ in filtered]
    sizes = [max(0.01, (v / total) * 100.0) for _, v in filtered]
    plt.figure(figsize=(4.8, 4.8))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, textprops={"fontsize": 8})
    plt.tight_layout()
    out = _tmp_path("sov_pie_")
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_wordcloud_from_corpus(texts: List[str], *, width: int = 800, height: int = 400, background_color: str = "white") -> Optional[str]:
    """
    Genera una imagen de nube de palabras a partir de una lista de textos.
    Si la librería wordcloud no está disponible o el corpus está vacío, devuelve None.
    """
    if not texts or not isinstance(texts, list):
        return None
    if WordCloud is None:
        return None
    try:
        joined = "\n".join([t for t in texts if isinstance(t, str) and t.strip()])
        if not joined.strip():
            return None
        wc = WordCloud(width=width, height=height, background_color=background_color, collocations=False)
        wc_img = wc.generate(joined).to_array()
        plt.figure(figsize=(width/100, height/100))
        plt.imshow(np.array(wc_img), interpolation="bilinear")
        plt.axis("off")
        out = _tmp_path("wordcloud_")
        plt.savefig(out, dpi=160, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        return out
    except Exception:
        return None


def plot_combined_visibility_sentiment(dates: List[str], visibility: List[float], sentiment: List[float]) -> Optional[str]:
    if not dates or not visibility or not sentiment:
        return None
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 3.2))
        ax1 = plt.gca()
        # Reducir etiquetas X
        step = max(1, len(dates) // 8)
        xticks_idx = list(range(0, len(dates), step))
        ax1.plot(range(len(dates)), visibility, color='tab:blue', marker='o', linewidth=1.6, label='Visibilidad')
        ax1.set_ylabel('Visibilidad (%)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2 = ax1.twinx()
        ax2.plot(range(len(dates)), sentiment, color='tab:green', marker='x', linestyle='--', linewidth=1.4, label='Sentimiento')
        ax2.set_ylabel('Sentimiento', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        ax2.set_ylim(-1, 1)
        plt.xticks(xticks_idx, [dates[i] for i in xticks_idx], rotation=45, ha='right')
        # Leyenda combinada
        lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        ax1.legend(lines, labels, loc='upper left', fontsize=8)
        plt.tight_layout()
        out = _tmp_path("combined_vis_sent_")
        plt.savefig(out, dpi=160)
        plt.close()
        return out
    except Exception:
        return None


def plot_mentions_volume(dates: List[str], counts: List[int]) -> Optional[str]:
    if not dates or not counts:
        return None
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 3))
    sns.lineplot(x=dates, y=counts, marker="o", linewidth=1.8, color="#1f77b4")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = _tmp_path("mentions_volume_")
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_top_topics(topic_counts: Dict[str, int], top_n: int = 10) -> Optional[str]:
    if not topic_counts:
        return None
    items = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:max(1, top_n)]
    if not items:
        return None
    topics = [k for k, _ in items][::-1]
    counts = [v for _, v in items][::-1]
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 3.6))
    sns.barplot(x=counts, y=topics, orient="h", palette="Blues_r")
    plt.tight_layout()
    out = _tmp_path("top_topics_")
    plt.savefig(out, dpi=160)
    plt.close()
    return out


# --- NUEVOS GRÁFICOS: SOV por categoría ---
def plot_sov_by_category(cat_map: Dict[str, Dict[str, int]]) -> Optional[str]:
    """
    Renderiza barras horizontales con el SOV (%) del cliente por categoría.
    Espera estructura: { category: {"client": int, "total": int, ...}, ... }
    """
    if not cat_map or not isinstance(cat_map, dict):
        return None
    items: List[Tuple[str, float]] = []
    for k, v in cat_map.items():
        try:
            total = float((v or {}).get("total", 0) or 0.0)
            client = float((v or {}).get("client", 0) or 0.0)
            pct = (client / max(total, 1.0)) * 100.0 if total > 0 else 0.0
            items.append((str(k), float(pct)))
        except Exception:
            continue
    if not items:
        return None
    items.sort(key=lambda x: x[1])
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 3.8))
    sns.barplot(x=vals, y=labels, orient="h", palette="Blues_r")
    plt.xlabel("SOV (%)")
    plt.xlim(0, 100)
    plt.tight_layout()
    out = _tmp_path("sov_by_category_")
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_sov_delta_by_category(delta_map: Dict[str, float]) -> Optional[str]:
    """
    Renderiza barras horizontales con Δ SOV (puntos porcentuales) vs. periodo previo.
    Espera estructura: { category: delta_pp_float }
    """
    if not delta_map or not isinstance(delta_map, dict):
        return None
    try:
        items = [(str(k), float(v or 0.0)) for k, v in delta_map.items()]
    except Exception:
        return None
    if not items:
        return None
    items.sort(key=lambda x: x[1])
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = ["#d9534f" if v < 0 else "#5cb85c" for v in vals]
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 3.8))
    sns.barplot(x=vals, y=labels, orient="h", palette=colors)
    plt.xlabel("Δ SOV (p.p.) vs periodo previo")
    plt.tight_layout()
    out = _tmp_path("sov_delta_by_category_")
    plt.savefig(out, dpi=160)
    plt.close()
    return out


# --- Nuevos gráficos para Correlaciones Transversales ---
def plot_category_sov_vs_sentiment_scatter(rows: List[Dict[str, float]], thresholds: Dict[str, float]) -> Optional[str]:
    """Dispersión SOV% vs Sentimiento por categoría con líneas de umbral.

    Espera elementos como: {"category": str, "sov_pct": float, "sentiment": float}
    """
    if not rows:
        return None
    try:
        import numpy as _np
        import matplotlib.pyplot as plt
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(6.4, 4.2))
        x = [float(r.get("sov_pct", 0.0)) for r in rows]
        y = [float(r.get("sentiment", 0.0)) for r in rows]
        labels = [str(r.get("category", "?")) for r in rows]
        plt.scatter(x, y, c="#0ea5e9")
        for xi, yi, lb in zip(x, y, labels):
            try:
                plt.text(xi + 0.3, yi + 0.01, lb, fontsize=7, alpha=0.8)
            except Exception:
                continue
        x_thr = float(thresholds.get("sov_pct", float(_np.median(x))))
        y_thr = float(thresholds.get("sentiment", float(_np.median(y))))
        plt.axvline(x_thr, color="#999", linestyle="--", linewidth=1)
        plt.axhline(y_thr, color="#999", linestyle="--", linewidth=1)
        plt.xlabel("SOV (%)")
        plt.ylabel("Sentimiento")
        plt.tight_layout()
        out = _tmp_path("sov_sent_scatter_")
        plt.savefig(out, dpi=160)
        plt.close()
        return out
    except Exception:
        return None


def plot_correlation_heatmap(labels: List[str], matrix: List[List[float]]) -> Optional[str]:
    """Heatmap de la matriz de correlaciones (r de Pearson)."""
    if not labels or not matrix:
        return None
    try:
        import numpy as np
        plt.figure(figsize=(4.2, 3.6))
        sns.heatmap(np.array(matrix), annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, vmin=-1, vmax=1, cmap="vlag")
        plt.tight_layout()
        out = _tmp_path("corr_heatmap_")
        plt.savefig(out, dpi=160)
        plt.close()
        return out
    except Exception:
        return None


def plot_lag_correlation(series: List[Dict[str, float]]) -> Optional[str]:
    """Curva r vs lag (días) para volumen -> sentimiento."""
    if not series:
        return None
    try:
        import matplotlib.pyplot as plt
        lags = [int(d.get("lag", 0)) for d in series if d.get("r") is not None]
        rs = [float(d.get("r", 0.0)) for d in series if d.get("r") is not None]
        if not lags:
            return None
        plt.figure(figsize=(6.0, 3.2))
        plt.plot(lags, rs, marker="o")
        plt.axhline(0, color="#999", linewidth=1)
        plt.xlabel("Lag (días, + volumen antes)")
        plt.ylabel("r de Pearson")
        plt.tight_layout()
        out = _tmp_path("lag_corr_")
        plt.savefig(out, dpi=160)
        plt.close()
        return out
    except Exception:
        return None
