"""
generate_report.py

Genera un PDF de evaluación completa por cada camión (truck_id) usando:
- XGBoostModel (clase del usuario)
- Matplotlib / SHAP para gráficas (guardadas como PNG)
- reportlab para ensamblar el PDF

Uso:
    from generate_report import generate_report_for_truck
    generate_report_for_truck("T-210", output_dir="reports")
"""

import os
import shutil
import tempfile
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from datetime import datetime

# Importa tu clase XGBoostModel según tu estructura de paquetes
# Ajusta la ruta si es necesario
from model.predictive.xgboost_model import XGBoostModel


# ---------------------------------------------------------
# Utilidades para gráficas y guardado
# ---------------------------------------------------------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_metrics_table_image(results: Dict[str, Any], out_path: str):
    """
    Dibuja una tabla con las métricas y la guarda como PNG.
    """
    metrics_4 = results["stage4"]["metrics"]
    metrics_8 = results["stage8"]["metrics"]
    rows = [
        ["Métrica", "Stage 4 (Vacío)", "Stage 8 (Lleno)"],
        [
            "R²",
            f"{metrics_4['R2']:.4f}",
            f"{metrics_8['R2']:.4f}",
        ],
        ["MAE (L)", f"{metrics_4['MAE']:.2f}", f"{metrics_8['MAE']:.2f}"],
        ["RMSE (L)", f"{metrics_4['RMSE']:.2f}", f"{metrics_8['RMSE']:.2f}"],
        ["MAPE (%)", f"{metrics_4['MAPE_Safe']:.2f}", f"{metrics_8['MAPE_Safe']:.2f}"],
        [
            "Median AE (L)",
            f"{metrics_4['MedianAE']:.2f}",
            f"{metrics_8['MedianAE']:.2f}",
        ],
        ["RMSLE", f"{metrics_4['RMSLE']:.4f}", f"{metrics_8['RMSLE']:.4f}"],
        [
            "ExplainedVar",
            f"{metrics_4['ExplainedVar']:.4f}",
            f"{metrics_8['ExplainedVar']:.4f}",
        ],
    ]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis("off")
    table = ax.table(cellText=rows, colWidths=[3.0, 2.0, 2.0], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title("Comparativa de métricas - Stage 4 vs Stage 8", fontsize=12, pad=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_comparison_bar(results: Dict[str, Any], out_path: str):
    """
    Gráfica comparativa simple (barras) de métricas seleccionadas.
    """
    labels = ["R2", "MAE", "RMSE", "MAPE"]
    metrics_4 = results["stage4"]["metrics"]
    metrics_8 = results["stage8"]["metrics"]

    val_4 = [
        metrics_4["R2"],
        metrics_4["MAE"],
        metrics_4["RMSE"],
        metrics_4["MAPE_Safe"],
    ]
    val_8 = [
        metrics_8["R2"],
        metrics_8["MAE"],
        metrics_8["RMSE"],
        metrics_8["MAPE_Safe"],
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, val_4, width, label="Stage 4", color="tab:blue")
    ax.bar(x + width / 2, val_8, width, label="Stage 8", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Comparación de Métricas")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_time_series(predictions_df: pd.DataFrame, out_path: str):
    """
    Serie temporal Predicho vs Real (usa PredictedFuelXGBoost y FuelConsumed).
    """
    df = predictions_df.sort_values("TimeStampIni").copy()
    # Convertir TimeStampIni a datetime si no lo es
    if not np.issubdtype(df["TimeStampIni"].dtype, np.datetime64):
        df["TimeStampIni"] = pd.to_datetime(df["TimeStampIni"])

    fig, ax = plt.subplots(figsize=(10, 3.5))
    df_stage4 = df[df["StageSequence"] == 4]
    df_stage8 = df[df["StageSequence"] == 8]

    if not df_stage4.empty:
        ax.plot(
            df_stage4["TimeStampIni"],
            df_stage4["FuelConsumed"],
            ".",
            label="Real (S4)",
            alpha=0.6,
        )
        ax.plot(
            df_stage4["TimeStampIni"],
            df_stage4["PredictedFuelXGBoost"],
            "o",
            markersize=3,
            label="Predicho (S4)",
            alpha=0.6,
            color="tab:blue",
        )

    if not df_stage8.empty:
        ax.plot(
            df_stage8["TimeStampIni"],
            df_stage8["FuelConsumed"],
            ".",
            label="Real (S8)",
            alpha=0.6,
        )
        ax.plot(
            df_stage8["TimeStampIni"],
            df_stage8["PredictedFuelXGBoost"],
            "o",
            markersize=3,
            label="Predicho (S8)",
            alpha=0.6,
            color="tab:red",
        )

    ax.set_ylabel("Combustible (L)")
    ax.set_xlabel("Timestamp")
    ax.set_title("Serie temporal: Real vs Predicho")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_and_residuals(
    test_y: np.ndarray, test_pred: np.ndarray, title: str, out_prefix: str
):
    """
    Genera scatter (real vs pred) y residuales y los guarda como imágenes.
    out_prefix es la ruta base sin extensión.
    """
    # Scatter real vs pred
    fig, ax = plt.subplots(figsize=(5.5, 4))
    residuals = test_y - test_pred
    sc = ax.scatter(test_y, test_pred, c=residuals, cmap="RdYlGn_r", s=30, alpha=0.7)
    minv = min(np.min(test_y), np.min(test_pred))
    maxv = max(np.max(test_y), np.max(test_pred))
    ax.plot([minv, maxv], [minv, maxv], "k--", lw=1)
    ax.set_xlabel("Real (L)")
    ax.set_ylabel("Predicho (L)")
    ax.set_title(f"{title} - Real vs Predicho")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Residual (L)")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_prefix + "_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Residuals vs Predicted
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.scatter(test_pred, residuals, alpha=0.6, s=30)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicho (L)")
    ax.set_ylabel("Residual (L)")
    ax.set_title(f"{title} - Residuales")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_prefix + "_residuals.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance_df(importance_df: pd.DataFrame, out_path: str, title: str):
    """
    Dibuja un barplot horizontal de importancia de features a partir de un DataFrame con columnas ['feature','importance'].
    """
    df = importance_df.copy()
    df_sorted = df.sort_values("importance", ascending=True).tail(30)  # top 30
    fig, ax = plt.subplots(figsize=(8, max(3, len(df_sorted) * 0.25)))
    ax.barh(df_sorted["feature"], df_sorted["importance"], color="tab:purple")
    ax.set_xlabel("Importancia")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_shap_summary_and_dependence(
    model, X_test: pd.DataFrame, base_path: str, stage_label: str
):
    """
    Genera SHAP summary plot y algunos dependence plots guardados como PNG.
    """
    # Prepara el explainer y shap_values
    explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
    # Algunos modelos XGBoost devuelven matrices; tratamos shap_values como numpy array
    shap_values = explainer.shap_values(X_test, check_additivity=False)

    # Summary plot (usa matplotlib)
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    fig.suptitle(f"SHAP Summary - {stage_label}", fontsize=12)
    plt.tight_layout()
    summary_path = os.path.join(base_path, f"shap_summary_{stage_label}.png")
    fig.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Dependence plots para top 3 features por importancia
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_idx = np.argsort(mean_abs)[::-1]
    top_idx = importance_idx[:3]
    dependence_paths = []
    for i, idx in enumerate(top_idx):
        feature_name = X_test.columns[idx]
        fig = plt.figure(figsize=(6, 4))
        shap.dependence_plot(feature_name, shap_values, X_test, show=False)
        plt.title(f"SHAP Dependence - {feature_name}")
        dep_path = os.path.join(base_path, f"shap_dependence_{stage_label}_{i+1}.png")
        plt.tight_layout()
        fig.savefig(dep_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        dependence_paths.append(dep_path)

    return summary_path, dependence_paths


# ---------------------------------------------------------
# Análisis automático (texto) generado desde resultados
# ---------------------------------------------------------
def generate_automatic_analysis(
    results: Dict[str, Any], predictions_df: pd.DataFrame
) -> Dict[str, str]:
    """
    Genera texto explicativo / análisis automático (resumen ejecutivo, interpretación de métricas,
    conclusiones operativas) en español.
    Devuelve un dict con secciones de texto.
    """
    m4 = results["stage4"]["metrics"]
    m8 = results["stage8"]["metrics"]
    samples4 = results["stage4"]["samples"]
    samples8 = results["stage8"]["samples"]

    # Resumen ejecutivo
    resumen = (
        "Resumen Ejecutivo:\n\n"
        "Este informe presenta la evaluación de modelos XGBoost diseñados para predecir "
        "el consumo de combustible en ciclos de carguío y acarreo de la mina San Cristóbal. "
        "Se entrenaron dos modelos especializados: uno para Stage 4 (camión vacío) y otro para "
        "Stage 8 (camión cargado). La separación en dos modelos se justifica por las diferencias "
        "operativas y físicas entre el tramo de retorno (vacío) y el tramo de acarreo (cargado). "
        f"Se utilizaron {samples4['train'] + samples4['test']} registros para Stage 4 "
        f"y {samples8['train'] + samples8['test']} registros para Stage 8.\n\n"
    )

    # Interpretación métrica automática (ejemplo)
    def interpret_metric(name, v4, v8):
        text = f"- {name}: Stage 4 = {v4:.4f} ; Stage 8 = {v8:.4f}. "
        if name == "R2":
            better = "Stage 8" if v8 > v4 else "Stage 4"
            text += f"R² mide la proporción de varianza explicada; mejor modelo: {better}.\n"
        else:
            # para errores: menor es mejor
            better = "Stage 4" if v4 < v8 else "Stage 8"
            text += f"Menor es mejor; mejor modelo: {better}.\n"
        return text

    metrics_analysis = "Interpretación de métricas:\n\n"
    metrics_analysis += interpret_metric("R2", m4["R2"], m8["R2"])
    metrics_analysis += interpret_metric("MAE", m4["MAE"], m8["MAE"])
    metrics_analysis += interpret_metric("RMSE", m4["RMSE"], m8["RMSE"])
    metrics_analysis += interpret_metric(
        "MAPE_Safe (MAPE %)", m4["MAPE_Safe"], m8["MAPE_Safe"]
    )

    # Sesgos y errores
    bias_text = (
        "\nAnálisis de sesgos y errores:\n\n"
        "Se observa la distribución de residuales para identificar sesgos sistemáticos.\n"
        "Si los residuales están centrados en cero y con varianza homogénea, el modelo no muestra "
        "sesgo evidente. Si existe asimetría o colas, recomendar recolección adicional de datos o "
        "transformaciones (por ejemplo, log) en rangos con altos errores.\n"
    )

    # Recomendaciones
    recommendations = (
        "\nRecomendaciones operativas y de modelado:\n\n"
        "1. Reentrenar periódicamente con datos recientes (p. ej. semanal o mensual) para captar "
        "cambios operacionales.\n"
        "2. Revisar variables con alta influencia SHAP: si provienen de sensores con ruido, "
        "considerar filtrado o imputación avanzada.\n"
        "3. Evaluar la inclusión de nuevas variables (pendiente por tramo, temperatura, estado de la "
        "carga) si están disponibles.\n"
        "4. Implementar alertas operacionales basadas en la diferencia (residual) > threshold (ej. 2 * MAE).\n"
    )

    conclusions = (
        "\nConclusiones:\n\n"
        "En resumen, ambos modelos entregan predicciones útiles para apoyar la gestión del consumo. "
        "Stage X (ver tabla) muestra mejor desempeño en la métrica R² y/o en errores absolutos, "
        "lo cual debe interpretarse junto con el volumen de datos y la variabilidad operativa.\n"
    )

    return {
        "resumen": resumen,
        "metrics_analysis": metrics_analysis,
        "bias_text": bias_text,
        "recommendations": recommendations,
        "conclusions": conclusions,
    }


# ---------------------------------------------------------
# Ensamblador del PDF con reportlab
# ---------------------------------------------------------
def build_pdf(
    report_path: str,
    truck_id: str,
    results: Dict[str, Any],
    predictions_df: pd.DataFrame,
    images: Dict[str, str],
    analysis_texts: Dict[str, str],
):
    """
    Genera el PDF final usando reportlab.
    images: diccionario con keys -> rutas de imagen
    analysis_texts: dict con textos
    """
    doc = SimpleDocTemplate(
        report_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    heading = styles["Heading2"]

    flow = []

    # Portada
    title_style = ParagraphStyle(
        "title", parent=styles["Title"], alignment=1, fontSize=18
    )
    flow.append(Paragraph(f"Evaluación de Modelos - Truck: {truck_id}", title_style))
    flow.append(Spacer(1, 12))
    flow.append(
        Paragraph(
            f"Fecha de generación: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            normal,
        )
    )
    flow.append(Spacer(1, 12))
    flow.append(
        Paragraph(
            "Informe automático: XGBoost dual (Stage 4 y Stage 8) — Consumo de combustible",
            normal,
        )
    )
    flow.append(Spacer(1, 24))

    # Resumen ejecutivo
    flow.append(Paragraph("Resumen Ejecutivo", heading))
    flow.append(Spacer(1, 6))
    for p in analysis_texts["resumen"].split("\n\n"):
        flow.append(Paragraph(p.replace("\n", "<br/>"), normal))
        flow.append(Spacer(1, 6))

    flow.append(PageBreak())

    # Métricas y tabla
    flow.append(Paragraph("Métricas de Rendimiento", heading))
    flow.append(Spacer(1, 6))
    if "metrics_table" in images:
        flow.append(Image(images["metrics_table"], width=16 * cm, height=6 * cm))
        flow.append(Spacer(1, 12))
    if "metrics_bar" in images:
        flow.append(Image(images["metrics_bar"], width=16 * cm, height=6 * cm))
        flow.append(Spacer(1, 12))

    flow.append(Paragraph("Interpretación de métricas", heading))
    flow.append(Spacer(1, 6))
    for p in analysis_texts["metrics_analysis"].split("\n"):
        if p.strip():
            flow.append(Paragraph(p.strip(), normal))
            flow.append(Spacer(1, 4))

    flow.append(Paragraph("Sesgos y errores", heading))
    flow.append(Spacer(1, 6))
    for p in analysis_texts["bias_text"].split("\n"):
        if p.strip():
            flow.append(Paragraph(p.strip(), normal))
            flow.append(Spacer(1, 4))

    flow.append(PageBreak())

    # Visualizaciones – Serie temporal
    flow.append(Paragraph("Series Temporales", heading))
    flow.append(Spacer(1, 8))
    if "time_series" in images:
        flow.append(Image(images["time_series"], width=16 * cm, height=6 * cm))
        flow.append(Spacer(1, 12))

    # Visualizaciones – Scatter y residuales (Stage 4 y 8)
    flow.append(Paragraph("Predicciones vs Reales y Residuales (Stage 4)", heading))
    flow.append(Spacer(1, 6))
    if "s4_scatter" in images:
        flow.append(Image(images["s4_scatter"], width=8 * cm, height=6 * cm))
        flow.append(Image(images["s4_residuals"], width=8 * cm, height=6 * cm))
        flow.append(Spacer(1, 12))

    flow.append(Paragraph("Predicciones vs Reales y Residuales (Stage 8)", heading))
    flow.append(Spacer(1, 6))
    if "s8_scatter" in images:
        flow.append(Image(images["s8_scatter"], width=8 * cm, height=6 * cm))
        flow.append(Image(images["s8_residuals"], width=8 * cm, height=6 * cm))
        flow.append(Spacer(1, 12))

    flow.append(PageBreak())

    # Feature importance
    flow.append(Paragraph("Importancia de Características", heading))
    flow.append(Spacer(1, 6))
    if "feat_imp_s4" in images:
        flow.append(Paragraph("Stage 4", styles["Heading3"]))
        flow.append(Image(images["feat_imp_s4"], width=16 * cm, height=8 * cm))
        flow.append(Spacer(1, 8))
    if "feat_imp_s8" in images:
        flow.append(Paragraph("Stage 8", styles["Heading3"]))
        flow.append(Image(images["feat_imp_s8"], width=16 * cm, height=8 * cm))
        flow.append(Spacer(1, 8))

    flow.append(PageBreak())

    # SHAP
    flow.append(Paragraph("Análisis SHAP (Interpretabilidad)", heading))
    flow.append(Spacer(1, 6))
    if "shap_s4_summary" in images:
        flow.append(Paragraph("Stage 4 - SHAP Summary", styles["Heading3"]))
        flow.append(Image(images["shap_s4_summary"], width=16 * cm, height=10 * cm))
        flow.append(Spacer(1, 8))
        # dependences
        for k in range(1, 4):
            key = f"shap_s4_dep_{k}"
            if key in images:
                flow.append(Image(images[key], width=16 * cm, height=6 * cm))
                flow.append(Spacer(1, 6))

    if "shap_s8_summary" in images:
        flow.append(Paragraph("Stage 8 - SHAP Summary", styles["Heading3"]))
        flow.append(Image(images["shap_s8_summary"], width=16 * cm, height=10 * cm))
        flow.append(Spacer(1, 8))
        for k in range(1, 4):
            key = f"shap_s8_dep_{k}"
            if key in images:
                flow.append(Image(images[key], width=16 * cm, height=6 * cm))
                flow.append(Spacer(1, 6))

    flow.append(PageBreak())

    # Recomendaciones y conclusiones
    flow.append(Paragraph("Recomendaciones y Conclusiones", heading))
    flow.append(Spacer(1, 6))
    for p in analysis_texts["recommendations"].split("\n"):
        if p.strip():
            flow.append(Paragraph(p.strip(), normal))
            flow.append(Spacer(1, 4))

    for p in analysis_texts["conclusions"].split("\n"):
        if p.strip():
            flow.append(Paragraph(p.strip(), normal))
            flow.append(Spacer(1, 4))

    # Construir PDF
    doc.build(flow)
    print(f"PDF generado en: {report_path}")


# ---------------------------------------------------------
# Función principal que orquesta todo
# ---------------------------------------------------------
def generate_report_for_truck(
    truck_id: str, output_dir: str = "reports", keep_images: bool = False
):
    """
    Genera reporte PDF completo por truck_id.
    - output_dir/reports_{truck_id}.pdf
    - guarda imágenes temporales en output_dir/{truck_id}/images/
    """
    # Rutas
    out_base = os.path.join(output_dir, f"{truck_id}")
    img_dir = os.path.join(out_base, "images")
    ensure_dir(img_dir)

    # 1) Entrenar / obtener modelo y resultados usando tu clase XGBoostModel
    print(f"⏳ Entrenando / cargando modelo para {truck_id} ...")
    model = XGBoostModel(
        truck_id=truck_id,
        numeric_predictor_vars=[],  # Opcional: si tu constructor requiere lista, ajusta
        categorical_vars=[],
    )

    # Cargar datos y entrenar (usa tus métodos)
    model.load_data()
    model.transform_cycles_data()
    results = model.train()  # retorna diccionario con métricas
    predictions_pl = model.get_predictions()  # DataFrame Polars
    predictions_df = predictions_pl.to_pandas()

    # 2) Generar imágenes (PNG)
    images = {}

    # 2.1 Tabla de métricas
    metrics_table_path = os.path.join(img_dir, "metrics_table.png")
    save_metrics_table_image(results, metrics_table_path)
    images["metrics_table"] = metrics_table_path

    # 2.2 Barra comparativa de métricas
    metrics_bar_path = os.path.join(img_dir, "metrics_bar.png")
    plot_metrics_comparison_bar(results, metrics_bar_path)
    images["metrics_bar"] = metrics_bar_path

    # 2.3 Serie temporal
    time_series_path = os.path.join(img_dir, "time_series.png")
    plot_time_series(predictions_df, time_series_path)
    images["time_series"] = time_series_path

    # 2.4 Gráficas Stage 4 y Stage 8 (scatter + residuals)
    # Stage 4
    if "stage4" in model.test_data and len(model.test_data["stage4"]["y_true"]) > 0:
        y4 = np.array(model.test_data["stage4"]["y_true"])
        p4 = np.array(model.test_data["stage4"]["y_pred"])
        plot_scatter_and_residuals(y4, p4, "Stage 4", os.path.join(img_dir, "s4"))
        images["s4_scatter"] = os.path.join(img_dir, "s4_scatter.png")
        images["s4_residuals"] = os.path.join(img_dir, "s4_residuals.png")
    else:
        print("⚠️ No hay datos de test para Stage 4")

    # Stage 8
    if "stage8" in model.test_data and len(model.test_data["stage8"]["y_true"]) > 0:
        y8 = np.array(model.test_data["stage8"]["y_true"])
        p8 = np.array(model.test_data["stage8"]["y_pred"])
        plot_scatter_and_residuals(y8, p8, "Stage 8", os.path.join(img_dir, "s8"))
        images["s8_scatter"] = os.path.join(img_dir, "s8_scatter.png")
        images["s8_residuals"] = os.path.join(img_dir, "s8_residuals.png")
    else:
        print("⚠️ No hay datos de test para Stage 8")

    # 2.5 Feature importance (desde tu método get_feature_importance)
    importance = model.get_feature_importance(stage="both")
    if "stage4" in importance:
        feat_s4_path = os.path.join(img_dir, "feat_imp_s4.png")
        plot_feature_importance_df(
            importance["stage4"], feat_s4_path, "Feature Importance - Stage 4"
        )
        images["feat_imp_s4"] = feat_s4_path
    if "stage8" in importance:
        feat_s8_path = os.path.join(img_dir, "feat_imp_s8.png")
        plot_feature_importance_df(
            importance["stage8"], feat_s8_path, "Feature Importance - Stage 8"
        )
        images["feat_imp_s8"] = feat_s8_path

    # 2.6 SHAP plots (summary + dependence) para Stage 4 y Stage 8 si hay datos
    try:
        if "stage4" in model.test_data and not model.test_data["stage4"]["X"].empty:
            Xs4 = model.test_data["stage4"]["X"]
            s4_sum, s4_deps = plot_shap_summary_and_dependence(
                model.model_stage4, Xs4, img_dir, "S4"
            )
            images["shap_s4_summary"] = s4_sum
            for i, p in enumerate(s4_deps, start=1):
                images[f"shap_s4_dep_{i}"] = p
    except Exception as e:
        print(f"⚠️ Error generando SHAP Stage 4: {e}")

    try:
        if "stage8" in model.test_data and not model.test_data["stage8"]["X"].empty:
            Xs8 = model.test_data["stage8"]["X"]
            s8_sum, s8_deps = plot_shap_summary_and_dependence(
                model.model_stage8, Xs8, img_dir, "S8"
            )
            images["shap_s8_summary"] = s8_sum
            for i, p in enumerate(s8_deps, start=1):
                images[f"shap_s8_dep_{i}"] = p
    except Exception as e:
        print(f"⚠️ Error generando SHAP Stage 8: {e}")

    # 3) Generar textos analíticos automáticos
    analysis_texts = generate_automatic_analysis(results, predictions_df)

    # 4) Ensamblar PDF
    report_path = os.path.join(out_base, f"{truck_id}_model_report.pdf")
    build_pdf(report_path, truck_id, results, predictions_df, images, analysis_texts)

    # 5) Manejo de imágenes temporales
    if not keep_images:
        try:
            # opcional: eliminar carpeta de imágenes
            shutil.rmtree(img_dir)
            print("Imágenes temporales eliminadas.")
        except Exception as e:
            print(f"⚠️ No se pudieron eliminar imágenes temporales: {e}")

    return report_path


# ---------------------------------------------------------
# Si se ejecuta directamente
# ---------------------------------------------------------
if __name__ == "__main__":
    # Ejemplo de uso:
    TRUCK_ID = "T-210"  # Cambia por el truck_id que quieras
    out = generate_report_for_truck(TRUCK_ID, output_dir="reports", keep_images=False)
    print("Reporte final generado en:", out)
