from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, PageBreak, Spacer
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.graphics.shapes import Drawing, String, Rect, Line
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart, HorizontalBarChart
from typing import Optional, Dict, Any
import numpy as np
from datetime import datetime
import io
import polars as pl

class ReportGenerator:
    def __init__(self, df: pl.DataFrame, truck_id: str, model_metrics: Optional[Dict[str, Any]] = None):
        self.df = df
        self.truck_id = truck_id
        self.styles = getSampleStyleSheet()
        self.model_metrics = model_metrics
        self._create_custom_styles()
        self.TRUCKS_METRICS = {
        "T-210": {
            "sin_carga": {"promedio_error_medio": 0.03, "desviacion_estandar_error": 1.32},
            "con_carga": {"promedio_error_medio": -0.94, "desviacion_estandar_error": 51.04},
        },
        "T-211": {
            "sin_carga": {"promedio_error_medio": 0.01, "desviacion_estandar_error": 1.69},
            "con_carga": {"promedio_error_medio": -1.07, "desviacion_estandar_error": 6.91},
        },
        "T-212": {
            "sin_carga": {"promedio_error_medio": 0.13, "desviacion_estandar_error": 1.50},
            "con_carga": {"promedio_error_medio": -0.05, "desviacion_estandar_error": 56.58},
        },
        "T-213": {
            "sin_carga": {"promedio_error_medio": 0.12, "desviacion_estandar_error": 0.86},
            "con_carga": {"promedio_error_medio": -0.97, "desviacion_estandar_error": 30.32},
        },
        "T-214": {
            "sin_carga": {"promedio_error_medio": 0.18, "desviacion_estandar_error": 1.34},
            "con_carga": {"promedio_error_medio": -0.67, "desviacion_estandar_error": 9.21},
        },
        "T-215": {
            "sin_carga": {"promedio_error_medio": 0.08, "desviacion_estandar_error": 1.04},
            "con_carga": {"promedio_error_medio": -1.36, "desviacion_estandar_error": 15.15},
        },
        "T-216": {
            "sin_carga": {"promedio_error_medio": 0.05, "desviacion_estandar_error": 0.54},
            "con_carga": {"promedio_error_medio": -0.84, "desviacion_estandar_error": 16.31},
        },
        "T-217": {
            "sin_carga": {"promedio_error_medio": 0.03, "desviacion_estandar_error": 0.95},
            "con_carga": {"promedio_error_medio": -0.27, "desviacion_estandar_error": 14.22},
        },
        "T-218": {
            "sin_carga": {"promedio_error_medio": 0.16, "desviacion_estandar_error": 1.47},
            "con_carga": {"promedio_error_medio": -0.84, "desviacion_estandar_error": 16.67},
        },
        "T-219": {
            "sin_carga": {"promedio_error_medio": 0.11, "desviacion_estandar_error": 1.57},
            "con_carga": {"promedio_error_medio": -0.57, "desviacion_estandar_error": 38.09},
        },
        "T-220": {
            "sin_carga": {"promedio_error_medio": -0.11, "desviacion_estandar_error": 1.68},
            "con_carga": {"promedio_error_medio": -0.84, "desviacion_estandar_error": 18.83},
        },
        "T-221": {
            "sin_carga": {"promedio_error_medio": 0.07, "desviacion_estandar_error": 0.55},
            "con_carga": {"promedio_error_medio": -0.39, "desviacion_estandar_error": 4.38},
        },
        "T-222": {
            "sin_carga": {"promedio_error_medio": -0.35, "desviacion_estandar_error": 2.35},
            "con_carga": {"promedio_error_medio": -0.93, "desviacion_estandar_error": 48.23},
        },
        "T-223": {
            "sin_carga": {"promedio_error_medio": -0.16, "desviacion_estandar_error": 2.12},
            "con_carga": {"promedio_error_medio": -1.38, "desviacion_estandar_error": 11.31},
        },
        "T-224": {
            "sin_carga": {"promedio_error_medio": 0.04, "desviacion_estandar_error": 1.27},
            "con_carga": {"promedio_error_medio": -1.22, "desviacion_estandar_error": 34.10},
        },
        "T-225": {
            "sin_carga": {"promedio_error_medio": -0.08, "desviacion_estandar_error": 1.65},
            "con_carga": {"promedio_error_medio": -0.27, "desviacion_estandar_error": 6.69},
        },
        "T-230": {
            "sin_carga": {"promedio_error_medio": 0.00, "desviacion_estandar_error": 0.17},
            "con_carga": {"promedio_error_medio": -0.66, "desviacion_estandar_error": 10.56},
        },
        "T-231": {
            "sin_carga": {"promedio_error_medio": 0.14, "desviacion_estandar_error": 1.89},
            "con_carga": {"promedio_error_medio": -0.20, "desviacion_estandar_error": 11.82},
        },
        "T-232": {
            "sin_carga": {"promedio_error_medio": -0.29, "desviacion_estandar_error": 2.75},
            "con_carga": {"promedio_error_medio": -3.44, "desviacion_estandar_error": 341.29},
        },
        "T-237": {
            "sin_carga": {"promedio_error_medio": -14.60, "desviacion_estandar_error": 1382.69},
            "con_carga": {"promedio_error_medio": -4.73, "desviacion_estandar_error": 155.81},
        },
        "T-233": {
            "sin_carga": {"promedio_error_medio": 0.13, "desviacion_estandar_error": 1.76},
            "con_carga": {"promedio_error_medio": -0.58, "desviacion_estandar_error": 7.89},
        },
        "T-236": {
            "sin_carga": {"promedio_error_medio": -0.43, "desviacion_estandar_error": 8.19},
            "con_carga": {"promedio_error_medio": -4.19, "desviacion_estandar_error": 190.83},
        },
        "T-238": {
            "sin_carga": {"promedio_error_medio": -0.21, "desviacion_estandar_error": 3.55},
            "con_carga": {"promedio_error_medio": -4.74, "desviacion_estandar_error": 295.53},
        },
        "T-240": {
            "sin_carga": {"promedio_error_medio": -0.20, "desviacion_estandar_error": 4.35},
            "con_carga": {"promedio_error_medio": -5.55, "desviacion_estandar_error": 223.20},
        },
        "T-241": {
            "sin_carga": {"promedio_error_medio": -1.27, "desviacion_estandar_error": 21.72},
            "con_carga": {"promedio_error_medio": -4.19, "desviacion_estandar_error": 164.15},
        },
        "T-242": {
            "sin_carga": {"promedio_error_medio": -0.13, "desviacion_estandar_error": 6.21},
            "con_carga": {"promedio_error_medio": -6.27, "desviacion_estandar_error": 285.41},
        },
        "T-243": {
            "sin_carga": {"promedio_error_medio": 0.06, "desviacion_estandar_error": 4.41},
            "con_carga": {"promedio_error_medio": -0.56, "desviacion_estandar_error": 304.31},
        },
    }

    def _create_custom_styles(self):
        for name, config in [
            ("CustomTitle", {"fontSize": 24, "textColor": colors.HexColor("#1f4788"), "spaceAfter": 30, "alignment": TA_CENTER}),
            ("SectionHeader", {"fontSize": 16, "textColor": colors.HexColor("#2c5aa0"), "spaceAfter": 12, "spaceBefore": 12}),
            ("ChartDescription", {"fontSize": 9, "textColor": colors.HexColor("#495057"), "spaceAfter": 8, "spaceBefore": 4, "leftIndent": 10, "rightIndent": 10})
        ]:
            self.styles.add(ParagraphStyle(name=name, parent=self.styles["Heading1" if name == "CustomTitle" else "Heading2" if name == "SectionHeader" else "Normal"], **config))

    def _calculate_daily_trends(self):
        return self.df.group_by([pl.col("TimeStampIni").dt.date().alias("Date"), pl.col("StageSequence")]).agg([
            pl.col("PredictedFuel").sum().alias("TotalPredictedFuel"),
            pl.col("PredictedFuel").mean().alias("AvgPredictedFuel"),
            pl.len().alias("Cycles")
        ]).sort("Date")

    def _calculate_monthly_trends(self):
        return self.df.with_columns([pl.col("TimeStampIni").dt.month().alias("Month"), pl.col("TimeStampIni").dt.year().alias("Year")]).group_by(["Year", "Month"]).agg([
            pl.col("PredictedFuel").sum().alias("TotalPredictedFuel"),
            pl.col("PredictedFuel").mean().alias("AvgPredictedFuel"),
            pl.len().alias("Cycles")
        ]).sort(["Year", "Month"])

    def _calculate_efficiency_metrics(self):
        return self.df.filter((pl.col("Distance") > 0) & (pl.col("TotalMeasuredTonnage").is_not_null()) & (pl.col("TotalMeasuredTonnage") > 0)).group_by("StageSequence").agg([
            pl.col("PredictedFuel").sum().alias("TotalFuel"),
            pl.col("Distance").sum().alias("TotalDistance"),
            pl.col("TotalMeasuredTonnage").sum().alias("TotalTonnage"),
            pl.len().alias("Cycles")
        ]).with_columns([
            (pl.col("TotalFuel") / (pl.col("TotalTonnage") * pl.col("TotalDistance") / 1000)).alias("L_per_ton_km"),
            (pl.col("TotalDistance") / pl.col("Cycles")).alias("AvgDistancePerCycle")
        ])

    def _detect_anomalies(self, threshold_percentile: float = 90):
        return self.df.filter(pl.col("PredictedFuel") > self.df["PredictedFuel"].quantile(threshold_percentile / 100)).sort("PredictedFuel", descending=True).head(20)

    def _calculate_stage_comparison(self):
        return self.df.group_by("StageSequence").agg([
            pl.col("PredictedFuel").mean().alias("AvgFuel"),
            pl.col("PredictedFuel").std().alias("StdFuel"),
            pl.col("PredictedFuel").min().alias("MinFuel"),
            pl.col("PredictedFuel").max().alias("MaxFuel"),
            pl.col("Distance").mean().alias("AvgDistance"),
            pl.col("SpeedAvg").mean().alias("AvgSpeed"),
            pl.len().alias("Cycles")
        ]).sort("StageSequence")

    def _calculate_kpis(self):
        return {
            "total_consumption": self.df["PredictedFuel"].sum(),
            "total_cycles": len(self.df.filter(pl.col("StageSequence") == 4)),
            "avg_consumption_per_cycle_st4": self.df.filter(pl.col("StageSequence") == 4)["PredictedFuel"].mean(),
            "avg_consumption_per_cycle_st8": self.df.filter(pl.col("StageSequence") == 8)["PredictedFuel"].mean(),
            "mean_error": self.TRUCKS_METRICS.get(self.truck_id, {}).get("sin_carga", {}).get("promedio_error_medio", None),
            "standard_deviation_error": self.TRUCKS_METRICS.get(self.truck_id, {}).get("sin_carga", {}).get("desviacion_estandar_error", None),
            "stage_consumption": self.df.group_by("StageSequence").agg(pl.col("PredictedFuel").sum().alias("TotalFuel")).sort("TotalFuel", descending=True),
            "best_cycle": self.df.sort("PredictedFuel").head(1),
            "worst_cycle": self.df.sort("PredictedFuel", descending=True).head(1)
        }

    def _create_daily_trend_chart(self):
        daily_pivot = self._calculate_daily_trends().pivot(index="Date", columns="StageSequence", values="AvgPredictedFuel").rename({"4": "Stage4_AvgFuel", "8": "Stage8_AvgFuel"}).sort("Date").tail(30)
        stage4_data, stage8_data = daily_pivot["Stage4_AvgFuel"].to_list(), daily_pivot["Stage8_AvgFuel"].to_list()
        all_values = stage4_data + stage8_data
        
        drawing = Drawing(450, 250)
        lc = HorizontalLineChart()
        lc.x, lc.y, lc.height, lc.width = 50, 70, 125, 350
        lc.data = [stage4_data, stage8_data]
        lc.joinedLines = 1
        lc.lines[0].strokeWidth, lc.lines[0].strokeColor = 2, colors.HexColor("#2c5aa0")
        lc.lines[1].strokeWidth, lc.lines[1].strokeColor = 2, colors.HexColor("#ff6b6b")
        lc.categoryAxis.categoryNames = [d.strftime('%m/%d') for d in daily_pivot["Date"].to_list()]
        lc.categoryAxis.labels.boxAnchor, lc.categoryAxis.labels.angle, lc.categoryAxis.labels.dy, lc.categoryAxis.labels.fontSize = 'n', 45, -5, 7
        lc.valueAxis.valueMin, lc.valueAxis.valueMax, lc.valueAxis.valueStep, lc.valueAxis.labels.fontSize = max(0, min(all_values) * 0.9), max(all_values) * 1.1, max(all_values) / 5, 8
        lc.valueAxis.labelTextFormat = lambda x: f'{x:.2f}'
        drawing.add(lc)
        drawing.add(String(225, 210, 'Comparativa diaria consumo promedio: camión vacío vs. camión lleno.', fontSize=10, textAnchor='middle', fontName='Helvetica-Bold'))
        for y, color, label in [(50, "#2c5aa0", "Stage 4 (Vacío)"), (40, "#ff6b6b", "Stage 8 (Cargado)")]:
            drawing.add(Line(65, y, 75, y, strokeColor=colors.HexColor(color), strokeWidth=2))
            drawing.add(String(80, y-5, label, fontSize=8, textAnchor='start', fillColor=colors.HexColor(color)))
        return drawing

    def _create_stage_comparison_chart(self):
        stage_stats = self._calculate_stage_comparison().to_pandas()
        avg_fuel = stage_stats['AvgFuel'].tolist()
        
        drawing = Drawing(450, 200)
        bc = VerticalBarChart()
        bc.x, bc.y, bc.height, bc.width = 100, 50, 125, 250
        bc.data = [avg_fuel]
        bc.strokeColor = colors.black
        bc.valueAxis.valueMin, bc.valueAxis.valueMax, bc.valueAxis.valueStep, bc.valueAxis.labels.fontSize = 0, max(avg_fuel) * 1.2, max(avg_fuel) / 5, 8
        bc.valueAxis.labelTextFormat = lambda x: f'{x:.2f}'
        bc.categoryAxis.categoryNames = ['Stage 4\n(Vacío)', 'Stage 8\n(Cargado)']
        bc.categoryAxis.labels.boxAnchor, bc.categoryAxis.labels.dy, bc.categoryAxis.labels.fontSize = 'n', -10, 9
        bc.bars[0].fillColor = colors.HexColor("#4dabf7")
        if len(bc.bars) > 1: bc.bars[1].fillColor = colors.HexColor("#ff922b")
        
        drawing.add(bc)
        drawing.add(String(225, 185, 'Consumo Promedio por Etapa', fontSize=10, textAnchor='middle', fontName='Helvetica-Bold'))
        return drawing

    def _create_hourly_distribution_chart(self):
        hourly = self.df.with_columns(pl.col("TimeStampIni").dt.hour().alias("Hour")).group_by("Hour").agg(pl.col("PredictedFuel").sum().alias("TotalFuel")).sort("Hour").to_pandas()
        fuel_by_hour = [dict(zip(hourly['Hour'], hourly['TotalFuel'])).get(h, 0) for h in range(24)]
        
        drawing = Drawing(450, 200)
        bc = VerticalBarChart()
        bc.x, bc.y, bc.height, bc.width = 50, 50, 125, 350
        bc.data = [fuel_by_hour]
        bc.strokeColor = colors.black
        bc.valueAxis.valueMin, bc.valueAxis.valueMax, bc.valueAxis.valueStep, bc.valueAxis.labels.fontSize = 0, max(fuel_by_hour) * 1.2, max(fuel_by_hour) / 4, 7
        bc.valueAxis.labelTextFormat = lambda x: f'{x:.2f}'
        bc.categoryAxis.categoryNames = [str(h) if h % 2 == 0 else '' for h in range(24)]
        bc.categoryAxis.labels.fontSize, bc.categoryAxis.labels.dy = 7, -5
        bc.bars[0].fillColor = colors.HexColor("#20c997")
        
        drawing.add(bc)
        drawing.add(String(225, 185, 'Distribución de Consumo por Hora del Día', fontSize=10, textAnchor='middle', fontName='Helvetica-Bold'))
        return drawing


    def _detect_anomalies_by_stage(self, stage: int, threshold_percentile: float = 90):
        """Detect anomalies for specific stage"""
        df_stage = self.df.filter(pl.col("StageSequence") == stage)
        if df_stage.is_empty():
            return None
            
        threshold = df_stage["PredictedFuel"].quantile(threshold_percentile / 100)
        anomalies = df_stage.filter(pl.col("PredictedFuel") > threshold)\
            .sort("PredictedFuel", descending=True)\
            .head(20)
        
        return anomalies
            
    def _create_destination_chart(self, stage: int):
        """Create destination chart for specific stage (4 or 8)"""
        stage_name = "Vacío" if stage == 4 else "Cargado"
        dest_consumption = self.df.filter(
            (pl.col("Destination").is_not_null()) & 
            (pl.col("StageSequence") == stage)
        ).group_by("Destination").agg(
            pl.col("PredictedFuel").sum().alias("TotalFuel")
        ).sort("TotalFuel", descending=True).head(10).to_pandas()
        
        if dest_consumption.empty: 
            return None
        
        fuel_data = dest_consumption['TotalFuel'].tolist()
        drawing = Drawing(450, 250)
        bc = VerticalBarChart()
        bc.x, bc.y, bc.height, bc.width = 50, 70, 150, 350
        bc.data = [fuel_data]
        bc.strokeColor = colors.black
        bc.valueAxis.valueMin, bc.valueAxis.valueMax, bc.valueAxis.valueStep, bc.valueAxis.labels.fontSize = 0, max(fuel_data) * 1.2, max(fuel_data) / 5, 7
        bc.valueAxis.labelTextFormat = lambda x: f'{x:.2f}'
        bc.categoryAxis.categoryNames = [str(d)[:12] for d in dest_consumption['Destination']]
        bc.categoryAxis.labels.boxAnchor, bc.categoryAxis.labels.angle, bc.categoryAxis.labels.fontSize = 'ne', 45, 7
        bc.bars[0].fillColor = colors.HexColor("#4dabf7") if stage == 4 else colors.HexColor("#ff922b")
        
        drawing.add(bc)
        drawing.add(String(225, 230, f'Top 10 Destinos por Consumo - Ciclo {stage_name} (Stage {stage})', 
                fontSize=10, textAnchor='middle', fontName='Helvetica-Bold'))
        return drawing
        
    def _create_efficiency_trend_chart(self):
        monthly = self._calculate_monthly_trends().to_pandas()
        if len(monthly) < 2: return None
        monthly = monthly.tail(12)
        avg_fuel = monthly['AvgPredictedFuel'].tolist()

        drawing = Drawing(450, 200)
        lc = HorizontalLineChart()
        lc.x, lc.y, lc.height, lc.width = 50, 50, 125, 350
        lc.data = [avg_fuel]
        lc.joinedLines = 1
        lc.lines[0].strokeWidth, lc.lines[0].strokeColor = 2, colors.HexColor("#e03131")
        lc.categoryAxis.categoryNames = [f"M{int(w)}" for w in monthly['Month']]
        lc.categoryAxis.labels.boxAnchor, lc.categoryAxis.labels.fontSize, lc.categoryAxis.labels.dy = 'n', 7, -5
        lc.valueAxis.valueMin, lc.valueAxis.valueMax, lc.valueAxis.valueStep, lc.valueAxis.labels.fontSize = min(avg_fuel) * 0.9, max(avg_fuel) * 1.1, (max(avg_fuel) - min(avg_fuel)) / 4, 7
        lc.valueAxis.labelTextFormat = lambda x: f'{x:.2f}'
        drawing.add(lc)
        drawing.add(String(225, 185, 'Tendencia Mensual de Eficiencia', fontSize=10, textAnchor='middle', fontName='Helvetica-Bold'))
        return drawing

    def _create_model_metrics_comparison_chart(self):
        if not self.model_metrics: return None
        s4, s8 = self.model_metrics.get('stage4', {}).get('metrics', {}), self.model_metrics.get('stage8', {}).get('metrics', {})
        if not s4 or not s8: return None
        
        r2_data = [s4.get('R2', 0), s8.get('R2', 0)]
        drawing = Drawing(450, 220)
        bc = VerticalBarChart()
        bc.x, bc.y, bc.height, bc.width = 100, 50, 130, 250
        bc.data = [r2_data]
        bc.strokeColor = colors.black
        bc.valueAxis.valueMin, bc.valueAxis.valueMax, bc.valueAxis.valueStep, bc.valueAxis.labels.fontSize = 0, 1.0, 0.2, 9
        bc.categoryAxis.categoryNames = ['Stage 4\n(Vacío)', 'Stage 8\n(Cargado)']
        bc.categoryAxis.labels.boxAnchor, bc.categoryAxis.labels.dy, bc.categoryAxis.labels.fontSize = 'n', -10, 10
        bc.bars[0].fillColor, bc.bars[1].fillColor = colors.HexColor("#4dabf7"), colors.HexColor("#ff922b")
        bc.valueAxis.labelTextFormat = lambda x: f'{x:.2f}'
        
        drawing.add(bc)
        drawing.add(String(225, 195, 'Coeficiente de Determinación (R²)', fontSize=11, textAnchor='middle', fontName='Helvetica-Bold'))
        for i, val in enumerate(r2_data):
            drawing.add(String(162 + i*125, 50 + val * 130 + 5, f"{val:.3f}", fontSize=9, textAnchor='middle', fontName='Helvetica-Bold'))
        return drawing

    def _create_error_metrics_chart(self):
        if not self.model_metrics: return None
        s4, s8 = self.model_metrics.get('stage4', {}).get('metrics', {}), self.model_metrics.get('stage8', {}).get('metrics', {})
        if not s4 or not s8: return None
        
        mae_data, rmse_data = [s4.get('MAE', 0), s8.get('MAE', 0)], [s4.get('RMSE', 0), s8.get('RMSE', 0)]
        drawing = Drawing(450, 220)
        bc = VerticalBarChart()
        bc.x, bc.y, bc.height, bc.width = 80, 50, 130, 300
        bc.data = [mae_data, rmse_data]
        bc.strokeColor = colors.black
        max_val = max(max(mae_data), max(rmse_data))
        bc.valueAxis.valueMin, bc.valueAxis.valueMax, bc.valueAxis.valueStep, bc.valueAxis.labels.fontSize = 0, max_val * 1.2, max_val / 5, 9
        bc.valueAxis.labelTextFormat = lambda x: f'{x:.2f}'
        bc.categoryAxis.categoryNames, bc.categoryAxis.labels.fontSize, bc.categoryAxis.labels.dy = ['Stage 4', 'Stage 8'], 10, -5
        bc.bars[0].fillColor, bc.bars[1].fillColor = colors.HexColor("#51cf66"), colors.HexColor("#ff6b6b")
        bc.barWidth, bc.groupSpacing = 25, 40
        
        drawing.add(bc)
        drawing.add(String(225, 195, 'Métricas de Error: MAE vs RMSE (Litros)', fontSize=11, textAnchor='middle', fontName='Helvetica-Bold'))
        for x, color, label in [(60, "#51cf66", "MAE"), (150, "#ff6b6b", "RMSE")]:
            drawing.add(Rect(x, 25, 15, 8, fillColor=colors.HexColor(color), strokeColor=colors.black))
            drawing.add(String(x+20, 29, label, fontSize=9, textAnchor='start'))
        return drawing

    def _create_mape_chart(self):
        if not self.model_metrics: return None
        s4, s8 = self.model_metrics.get('stage4', {}).get('metrics', {}), self.model_metrics.get('stage8', {}).get('metrics', {})
        if not s4 or not s8: return None
        
        mape_data = [s4.get('MAPE_Safe', 0), s8.get('MAPE_Safe', 0)]
        drawing = Drawing(450, 220)
        bc = VerticalBarChart()
        bc.x, bc.y, bc.height, bc.width = 100, 50, 130, 250
        bc.data = [mape_data]
        bc.strokeColor = colors.black
        bc.valueAxis.valueMin, bc.valueAxis.valueMax, bc.valueAxis.valueStep, bc.valueAxis.labels.fontSize = 0, max(mape_data) * 1.3, max(mape_data) / 5, 9
        bc.valueAxis.labelTextFormat = lambda x: f'{x:.2f}'
        bc.categoryAxis.categoryNames, bc.categoryAxis.labels.fontSize, bc.categoryAxis.labels.dy = ['Stage 4', 'Stage 8'], 10, -5
        bc.bars[0].fillColor, bc.bars[1].fillColor = colors.HexColor("#845ef7"), colors.HexColor("#ff6b6b")
        
        drawing.add(bc)
        drawing.add(String(225, 195, 'Error Porcentual Absoluto Medio (MAPE %)', fontSize=11, textAnchor='middle', fontName='Helvetica-Bold'))
        for i, val in enumerate(mape_data):
            drawing.add(String(162 + i*125, 50 + (val / bc.valueAxis.valueMax * 130) + 5, f"{val:.1f}%", fontSize=9, textAnchor='middle', fontName='Helvetica-Bold'))
        return drawing

    def _create_feature_importance_chart(self, stage: str = 'stage4', top_n: int = 10):
        if not self.model_metrics or stage not in self.model_metrics: return None
        df_stage = self.df.filter(pl.col("StageSequence") == (4 if stage == 'stage4' else 8))
        if len(df_stage) == 0: return None
        
        available_cols = [c for c in ['SpeedAvg', 'Distance', 'CycleDurationSeconds', 'TimeEfficiencyPercentage', 'TotalMeasuredTonnage'] if c in df_stage.columns]
        if not available_cols: return None
        
        correlations = []
        for col in available_cols:
            try:
                corr = df_stage.select([pl.corr(col, 'PredictedFuel').alias('corr')])['corr'][0]
                if corr is not None and not np.isnan(corr): correlations.append((col, abs(corr)))
            except: continue
        
        correlations = sorted(correlations, key=lambda x: x[1], reverse=True)[:top_n]
        if not correlations: return None
        
        importance_values, feature_names = [c[1] for c in correlations], [c[0] for c in correlations]
        drawing = Drawing(450, 280)
        bc = HorizontalBarChart()
        bc.x, bc.y, bc.height, bc.width = 120, 50, 180, 280
        bc.data = [importance_values]
        bc.strokeColor = colors.black
        bc.valueAxis.valueMin, bc.valueAxis.valueMax, bc.valueAxis.valueStep, bc.valueAxis.labels.fontSize = 0, max(importance_values) * 1.1, max(importance_values) * 1.1 / 5, 8
        bc.valueAxis.labelTextFormat = lambda x: f'{x:.2f}'
        bc.categoryAxis.categoryNames, bc.categoryAxis.labels.fontSize, bc.categoryAxis.labels.dx = feature_names[::-1], 8, -5
        bc.bars[0].fillColor = colors.HexColor("#228be6")
        
        drawing.add(bc)
        stage_name = 'Stage 4 (Vacío)' if stage == 'stage4' else 'Stage 8 (Cargado)'
        drawing.add(String(260, 255, f'Importancia de Variables - {stage_name}', fontSize=11, textAnchor='middle', fontName='Helvetica-Bold'))
        drawing.add(String(260, 240, '(Correlación con consumo predicho)', fontSize=9, textAnchor='middle', fontName='Helvetica'))
        return drawing

    def generate_pdf_report(self, output_path: Optional[str] = None):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        elements = []

        elements.append(Paragraph(f"Informe de Análisis del Consumo de Combustible<br/>{self.truck_id}", self.styles["CustomTitle"]))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(f"<b>Reporte generado:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/><b>Período de análisis:</b> {self.df['TimeStampIni'].min().strftime('%Y-%m-%d')} a {self.df['TimeStampIni'].max().strftime('%Y-%m-%d')}<br/><b>Total de registros:</b> {len(self.df):,}", self.styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))

        elements.append(Paragraph("1. Resumen General del modelo", self.styles["SectionHeader"]))
        elements.append(Paragraph(f"Esta sección presenta los resultados generales del modelo de consumo de combustible, para el periodo de análisis {self.df['TimeStampIni'].min().strftime('%Y-%m-%d')} a {self.df['TimeStampIni'].max().strftime('%Y-%m-%d')}.", self.styles["ChartDescription"]))
        elements.append(Spacer(1, 0.15 * inch))

        kpis = self._calculate_kpis()
        kpi_table = Table([["Métrica", "Valor"], ["Consumo Total de Combustible", f"{kpis['total_consumption']:.2f} L"], ["Total de Ciclos", f"{kpis['total_cycles']:,}"], ["Consumo Promedio del Ciclo con camión vacío", f"{kpis['avg_consumption_per_cycle_st4']:.2f} L"], ["Consumo Promedio del Ciclo con camión lleno", f"{kpis['avg_consumption_per_cycle_st8']:.2f} L"], ["Error Medio en las predicciones", f"{kpis['mean_error']:.2f} %"], ["Desviación Estándar del Error en las predicciones", f"{kpis['standard_deviation_error']:.2f} %"]], colWidths=[3*inch, 2*inch])
        kpi_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2c5aa0")), ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke), ("ALIGN", (0,0), (-1,-1), "LEFT"), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,0), 12), ("BOTTOMPADDING", (0,0), (-1,0), 12), ("BACKGROUND", (0,1), (-1,-1), colors.beige), ("GRID", (0,0), (-1,-1), 1, colors.black)]))
        elements.extend([kpi_table, Spacer(1, 0.3*inch)])

        elements.append(Paragraph("2. Tendencias de Consumo", self.styles["SectionHeader"]))
        elements.append(Paragraph("Esta gráfica de líneas compara el consumo promedio diario de combustible entre dos etapas operacionales: (camión vacío, línea azul) y (camión lleno, línea roja). Permite vizualizar la tendencia de consumo de combustible de los últimos 30 días.", self.styles["ChartDescription"]))
        elements.extend([Spacer(1, 0.1*inch), self._create_daily_trend_chart(), Spacer(1, 0.3*inch)])

        # Daily trend tables separated by stage
        daily_trends_st4 = self._calculate_daily_trends().filter(pl.col("StageSequence") == 4).to_pandas()
        daily_trends_st8 = self._calculate_daily_trends().filter(pl.col("StageSequence") == 8).to_pandas()
        
        elements.append(Paragraph("2.1 Tendencia Diaria - Ciclo Vacío (Stage 4)", self.styles["Normal"]))
        daily_table_st4 = Table([["Fecha", "Combustible Total (L)", "Promedio/Ciclo (L)", "Ciclos"]] + [[str(row["Date"]), f"{row['TotalPredictedFuel']:.2f}", f"{row['AvgPredictedFuel']:.2f}", f"{row['Cycles']}"] for _, row in daily_trends_st4.tail(10).iterrows()], colWidths=[1.5*inch]*4)
        daily_table_st4.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#4dabf7")), ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke), ("ALIGN", (0,0), (-1,-1), "CENTER"), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 9), ("BOTTOMPADDING", (0,0), (-1,0), 12), ("GRID", (0,0), (-1,-1), 1, colors.black)]))
        elements.extend([daily_table_st4, Spacer(1, 0.3*inch)])
        
        elements.append(Paragraph("2.2 Tendencia Diaria - Ciclo Cargado (Stage 8)", self.styles["Normal"]))
        daily_table_st8 = Table([["Fecha", "Combustible Total (L)", "Promedio/Ciclo (L)", "Ciclos"]] + [[str(row["Date"]), f"{row['TotalPredictedFuel']:.2f}", f"{row['AvgPredictedFuel']:.2f}", f"{row['Cycles']}"] for _, row in daily_trends_st8.tail(10).iterrows()], colWidths=[1.5*inch]*4)
        daily_table_st8.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#ff922b")), ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke), ("ALIGN", (0,0), (-1,-1), "CENTER"), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 9), ("BOTTOMPADDING", (0,0), (-1,0), 12), ("GRID", (0,0), (-1,-1), 1, colors.black)]))
        elements.extend([daily_table_st8, Spacer(1, 0.3*inch)])

        # Destination charts separated by stage
        elements.append(Paragraph("2.3 Zonas de carga con mas consumo", self.styles["SectionHeader"]))
        elements.append(Paragraph("Este gráfico de barras verticales muestra los 10 zonas de carga con mayor consumo total de combustible para el ciclo vacío.", self.styles["ChartDescription"]))
        dest_chart_st4 = self._create_destination_chart(4)
        if dest_chart_st4 is not None: 
            elements.extend([Spacer(1, 0.1*inch), dest_chart_st4, Spacer(1, 0.3*inch)])
        else:
            elements.append(Paragraph("No hay datos de destinos para Stage 4.", self.styles["Normal"]))

        elements.append(Paragraph("2.4 Zonas de descarga con mas consumo", self.styles["SectionHeader"]))
        elements.append(Paragraph("Este gráfico de barras verticales muestra los 10 zonas de descarga con mayor consumo total de combustible para el ciclo cargado.", self.styles["ChartDescription"]))
        dest_chart_st8 = self._create_destination_chart(8)
        if dest_chart_st8 is not None: 
            elements.extend([Spacer(1, 0.1*inch), dest_chart_st8, Spacer(1, 0.3*inch)])
        else:
            elements.append(Paragraph("No hay datos de destinos para Stage 8.", self.styles["Normal"]))

        elements.append(Paragraph("3. Consumo promedio por Etapa", self.styles["SectionHeader"]))
        elements.extend([Spacer(1, 0.1*inch), self._create_stage_comparison_chart(), Spacer(1, 0.3*inch)])

        stage_stats_st4 = self._calculate_stage_comparison().filter(pl.col("StageSequence") == 4).to_pandas()
        stage_stats_st8 = self._calculate_stage_comparison().filter(pl.col("StageSequence") == 8).to_pandas()
        
        elements.append(Paragraph("3.1 Estadísticas del consumo durante ciclo vacío", self.styles["Normal"]))
        stage_table_st4 = Table([["Promedio (L)", "Desv. Est.", "Mín", "Máx", "Vel. Prom.", "Ciclos"]] + [[f"{row['AvgFuel']:.2f}", f"{row['StdFuel']:.2f}", f"{row['MinFuel']:.2f}", f"{row['MaxFuel']:.2f}", f"{row['AvgSpeed']:.1f} km/h", f"{row['Cycles']}"] for _, row in stage_stats_st4.iterrows()], colWidths=[1.2*inch]*6)
        stage_table_st4.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#4dabf7")), ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke), ("ALIGN", (0,0), (-1,-1), "CENTER"), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 8), ("GRID", (0,0), (-1,-1), 1, colors.black)]))
        elements.extend([stage_table_st4, Spacer(1, 0.2*inch)])
        
        elements.append(Paragraph("3.2 Estadísticas del consumo durante ciclo lleno", self.styles["Normal"]))
        stage_table_st8 = Table([["Promedio (L)", "Desv. Est.", "Mín", "Máx", "Vel. Prom.", "Ciclos"]] + [[f"{row['AvgFuel']:.2f}", f"{row['StdFuel']:.2f}", f"{row['MinFuel']:.2f}", f"{row['MaxFuel']:.2f}", f"{row['AvgSpeed']:.1f} km/h", f"{row['Cycles']}"] for _, row in stage_stats_st8.iterrows()], colWidths=[1.2*inch]*6)
        stage_table_st8.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#ff922b")), ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke), ("ALIGN", (0,0), (-1,-1), "CENTER"), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 8), ("GRID", (0,0), (-1,-1), 1, colors.black)]))
        elements.extend([stage_table_st8, Spacer(1, 0.3*inch)])

        elements.append(Paragraph("4. Distribución Horaria", self.styles["SectionHeader"]))
        elements.append(Paragraph("Estos histogramas muestran la distribución del consumo total de combustible a lo largo de las 24 horas del día.", self.styles["ChartDescription"]))
        elements.extend([Spacer(1, 0.1*inch)])
        
        hourly_st4 = self.df.filter(pl.col("StageSequence") == 4).with_columns(pl.col("TimeStampIni").dt.hour().alias("Hour")).group_by("Hour").agg(pl.col("PredictedFuel").sum().alias("TotalFuel")).sort("Hour").to_pandas()
        fuel_by_hour_st4 = [dict(zip(hourly_st4['Hour'], hourly_st4['TotalFuel'])).get(h, 0) for h in range(24)]
        
        drawing_st4 = Drawing(450, 200)
        bc_st4 = VerticalBarChart()
        bc_st4.x, bc_st4.y, bc_st4.height, bc_st4.width = 50, 50, 125, 350
        bc_st4.data = [fuel_by_hour_st4]
        bc_st4.strokeColor = colors.black
        bc_st4.valueAxis.valueMin, bc_st4.valueAxis.valueMax, bc_st4.valueAxis.valueStep, bc_st4.valueAxis.labels.fontSize = 0, max(fuel_by_hour_st4) * 1.2 if fuel_by_hour_st4 else 1, (max(fuel_by_hour_st4) / 4) if fuel_by_hour_st4 else 1, 7
        bc_st4.valueAxis.labelTextFormat = lambda x: f'{x:.2f}'
        bc_st4.categoryAxis.categoryNames = [str(h) if h % 2 == 0 else '' for h in range(24)]
        bc_st4.categoryAxis.labels.fontSize, bc_st4.categoryAxis.labels.dy = 7, -5
        bc_st4.bars[0].fillColor = colors.HexColor("#4dabf7")
        drawing_st4.add(bc_st4)
        drawing_st4.add(String(225, 185, 'Distribución Horaria - Ciclo vacío', fontSize=10, textAnchor='middle', fontName='Helvetica-Bold'))
        elements.extend([drawing_st4, Spacer(1, 0.3*inch)])
        
        hourly_st8 = self.df.filter(pl.col("StageSequence") == 8).with_columns(pl.col("TimeStampIni").dt.hour().alias("Hour")).group_by("Hour").agg(pl.col("PredictedFuel").sum().alias("TotalFuel")).sort("Hour").to_pandas()
        fuel_by_hour_st8 = [dict(zip(hourly_st8['Hour'], hourly_st8['TotalFuel'])).get(h, 0) for h in range(24)]
        
        drawing_st8 = Drawing(450, 200)
        bc_st8 = VerticalBarChart()
        bc_st8.x, bc_st8.y, bc_st8.height, bc_st8.width = 50, 50, 125, 350
        bc_st8.data = [fuel_by_hour_st8]
        bc_st8.strokeColor = colors.black
        bc_st8.valueAxis.valueMin, bc_st8.valueAxis.valueMax, bc_st8.valueAxis.valueStep, bc_st8.valueAxis.labels.fontSize = 0, max(fuel_by_hour_st8) * 1.2 if fuel_by_hour_st8 else 1, (max(fuel_by_hour_st8) / 4) if fuel_by_hour_st8 else 1, 7
        bc_st8.valueAxis.labelTextFormat = lambda x: f'{x:.2f}'
        bc_st8.categoryAxis.categoryNames = [str(h) if h % 2 == 0 else '' for h in range(24)]
        bc_st8.categoryAxis.labels.fontSize, bc_st8.categoryAxis.labels.dy = 7, -5
        bc_st8.bars[0].fillColor = colors.HexColor("#ff922b")
        drawing_st8.add(bc_st8)
        drawing_st8.add(String(225, 185, 'Distribución Horaria - Ciclo lleno', fontSize=10, textAnchor='middle', fontName='Helvetica-Bold'))
        elements.extend([drawing_st8, Spacer(1, 0.3*inch)])

        # --- SECTION 5: High Consumption Events ---
        elements.append(Paragraph("5. Eventos de alto consumo", self.styles["SectionHeader"]))
        
        efficiency_chart = self._create_efficiency_trend_chart()
        if efficiency_chart is not None:
            elements.append(Paragraph("Esta gráfica de líneas muestra la tendencia mensual del consumo promedio de combustible. Para identificar los eventos de alto consumo", self.styles["ChartDescription"]))
            elements.extend([Spacer(1, 0.1*inch), efficiency_chart, Spacer(1, 0.3*inch)])

        # Anomalies for Stage 4
        elements.append(Paragraph("5.1 Eventos de Alto Consumo durante ciclo vacio", self.styles["Normal"]))
        anomalies_st4_df = self._detect_anomalies_by_stage(4)
        if anomalies_st4_df is not None and not anomalies_st4_df.is_empty():
            anomalies_st4 = anomalies_st4_df.to_pandas()
            anom_data_st4 = [["Fecha/Hora", "Combustible (L)", "Distancia (m)", "Destino"]]
            for _, row in anomalies_st4.head(15).iterrows():
                anom_data_st4.append([
                    row["TimeStampIni"].strftime("%Y-%m-%d %H:%M"),
                    f"{row['PredictedFuel']:.2f}",
                    f"{row['Distance']:.0f}",
                    str(row["Destination"])[:15]
                ])
            anom_table_st4 = Table(anom_data_st4, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.5*inch])
            anom_table_st4.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#4dabf7")),
                ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
                ("ALIGN", (0,0), (-1,-1), "CENTER"),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE", (0,0), (-1,-1), 8),
                ("GRID", (0,0), (-1,-1), 1, colors.black)
            ]))
            elements.append(anom_table_st4)
        else:
            elements.append(Paragraph("No se detectaron anomalías para Stage 4.", self.styles["Normal"]))
        
        elements.append(Spacer(1, 0.3*inch))

        # Anomalies for Stage 8
        elements.append(Paragraph("5.2 Eventos de Alto Consumo durante el ciclo lleno", self.styles["Normal"]))
        anomalies_st8_df = self._detect_anomalies_by_stage(8)
        if anomalies_st8_df is not None and not anomalies_st8_df.is_empty():
            anomalies_st8 = anomalies_st8_df.to_pandas()
            anom_data_st8 = [["Fecha/Hora", "Combustible (L)", "Distancia (m)", "Destino"]]
            for _, row in anomalies_st8.head(15).iterrows():
                anom_data_st8.append([
                    row["TimeStampIni"].strftime("%Y-%m-%d %H:%M"),
                    f"{row['PredictedFuel']:.2f}",
                    f"{row['Distance']:.0f}",
                    str(row["Destination"])[:15]
                ])
            anom_table_st8 = Table(anom_data_st8, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.5*inch])
            anom_table_st8.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#ff922b")),
                ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
                ("ALIGN", (0,0), (-1,-1), "CENTER"),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE", (0,0), (-1,-1), 8),
                ("GRID", (0,0), (-1,-1), 1, colors.black)
            ]))
            elements.append(anom_table_st8)
        else:
            elements.append(Paragraph("No se detectaron anomalías para Stage 8.", self.styles["Normal"]))
        elements.append(Spacer(1, 0.3*inch))
        
        elements.append(Paragraph("6. Distribución de Combustible por Etapa", self.styles["SectionHeader"]))
        elements.append(Paragraph("Esta tabla resume la distribución total de combustible entre las diferentes etapas, mostrando tanto el volumen absoluto en litros como el porcentaje relativo del consumo total.", self.styles["ChartDescription"]))
        elements.append(Spacer(1, 0.1*inch))

        stage_consumption = self.df.group_by("StageSequence").agg(pl.col("PredictedFuel").sum().alias("TotalFuel")).to_pandas()
        total_fuel = stage_consumption["TotalFuel"].sum()
        consumption_table = Table([["Etapa", "Combustible Total (L)", "% del Total"]] + [["Vacío (4)" if row["StageSequence"] == 4 else "Cargado (8)", f"{row['TotalFuel']:.2f}", f"{(row['TotalFuel']/total_fuel)*100:.1f}%"] for _, row in stage_consumption.iterrows()], colWidths=[2.5*inch, 2*inch, 2*inch])
        consumption_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#20c997")), ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke), ("ALIGN", (0,0), (-1,-1), "CENTER"), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 9), ("GRID", (0,0), (-1,-1), 1, colors.black)]))
        elements.extend([consumption_table, Spacer(1, 0.3*inch)])

        if self.model_metrics:
            elements.append(PageBreak())
            elements.append(Paragraph("7. Métricas de Rendimiento del Modelo", self.styles["SectionHeader"]))
            elements.append(Paragraph("Esta sección entrega las métricas de la calidad y precisión del modelo predictivo durante su entrenamiento.", self.styles["ChartDescription"]))
            elements.append(Spacer(1, 0.15*inch))

            s4, s8 = self.model_metrics.get('stage4', {}).get('metrics', {}), self.model_metrics.get('stage8', {}).get('metrics', {})
            if s4 and s8:
                metrics_table = Table([["Métrica", "Stage 4 (Vacío)", "Stage 8 (Cargado)"]] + [[label, fmt.format(s4.get(key, 0)), fmt.format(s8.get(key, 0))] for label, key, fmt in [("R² Score", "R2", "{:.4f}"), ("MAE (L)", "MAE", "{:.2f}"), ("RMSE (L)", "RMSE", "{:.2f}"), ("MAPE (%)", "MAPE_Safe", "{:.2f}"), ("Median AE (L)", "MedianAE", "{:.2f}"), ("RMSLE", "RMSLE", "{:.4f}"), ("Explained Var", "ExplainedVar", "{:.4f}")]], colWidths=[2.5*inch, 2*inch, 2*inch])
                metrics_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#5c7cfa")), ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke), ("ALIGN", (0,0), (-1,-1), "CENTER"), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 9), ("GRID", (0,0), (-1,-1), 1, colors.black), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.lightgrey])]))
                elements.extend([metrics_table, Spacer(1, 0.3*inch)])

                for chart_func, desc in [
                    (self._create_model_metrics_comparison_chart, "El Coeficiente de Determinación (R²) mide qué tan bien las predicciones del modelo se ajustan a los datos reales. Un valor cercano a 1 indica que el modelo explica casi toda la variabilidad de los datos, mientras que un valor más bajo sugiere menor capacidad predictiva."),
                    (self._create_error_metrics_chart, "MAE (Error Absoluto Medio) y RMSE (Raíz del Error Cuadrático Medio) miden la magnitud promedio de los errores de predicción en litros. RMSE penaliza más los errores grandes, mientras que MAE proporciona una medida más directa del error promedio."),
                    (self._create_mape_chart, "MAPE (Error Porcentual Absoluto Medio) expresa el error de predicción como porcentaje del valor real. Es útil para entender la precisión del modelo en términos relativos, independientemente de la escala de los valores de consumo.")
                ]:
                    chart = chart_func()
                    if chart:
                        elements.append(Paragraph(f"<b>Explicación:</b> {desc}", self.styles["ChartDescription"]))
                        elements.extend([Spacer(1, 0.1*inch), chart, Spacer(1, 0.2*inch)])

                elements.append(PageBreak())
                for stage, desc in [('stage4', "camión vacío"), ('stage8', "camión lleno")]:
                    chart = self._create_feature_importance_chart(stage=stage, top_n=8)
                    if chart:
                        elements.append(Paragraph(f"<b>Explicación:</b> Esta gráfica muestra las variables más influyentes en las predicciones del modelo para el {desc}. La importancia se calcula basándose en la teoria de juegos, a traves de iteraciones y entrega de valores aleatorios, determina que tanto contribuye una unidad de cada variable a la predicción final.", self.styles["ChartDescription"]))
                        elements.extend([Spacer(1, 0.1*inch), chart, Spacer(1, 0.3*inch)])

        doc.build(elements)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        if output_path:
            with open(output_path, "wb") as f: f.write(pdf_bytes)
        return pdf_bytes
