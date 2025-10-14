import streamlit as st
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    PageBreak,
    Image,
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import io
from typing import Optional
import tempfile
import os


class ReportGenerator:
    """Generate comprehensive PDF reports for fuel consumption analysis."""

    """
        PDF Report Generator for Fuel Consumption Analysis

        This module generates comprehensive PDF reports with:
        - Daily/weekly consumption trends
        - Real vs predicted consumption by unit
        - Efficiency metrics (L/ton-km)
        - Anomaly detection (consumption exceeding predictions)
        - Stage-by-stage consumption comparison
        - Cost analysis and KPIs
    """

    def __init__(
        self,
        df: pl.DataFrame,
        truck_id: str,
        diesel_price_per_liter: float = 3.96,
    ):
        """
        Initialize report generator.

        Args:
            df: DataFrame with predictions and actual data
            truck_id: Truck identifier
            diesel_price_per_liter: Current diesel price in BOB/L
        """
        self.df = df
        self.truck_id = truck_id
        self.diesel_price = diesel_price_per_liter
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()

    def _create_custom_styles(self):
        """Create custom paragraph styles."""
        self.styles.add(
            ParagraphStyle(
                name="CustomTitle",
                parent=self.styles["Heading1"],
                fontSize=24,
                textColor=colors.HexColor("#1f4788"),
                spaceAfter=30,
                alignment=TA_CENTER,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SectionHeader",
                parent=self.styles["Heading2"],
                fontSize=16,
                textColor=colors.HexColor("#2c5aa0"),
                spaceAfter=12,
                spaceBefore=12,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="MetricLabel",
                parent=self.styles["Normal"],
                fontSize=10,
                textColor=colors.grey,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="MetricValue",
                parent=self.styles["Normal"],
                fontSize=14,
                textColor=colors.HexColor("#1f4788"),
                fontName="Helvetica-Bold",
            )
        )

    def _calculate_daily_trends(self) -> pl.DataFrame:
        """Calculate daily consumption trends."""
        daily = (
            self.df.group_by(pl.col("TimeStamp").dt.date().alias("Date"))
            .agg(
                [
                    pl.col("PredictedFuel").sum().alias("TotalPredictedFuel"),
                    pl.col("PredictedFuel").mean().alias("AvgPredictedFuel"),
                    pl.len().alias("Cycles"),
                ]
            )
            .sort("Date")
        )
        return daily

    def _calculate_weekly_trends(self) -> pl.DataFrame:
        """Calculate weekly consumption trends."""
        weekly = (
            self.df.with_columns(
                pl.col("TimeStamp").dt.week().alias("Week"),
                pl.col("TimeStamp").dt.year().alias("Year"),
            )
            .group_by(["Year", "Week"])
            .agg(
                [
                    pl.col("PredictedFuel").sum().alias("TotalPredictedFuel"),
                    pl.col("PredictedFuel").mean().alias("AvgPredictedFuel"),
                    pl.len().alias("Cycles"),
                ]
            )
            .sort(["Year", "Week"])
        )
        return weekly

    def _calculate_efficiency_metrics(self) -> pl.DataFrame:
        """Calculate L/ton-km efficiency by stage."""
        # Filter valid data
        df_valid = self.df.filter(
            (pl.col("Distance") > 0)
            & (pl.col("TotalMeasuredTonnage").is_not_null())
            & (pl.col("TotalMeasuredTonnage") > 0)
        )

        efficiency = (
            df_valid.group_by("StageSequence")
            .agg(
                [
                    pl.col("PredictedFuel").sum().alias("TotalFuel"),
                    pl.col("Distance").sum().alias("TotalDistance"),
                    pl.col("TotalMeasuredTonnage").sum().alias("TotalTonnage"),
                    pl.len().alias("Cycles"),
                ]
            )
            .with_columns(
                [
                    (
                        pl.col("TotalFuel")
                        / (pl.col("TotalTonnage") * pl.col("TotalDistance") / 1000)
                    ).alias("L_per_ton_km"),
                    (pl.col("TotalDistance") / pl.col("Cycles")).alias(
                        "AvgDistancePerCycle"
                    ),
                ]
            )
        )
        return efficiency

    def _detect_anomalies(self, threshold_percentile: float = 90) -> pl.DataFrame:
        """
        Detect cycles with abnormally high consumption.

        Args:
            threshold_percentile: Percentile above which to flag anomalies

        Returns:
            DataFrame with anomalous cycles
        """
        threshold = self.df["PredictedFuel"].quantile(threshold_percentile / 100)

        anomalies = (
            self.df.filter(pl.col("PredictedFuel") > threshold)
            .sort("PredictedFuel", descending=True)
            .head(20)  # Top 20 anomalies
        )

        return anomalies

    def _calculate_stage_comparison(self) -> pl.DataFrame:
        """Compare consumption across operational stages."""
        stage_stats = (
            self.df.group_by("StageSequence")
            .agg(
                [
                    pl.col("PredictedFuel").mean().alias("AvgFuel"),
                    pl.col("PredictedFuel").std().alias("StdFuel"),
                    pl.col("PredictedFuel").min().alias("MinFuel"),
                    pl.col("PredictedFuel").max().alias("MaxFuel"),
                    pl.col("Distance").mean().alias("AvgDistance"),
                    pl.col("SpeedAvg").mean().alias("AvgSpeed"),
                    pl.len().alias("Cycles"),
                ]
            )
            .sort("StageSequence")
        )
        return stage_stats

    def _calculate_cost_analysis(self) -> dict:
        """Calculate cost metrics."""
        total_fuel = self.df["PredictedFuel"].sum()
        total_cost = total_fuel * self.diesel_price

        # Cost by stage
        cost_by_stage = (
            self.df.group_by("StageSequence")
            .agg(pl.col("PredictedFuel").sum().alias("TotalFuel"))
            .with_columns((pl.col("TotalFuel") * self.diesel_price).alias("Cost"))
        )

        # Cost per ton (only for Stage 8 - loaded)
        df_stage8 = self.df.filter(
            (pl.col("StageSequence") == 8)
            & (pl.col("TotalMeasuredTonnage").is_not_null())
            & (pl.col("TotalMeasuredTonnage") > 0)
        )

        cost_per_ton = None
        if len(df_stage8) > 0:
            total_tonnage = df_stage8["TotalMeasuredTonnage"].sum()
            fuel_stage8 = df_stage8["PredictedFuel"].sum()
            cost_per_ton = (fuel_stage8 * self.diesel_price) / total_tonnage

        return {
            "total_fuel": total_fuel,
            "total_cost": total_cost,
            "cost_by_stage": cost_by_stage,
            "cost_per_ton": cost_per_ton,
        }

    def _calculate_kpis(self) -> dict:
        """Calculate critical KPIs."""
        total_consumption = self.df["PredictedFuel"].sum()
        total_cycles = len(self.df)
        avg_consumption_per_cycle = self.df["PredictedFuel"].mean()

        # Consumption by stage
        stage_consumption = (
            self.df.group_by("StageSequence")
            .agg(pl.col("PredictedFuel").sum().alias("TotalFuel"))
            .sort("TotalFuel", descending=True)
        )

        # Most/least efficient cycles
        best_cycle = self.df.sort("PredictedFuel").head(1)
        worst_cycle = self.df.sort("PredictedFuel", descending=True).head(1)

        return {
            "total_consumption": total_consumption,
            "total_cycles": total_cycles,
            "avg_consumption_per_cycle": avg_consumption_per_cycle,
            "stage_consumption": stage_consumption,
            "best_cycle": best_cycle,
            "worst_cycle": worst_cycle,
        }

    def _create_plot_image(self, fig, filename: str) -> str:
        """Save plotly figure as image and return path."""
        img_bytes = fig.to_image(format="png", width=800, height=400)
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        with open(temp_path, "wb") as f:
            f.write(img_bytes)
        return temp_path

    def _plot_daily_trends(self) -> str:
        """Create daily trends plot."""
        daily = self._calculate_daily_trends().to_pandas()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=daily["Date"],
                y=daily["TotalPredictedFuel"],
                mode="lines+markers",
                name="Daily Consumption",
                line=dict(color="#2c5aa0", width=2),
            )
        )

        fig.update_layout(
            title="Daily Fuel Consumption Trend",
            xaxis_title="Date",
            yaxis_title="Total Fuel (L)",
            template="plotly_white",
            height=400,
        )

        return self._create_plot_image(fig, "daily_trend.png")

    def _plot_stage_comparison(self) -> str:
        """Create stage comparison plot."""
        stage_stats = self._calculate_stage_comparison().to_pandas()

        fig = go.Figure(
            data=[
                go.Bar(
                    x=stage_stats["StageSequence"].map(
                        {4: "Empty Truck", 8: "Loaded Truck"}
                    ),
                    y=stage_stats["AvgFuel"],
                    error_y=dict(type="data", array=stage_stats["StdFuel"]),
                    marker_color=["#4dabf7", "#ff922b"],
                )
            ]
        )

        fig.update_layout(
            title="Average Fuel Consumption by Stage",
            xaxis_title="Stage",
            yaxis_title="Average Fuel (L)",
            template="plotly_white",
            height=400,
        )

        return self._create_plot_image(fig, "stage_comparison.png")

    def generate_pdf_report(self, output_path: Optional[str] = None) -> bytes:
        """
        Generate comprehensive PDF report.

        Args:
            output_path: Optional file path to save PDF

        Returns:
            PDF file as bytes
        """
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )

        # Container for PDF elements
        elements = []

        # Title
        title = Paragraph(
            f"Fuel Consumption Analysis Report<br/>{self.truck_id}",
            self.styles["CustomTitle"],
        )
        elements.append(title)
        elements.append(Spacer(1, 0.2 * inch))

        # Report metadata
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = Paragraph(
            f"<b>Report Generated:</b> {report_date}<br/>"
            f"<b>Analysis Period:</b> {self.df['TimeStamp'].min().strftime('%Y-%m-%d')} to "
            f"{self.df['TimeStamp'].max().strftime('%Y-%m-%d')}<br/>"
            f"<b>Total Records:</b> {len(self.df):,}",
            self.styles["Normal"],
        )
        elements.append(metadata)
        elements.append(Spacer(1, 0.3 * inch))

        # --- SECTION 1: Executive Summary ---
        elements.append(Paragraph("1. Executive Summary", self.styles["SectionHeader"]))

        kpis = self._calculate_kpis()
        cost_analysis = self._calculate_cost_analysis()

        kpi_data = [
            ["Metric", "Value"],
            ["Total Fuel Consumption", f"{kpis['total_consumption']:.2f} L"],
            ["Total Cost", f"{cost_analysis['total_cost']:.2f} BOB"],
            ["Total Cycles", f"{kpis['total_cycles']:,}"],
            [
                "Average Consumption/Cycle",
                f"{kpis['avg_consumption_per_cycle']:.2f} L",
            ],
        ]

        if cost_analysis["cost_per_ton"] is not None:
            kpi_data.append(
                ["Cost per Ton", f"{cost_analysis['cost_per_ton']:.2f} BOB/ton"]
            )

        kpi_table = Table(kpi_data, colWidths=[3 * inch, 2 * inch])
        kpi_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5aa0")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        elements.append(kpi_table)
        elements.append(Spacer(1, 0.3 * inch))

        # --- SECTION 2: Consumption Trends ---
        elements.append(
            Paragraph("2. Consumption Trends", self.styles["SectionHeader"])
        )

        # Daily trends plot
        daily_plot_path = self._plot_daily_trends()
        elements.append(Image(daily_plot_path, width=5 * inch, height=2.5 * inch))
        elements.append(Spacer(1, 0.2 * inch))

        # Daily statistics table
        daily_trends = self._calculate_daily_trends().to_pandas()
        daily_data = [["Date", "Total Fuel (L)", "Avg Fuel/Cycle (L)", "Cycles"]]
        for _, row in daily_trends.head(10).iterrows():
            daily_data.append(
                [
                    str(row["Date"]),
                    f"{row['TotalPredictedFuel']:.2f}",
                    f"{row['AvgPredictedFuel']:.2f}",
                    f"{row['Cycles']}",
                ]
            )

        daily_table = Table(daily_data, colWidths=[1.5 * inch] * 4)
        daily_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4dabf7")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        elements.append(daily_table)
        elements.append(PageBreak())

        # --- SECTION 3: Efficiency Metrics ---
        elements.append(
            Paragraph("3. Efficiency Metrics (L/ton-km)", self.styles["SectionHeader"])
        )

        efficiency = self._calculate_efficiency_metrics().to_pandas()
        if not efficiency.empty:
            eff_data = [
                [
                    "Stage",
                    "Total Fuel (L)",
                    "Total Distance (km)",
                    "Total Tonnage",
                    "L/ton-km",
                    "Cycles",
                ]
            ]
            for _, row in efficiency.iterrows():
                stage_name = (
                    "Empty Truck" if row["StageSequence"] == 4 else "Loaded Truck"
                )
                eff_data.append(
                    [
                        stage_name,
                        f"{row['TotalFuel']:.2f}",
                        f"{row['TotalDistance']/1000:.2f}",
                        f"{row['TotalTonnage']:.0f}",
                        f"{row['L_per_ton_km']:.4f}",
                        f"{row['Cycles']}",
                    ]
                )

            eff_table = Table(eff_data, colWidths=[1.2 * inch] * 6)
            eff_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#74c0fc")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            elements.append(eff_table)
        else:
            elements.append(
                Paragraph(
                    "Insufficient data for efficiency calculation.",
                    self.styles["Normal"],
                )
            )

        elements.append(Spacer(1, 0.3 * inch))

        # --- SECTION 4: Stage Comparison ---
        elements.append(
            Paragraph("4. Stage-by-Stage Comparison", self.styles["SectionHeader"])
        )

        stage_plot_path = self._plot_stage_comparison()
        elements.append(Image(stage_plot_path, width=5 * inch, height=2.5 * inch))
        elements.append(Spacer(1, 0.2 * inch))

        stage_stats = self._calculate_stage_comparison().to_pandas()
        stage_data = [
            [
                "Stage",
                "Avg Fuel (L)",
                "Std Dev",
                "Min",
                "Max",
                "Avg Speed",
                "Cycles",
            ]
        ]
        for _, row in stage_stats.iterrows():
            stage_name = "Empty Truck" if row["StageSequence"] == 4 else "Loaded Truck"
            stage_data.append(
                [
                    stage_name,
                    f"{row['AvgFuel']:.2f}",
                    f"{row['StdFuel']:.2f}",
                    f"{row['MinFuel']:.2f}",
                    f"{row['MaxFuel']:.2f}",
                    f"{row['AvgSpeed']:.1f} km/h",
                    f"{row['Cycles']}",
                ]
            )

        stage_table = Table(stage_data, colWidths=[1.1 * inch] * 7)
        stage_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ff922b")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        elements.append(stage_table)
        elements.append(PageBreak())

        # --- SECTION 5: Anomaly Detection ---
        elements.append(
            Paragraph(
                "5. High Consumption Events (Top 90th Percentile)",
                self.styles["SectionHeader"],
            )
        )

        anomalies = self._detect_anomalies().to_pandas()
        if not anomalies.empty:
            anom_data = [
                ["Timestamp", "Stage", "Fuel (L)", "Distance (m)", "Destination"]
            ]
            for _, row in anomalies.head(15).iterrows():
                stage_name = "Empty" if row["StageSequence"] == 4 else "Loaded"
                anom_data.append(
                    [
                        row["TimeStamp"].strftime("%Y-%m-%d %H:%M"),
                        stage_name,
                        f"{row['PredictedFuel']:.2f}",
                        f"{row['Distance']:.0f}",
                        row["Destination"],
                    ]
                )

            anom_table = Table(anom_data, colWidths=[1.5 * inch] * 5)
            anom_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fa5252")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            elements.append(anom_table)
        else:
            elements.append(Paragraph("No anomalies detected.", self.styles["Normal"]))

        elements.append(Spacer(1, 0.3 * inch))

        # --- SECTION 6: Cost Analysis ---
        elements.append(Paragraph("6. Cost Analysis", self.styles["SectionHeader"]))

        cost_by_stage = cost_analysis["cost_by_stage"].to_pandas()
        cost_data = [["Stage", "Total Fuel (L)", "Total Cost (BOB)", "% of Total"]]
        total_cost = cost_analysis["total_cost"]
        for _, row in cost_by_stage.iterrows():
            stage_name = "Empty Truck" if row["StageSequence"] == 4 else "Loaded Truck"
            percentage = (row["Cost"] / total_cost) * 100
            cost_data.append(
                [
                    stage_name,
                    f"{row['TotalFuel']:.2f}",
                    f"{row['Cost']:.2f}",
                    f"{percentage:.1f}%",
                ]
            )

        cost_table = Table(
            cost_data, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch]
        )
        cost_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#20c997")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        elements.append(cost_table)

        # Build PDF
        doc.build(elements)

        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()

        # Save to file if path provided
        if output_path:
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)

        # Clean up temporary images
        for img_file in ["daily_trend.png", "stage_comparison.png"]:
            temp_path = os.path.join(tempfile.gettempdir(), img_file)
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return pdf_bytes


def add_report_generation_to_ui(df_predictions: pl.DataFrame, truck_id: str):
    """
    Add report generation section to Streamlit UI.

    Args:
        df_predictions: DataFrame with predictions
        truck_id: Truck identifier
    """
    st.header("📄 Generate Comprehensive Report")

    col1, col2 = st.columns(2)

    with col1:
        diesel_price = st.number_input(
            "Diesel Price (BOB/L)",
            min_value=0.0,
            value=3.96,
            step=0.01,
            help="Current diesel price for cost calculations",
        )

    with col2:
        st.metric("Records in Report", f"{len(df_predictions):,}")

    if st.button("🎯 Generate PDF Report", type="primary", use_container_width=True):
        with st.spinner("Generating comprehensive report..."):
            try:
                generator = ReportGenerator(
                    df=df_predictions,
                    truck_id=truck_id,
                    diesel_price_per_liter=diesel_price,
                )

                pdf_bytes = generator.generate_pdf_report()

                st.success("✅ Report generated successfully!")

                # Download button
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"Fuel_Report_{truck_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                st.exception(e)
