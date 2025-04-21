import pandas as pd
import plotly.express as px
from datetime import time

# 1. Cargar datos del sensor
etl_processor = ETLDataProcessor("T-210")
processed_data = etl_processor.run_etl()
sensor_df = processed_data['sensor'].copy()
sensor_df['TimeStamp'] = pd.to_datetime(sensor_df['TimeStamp'])

# 2. Establecer fecha objetivo: 2 de febrero de 2025
fecha_objetivo = pd.to_datetime("2025-01-31")
start_dt = fecha_objetivo.normalize()
end_dt = start_dt + pd.Timedelta(days=1)

# 3. Filtrar datos del sensor para ese día
sensor_day = sensor_df[
    (sensor_df['TimeStamp'] >= start_dt) &
    (sensor_df['TimeStamp'] < end_dt)
].copy()

# 4. Cargar y preparar fuel_data.csv
fuel_df = pd.read_csv("fuel_data.csv")
fuel_df['ShiftDate'] = pd.to_datetime(fuel_df['ShiftDate'])
fuel_df['TimeStamp'] = pd.to_datetime(fuel_df['TimeStamp'], errors='coerce').dt.time
fuel_df = fuel_df.dropna(subset=['TimeStamp'])

# Combinar fecha + hora
fuel_df['FullDateTime'] = fuel_df.apply(
    lambda row: pd.Timestamp.combine(row['ShiftDate'], row['TimeStamp']),
    axis=1
)

# Filtrar registros válidos para ese día y ese equipo
fuel_filtered = fuel_df[
    (fuel_df['Equipment'] == "T-210") &
    (fuel_df['FullDateTime'] >= start_dt) &
    (fuel_df['FullDateTime'] < end_dt)
]

# 5. Crear el gráfico si hay datos
if not sensor_day.empty:
    fig = px.line(
        sensor_day,
        x='TimeStamp',
        y='FuelLevelLiters',
        title=f'Nivel de Combustible - T-210 {fecha_objetivo}',
        labels={'FuelLevelLiters': 'Litros', 'TimeStamp': 'Hora'},
        markers=True
    )

    fig.update_traces(
        hovertemplate="<b>Hora</b>: %{x|%H:%M}<br><b>Combustible</b>: %{y:.2f} L"
    )

    # Agregar líneas horizontales desde CSV si hay valores
    if not fuel_filtered.empty:
        for i, row in fuel_filtered.iterrows():
            val = row['FuelLevelLiters']
            hora = row['FullDateTime'].strftime('%H:%M:%S')
            fig.add_hline(
                y=val,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Validación CSV: {val:.2f}L {hora}",
                annotation_position="top right" if i == 0 else "bottom right"
            )

    fig.update_layout(
        xaxis_title="Hora del Día",
        yaxis_title="Litros de Combustible",
        hovermode="x unified",
        template="plotly_white"
    )

    fig.show()
else:
    print("⚠️ No hay datos del sensor para T-210 el 2 de febrero de 2025.")
