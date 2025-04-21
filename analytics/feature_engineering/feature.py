# =============================================
# Block 5 - Advanced Feature Engineering & Aggregation
# =============================================
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer

def prepare_visualization_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma los datos crudos en un dataset listo para visualización mediante:
    - Agregación temporal inteligente
    - Ingeniería de características contextuales
    - Segmentación adaptativa
    - Control de calidad integrado
    """
    # --------------------------------------------------
    # 5.1 Pre-procesamiento avanzado
    # --------------------------------------------------
    # Normalización de zona horaria y ordenamiento
    df = (
        df.sort_values('created_at_local')
        .assign(timestamp=lambda x: x['created_at_local'].dt.tz_convert('America/Santiago'))
        .drop(columns='created_at_local')
    )
    
    # --------------------------------------------------
    # 5.2 Agregación temporal adaptativa
    # --------------------------------------------------
    def calculate_optimal_freq(ts_series):
        """Calcula frecuencia óptima usando densidad de puntos y variabilidad"""
        freq = pd.infer_freq(ts_series.sort_values().head(1000))
        if freq: 
            return freq
        intervals = ts_series.diff().dropna().dt.total_seconds()
        mode = stats.mode(intervals).mode[0]
        return f'{int(mode)}S' if mode < 60 else f'{int(mode/60)}T'

    # Agregación por camión y flota
    agg_config = {
        'value': ['mean', 'max', lambda x: x.quantile(0.95)],
        'speed': ['mean', 'std', 'max'],
        'metric': 'first',
        'fleet': 'first'
    }
    
    df_agg = (
        df.groupby(['truck', 'metric', pd.Grouper(key='timestamp', freq=calculate_optimal_freq(df['timestamp']))])
        .agg(agg_config)
        .rename(columns={
            '<lambda_0>': 'value_p95',
            'mean': 'speed_mean',
            'std': 'speed_std',
            'max': 'speed_max'
        })
        .droplevel(0, axis=1)  # Eliminar nivel multiindex
        .reset_index()
    )
    
    # --------------------------------------------------
    # 5.3 Ingeniería de características avanzadas
    # --------------------------------------------------
    # 5.3.1 Segmentación de velocidad adaptativa
    discretizer = KBinsDiscretizer(
        n_bins=5, 
        encode='ordinal', 
        strategy='quantile',
        subsample=2_000_000  # Manejo de grandes datasets
    )
    df_agg['speed_category'] = discretizer.fit_transform(
        df_agg[['speed_mean']]
    ).astype(int)
    
    speed_labels = [
        'Muy Baja', 
        'Baja', 
        'Media', 
        'Alta', 
        'Muy Alta'
    ]
    df_agg['speed_category'] = (
        df_agg['speed_category']
        .map(dict(enumerate(speed_labels)))
    )
    
    # 5.3.2 Características de tendencia
    df_agg['value_rolling'] = (
        df_agg.groupby(['truck', 'metric'])['value']
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )
    
    # 5.3.3 Detección de eventos anómalos
    df_agg['speed_anomaly'] = (
        (df_agg['speed_mean'] > (df_agg['speed_mean'].mean() + 3 * df_agg['speed_mean'].std())) |
        (df_agg['speed_mean'] < (df_agg['speed_mean'].mean() - 3 * df_agg['speed_mean'].std()))
    ).astype(int)
    
    # --------------------------------------------------
    # 5.4 Validación y control de calidad
    # --------------------------------------------------
    # Validación de integridad
    assert df_agg['timestamp'].is_monotonic_increasing, "Error en orden temporal"
    
    # Reporte de calidad
    print("\n✅ Transformación completada:")
    print(f"- Registros originales: {len(df):,}")
    print(f"- Registros transformados: {len(df_agg):,}")
    print(f"- Reducción de datos: {len(df)/len(df_agg):.1f}x")
    
    print("\n📊 Distribución temporal:")
    print(pd.crosstab(
        df_agg['timestamp'].dt.hour,
        df_agg['metric'],
        values=df_agg['value'],
        aggfunc='mean'
    ).to_string())
    
    print("\n🔍 Balance de categorías de velocidad:")
    print(df_agg['speed_category'].value_counts(normalize=True).to_string())
    
    return df_agg