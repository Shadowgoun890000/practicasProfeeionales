import numpy as np
import pandas as pd

from app.config import DATETIME_COLUMN
from ml.schema import MODEL_FEATURES


PASOS_POR_DIA = 288  # 24 h * 60 / 5


def _perfil_diario(df: pd.DataFrame, columna: str, dias_base: int = 15):
    """
    Obtiene un perfil típico de 5 minutos usando los últimos días disponibles.
    Se utiliza la mediana de cada posición del día para reducir el efecto
    de valores atípicos.
    """

    if columna not in df.columns:
        return None

    base = df.copy().sort_values(DATETIME_COLUMN)

    fecha_max = base[DATETIME_COLUMN].max()
    fecha_min = fecha_max - pd.Timedelta(days=dias_base)

    base = base[base[DATETIME_COLUMN] >= fecha_min].copy()

    base["_minuto_dia"] = (
        base[DATETIME_COLUMN].dt.hour * 60
        + base[DATETIME_COLUMN].dt.minute
    )

    perfil = (
        base.groupby("_minuto_dia")[columna]
        .median()
    )

    return perfil


def _aplicar_perfil(fechas, perfil):
    """
    Asigna a cada timestamp futuro el valor correspondiente
    a su minuto del día.
    """

    if perfil is None:
        return np.full(len(fechas), np.nan)

    minutos = fechas.hour * 60 + fechas.minute

    return np.array([
        perfil.get(minuto, np.nan)
        for minuto in minutos
    ])


def make_future_dataframe(
    df: pd.DataFrame,
    days: int
) -> pd.DataFrame:

    df = df.copy().sort_values(DATETIME_COLUMN)

    total_pasos = days * PASOS_POR_DIA
    ultimo_timestamp = df[DATETIME_COLUMN].max()

    fechas_futuras = pd.date_range(
        start=ultimo_timestamp + pd.Timedelta(minutes=5),
        periods=total_pasos,
        freq="5min",
    )

    future_df = pd.DataFrame({
        DATETIME_COLUMN: fechas_futuras
    })

    # ===================================
    # VARIABLES TEMPORALES
    # ===================================

    future_df["hora"] = future_df[DATETIME_COLUMN].dt.hour
    future_df["dia_semana"] = future_df[DATETIME_COLUMN].dt.dayofweek

    future_df["es_fin_semana"] = (
        future_df["dia_semana"]
        .isin([5, 6])
        .astype(int)
    )

    future_df["mes"] = future_df[DATETIME_COLUMN].dt.month

    future_df["estacion"] = (
        future_df["mes"] % 12 + 3
    ) // 3

    # ===================================
    # POWER
    # Perfil operativo histórico
    # ===================================

    if "power (kW)" in df.columns:

        perfil_power = _perfil_diario(
            df,
            "power (kW)"
        )

        future_df["power (kW)"] = _aplicar_perfil(
            fechas_futuras,
            perfil_power
        )

    # ===================================
    # ETODAY
    # Perfil diario histórico
    # ===================================

    if "eToday (kWh)" in df.columns:

        perfil_etoday = _perfil_diario(
            df,
            "eToday (kWh)"
        )

        future_df["eToday (kWh)"] = _aplicar_perfil(
            fechas_futuras,
            perfil_etoday
        )

    # ===================================
    # ETOTAL
    # Mantener comportamiento acumulativo
    # ===================================

    if "eTotal (kWh)" in df.columns:

        ultimo_etotal = pd.to_numeric(
            df["eTotal (kWh)"],
            errors="coerce"
        ).dropna().iloc[-1]

        # Energía típica diaria según eToday
        if "eToday (kWh)" in future_df.columns:

            temp = future_df[
                [DATETIME_COLUMN, "eToday (kWh)"]
            ].copy()

            temp["_fecha"] = temp[DATETIME_COLUMN].dt.date

            energia_diaria = (
                temp.groupby("_fecha")["eToday (kWh)"]
                .transform("max")
            )

            # Incremento acumulado aproximado
            dias_transcurridos = (
                future_df[DATETIME_COLUMN].dt.normalize()
                - future_df[DATETIME_COLUMN].iloc[0].normalize()
            ).dt.days

            energia_tipica = (
                pd.Series(energia_diaria)
                .median()
            )

            future_df["eTotal (kWh)"] = (
                ultimo_etotal
                + dias_transcurridos * energia_tipica
                + future_df["eToday (kWh)"]
            )

        else:
            future_df["eTotal (kWh)"] = ultimo_etotal

    # ===================================
    # RESTO DE VARIABLES
    # ===================================

    columnas_especiales = {
        "hora",
        "dia_semana",
        "es_fin_semana",
        "mes",
        "estacion",
        "power (kW)",
        "eToday (kWh)",
        "eTotal (kWh)",
    }

    for columna in MODEL_FEATURES:

        if columna in columnas_especiales:
            continue

        if columna not in df.columns:
            continue

        perfil = _perfil_diario(
            df,
            columna
        )

        future_df[columna] = _aplicar_perfil(
            fechas_futuras,
            perfil
        )

    return future_df


def build_model_matrix(df: pd.DataFrame) -> pd.DataFrame:

    available = [
        c for c in MODEL_FEATURES
        if c in df.columns
    ]

    return df[available].copy()


def forecast_with_model(
    model,
    df: pd.DataFrame,
    days: int
) -> pd.DataFrame:

    if model is None:
        return pd.DataFrame()

    future_df = make_future_dataframe(
        df,
        days
    )

    if future_df.empty:
        return pd.DataFrame()

    x_future = build_model_matrix(
        future_df
    )

    if x_future.empty:
        return pd.DataFrame()

    # Evitar que una pequeña ausencia dentro del
    # perfil histórico impida generar la predicción
    x_future = (
        x_future
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
    )

    y_pred = model.predict(
        x_future
    )

    result = pd.DataFrame({
        DATETIME_COLUMN:
            future_df[DATETIME_COLUMN],

        "prediccion":
            y_pred
    })

    return result