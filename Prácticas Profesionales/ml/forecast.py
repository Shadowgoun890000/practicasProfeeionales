import pandas as pd

from app.config import DATETIME_COLUMN, TARGET_COLUMN
from ml.schema import MODEL_FEATURES


def make_future_dataframe(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Construye un dataframe futuro con frecuencia de 5 minutos.
    Usa como base el patrón reciente de variables explicativas.
    """
    df = df.copy().sort_values(DATETIME_COLUMN)

    pasos_por_dia = 288  # 24 * 60 / 5
    total_pasos = days * pasos_por_dia

    ultimo_timestamp = df[DATETIME_COLUMN].max()

    fechas_futuras = pd.date_range(
        start=ultimo_timestamp + pd.Timedelta(minutes=5),
        periods=total_pasos,
        freq="5min",
    )

    # Repetimos el patrón más reciente de las variables disponibles
    base_cols = [c for c in MODEL_FEATURES if c in df.columns]
    patron = df[base_cols].tail(total_pasos).copy()

    if patron.empty:
        return pd.DataFrame()

    if len(patron) < total_pasos:
        repeticiones = (total_pasos // len(patron)) + 1
        patron = pd.concat([patron] * repeticiones, ignore_index=True).head(total_pasos)
    else:
        patron = patron.reset_index(drop=True)

    future_df = patron.copy()
    future_df[DATETIME_COLUMN] = fechas_futuras

    # Recalcular variables temporales con base en la fecha futura
    future_df["hora"] = future_df[DATETIME_COLUMN].dt.hour
    future_df["dia_semana"] = future_df[DATETIME_COLUMN].dt.dayofweek
    future_df["es_fin_semana"] = future_df["dia_semana"].isin([5, 6]).astype(int)
    future_df["mes"] = future_df[DATETIME_COLUMN].dt.month
    future_df["estacion"] = (future_df["mes"] % 12 + 3) // 3

    return future_df


def build_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in MODEL_FEATURES if c in df.columns]
    return df[available].copy()


def forecast_with_model(model, df: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Genera pronóstico futuro usando el modelo real.
    """
    if model is None:
        return pd.DataFrame()

    future_df = make_future_dataframe(df, days)
    if future_df.empty:
        return pd.DataFrame()

    x_future = build_model_matrix(future_df)
    if x_future.empty:
        return pd.DataFrame()

    y_pred = model.predict(x_future)

    result = pd.DataFrame({
        DATETIME_COLUMN: future_df[DATETIME_COLUMN],
        "prediccion": y_pred
    })

    return result