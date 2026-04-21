import pandas as pd

from app.config import DATETIME_COLUMN


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["hora"] = out[DATETIME_COLUMN].dt.hour
    out["dia_semana"] = out[DATETIME_COLUMN].dt.dayofweek
    out["es_fin_semana"] = out["dia_semana"].isin([5, 6]).astype(int)
    out["mes"] = out[DATETIME_COLUMN].dt.month
    out["estacion"] = (out["mes"] % 12 + 3) // 3

    return out