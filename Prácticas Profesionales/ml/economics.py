import pandas as pd


def estimate_cost(pred_df: pd.DataFrame, tariff_per_kwh: float) -> dict:
    """
    Convierte la predicción energética a una estimación económica simple.
    """
    if pred_df.empty or "prediccion" not in pred_df.columns:
        return {
            "energia_total": 0.0,
            "energia_promedio": 0.0,
            "costo_estimado": 0.0,
        }

    energia_total = float(pred_df["prediccion"].sum())
    energia_promedio = float(pred_df["prediccion"].mean())
    costo_estimado = energia_total * tariff_per_kwh

    return {
        "energia_total": energia_total,
        "energia_promedio": energia_promedio,
        "costo_estimado": costo_estimado,
    }