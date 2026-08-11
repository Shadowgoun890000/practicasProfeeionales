import pandas as pd


def estimate_cost(
    pred_df: pd.DataFrame,
    tariff_per_kwh: float
) -> dict:

    """
    Calcula la energía estimada a partir de una serie
    de potencia en W registrada cada cinco minutos.
    """

    if pred_df.empty or "prediccion" not in pred_df.columns:
        return {
            "energia_total_kwh": 0.0,
            "potencia_promedio_w": 0.0,
            "costo_estimado": 0.0,
        }

    potencia = pd.to_numeric(
        pred_df["prediccion"],
        errors="coerce"
    ).fillna(0)

    intervalo_horas = 5 / 60

    energia_total_kwh = (
        potencia.sum()
        * intervalo_horas
        / 1000
    )

    potencia_promedio_w = potencia.mean()

    costo_estimado = (
        energia_total_kwh
        * tariff_per_kwh
    )

    return {
        "energia_total_kwh": float(energia_total_kwh),
        "potencia_promedio_w": float(potencia_promedio_w),
        "costo_estimado": float(costo_estimado),
    }