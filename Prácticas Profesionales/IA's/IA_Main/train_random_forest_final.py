import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# =========================
# CONFIGURACIÓN
# =========================
BASE_DIR = Path("/home/to-o/practicasProfeeionales/Prácticas Profesionales")
DATA_FILE = BASE_DIR / "JSON" / "Resultado_Homogenizado.xlsx"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "random_forest_60d.joblib"

DATETIME_COLUMN = "fecha_hora"
SOURCE_TARGET_COLUMN = "valor (kWh)"   # Nombre histórico del Excel
TARGET_COLUMN = "valor (W)"            # Interpretación correcta

MODEL_FEATURES = [
    "eToday (kWh)",
    "eTotal (kWh)",
    "air_temp",
    "relative_humidity",
    "power (kW)",
    "wind_speed_10m",
    "wind_direction_10m",
    "ghi",
    "dni",
    "gti",
    "hora",
    "dia_semana",
    "es_fin_semana",
    "mes",
    "estacion",
]

RF_PARAMS = {
    "n_estimators": 150,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}

N_SPLITS = 5


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hora"] = out[DATETIME_COLUMN].dt.hour
    out["dia_semana"] = out[DATETIME_COLUMN].dt.dayofweek
    out["es_fin_semana"] = out["dia_semana"].isin([5, 6]).astype(int)
    out["mes"] = out[DATETIME_COLUMN].dt.month
    out["estacion"] = (out["mes"] % 12 + 3) // 3
    return out


def calcular_metricas(y_true, y_pred) -> dict:
    """RMSE, MAE, R² y MAPE excluyendo y_true == 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    mask = y_true != 0
    mape = (
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        if np.any(mask)
        else np.nan
    )

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def main():
    print("Cargando datos...")
    df = pd.read_excel(DATA_FILE)

    if DATETIME_COLUMN not in df.columns:
        raise ValueError(f"No existe la columna {DATETIME_COLUMN}")

    # Homologar únicamente el nombre. No se aplica conversión numérica porque
    # la columna histórica ya contiene la magnitud interpretada como potencia W.
    if SOURCE_TARGET_COLUMN in df.columns and TARGET_COLUMN not in df.columns:
        df = df.rename(columns={SOURCE_TARGET_COLUMN: TARGET_COLUMN})

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"No existe la columna objetivo {TARGET_COLUMN}")

    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], errors="coerce")
    df = df.dropna(subset=[DATETIME_COLUMN]).sort_values(DATETIME_COLUMN).reset_index(drop=True)
    df = add_time_features(df)

    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan variables requeridas por el modelo: {missing}")

    train_df = df[[DATETIME_COLUMN, TARGET_COLUMN] + MODEL_FEATURES].dropna().copy()
    X = train_df[MODEL_FEATURES]
    y = train_df[TARGET_COLUMN]

    print(f"Registros: {len(train_df):,}")
    print(f"Periodo: {train_df[DATETIME_COLUMN].min()} -> {train_df[DATETIME_COLUMN].max()}")
    print(f"Variable objetivo: {TARGET_COLUMN}")
    print(f"Variables predictoras ({len(MODEL_FEATURES)}): {MODEL_FEATURES}")

    # Validación temporal para registrar el comportamiento del artefacto
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        met = calcular_metricas(y.iloc[test_idx], pred)
        fold_metrics.append(met)
        print(
            f"Fold {fold}: RMSE={met['RMSE']:.4f} W | "
            f"MAE={met['MAE']:.4f} W | R²={met['R2']:.4f} | MAPE={met['MAPE']:.2f}%"
        )

    avg_metrics = {
        key: float(np.mean([m[key] for m in fold_metrics]))
        for key in fold_metrics[0]
    }
    std_metrics = {
        key: float(np.std([m[key] for m in fold_metrics]))
        for key in fold_metrics[0]
    }

    # Modelo final entrenado con todo el historial disponible
    final_model = RandomForestRegressor(**RF_PARAMS)
    final_model.fit(X, y)

    feature_importance = (
        pd.DataFrame({
            "feature": MODEL_FEATURES,
            "importance": final_model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    artifact = {
        "model": final_model,
        "features": MODEL_FEATURES,
        "target": TARGET_COLUMN,
        "source_target": SOURCE_TARGET_COLUMN,
        "datetime_column": DATETIME_COLUMN,
        "metrics_mean": avg_metrics,
        "metrics_std": std_metrics,
        "metrics_definition": {
            "RMSE": "W",
            "MAE": "W",
            "R2": "sin unidad",
            "MAPE": "porcentaje; excluye observaciones con y_true == 0",
        },
        "validation": "TimeSeriesSplit(n_splits=5)",
        "model_params": RF_PARAMS,
        "feature_importance": feature_importance,
    }

    joblib.dump(artifact, MODEL_PATH)

    print("\nPromedio TimeSeriesSplit:")
    print(f"RMSE: {avg_metrics['RMSE']:.4f} ± {std_metrics['RMSE']:.4f} W")
    print(f"MAE : {avg_metrics['MAE']:.4f} ± {std_metrics['MAE']:.4f} W")
    print(f"R²  : {avg_metrics['R2']:.4f} ± {std_metrics['R2']:.4f}")
    print(f"MAPE: {avg_metrics['MAPE']:.2f} ± {std_metrics['MAPE']:.2f}%")
    print(f"\nModelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
