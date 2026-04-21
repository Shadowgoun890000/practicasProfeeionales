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
TARGET_COLUMN = "valor (kWh)"

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

N_SPLITS = 5
RANDOM_STATE = 42


# =========================
# FUNCIONES AUXILIARES
# =========================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hora"] = df[DATETIME_COLUMN].dt.hour
    df["dia_semana"] = df[DATETIME_COLUMN].dt.dayofweek
    df["es_fin_semana"] = df["dia_semana"].isin([5, 6]).astype(int)
    df["mes"] = df[DATETIME_COLUMN].dt.month
    df["estacion"] = (df["mes"] % 12 + 3) // 3
    return df


def calcular_metricas(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE robusto a ceros
    denom = np.where(y_true == 0, 1, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape,
    }


def print_metricas(nombre: str, metricas: dict):
    print(
        f"{nombre} -> "
        f"RMSE: {metricas['RMSE']:.4f} | "
        f"MAE: {metricas['MAE']:.4f} | "
        f"R²: {metricas['R2']:.4f} | "
        f"MAPE: {metricas['MAPE']:.2f}%"
    )


# =========================
# CARGA Y PREPARACIÓN
# =========================
print("Cargando datos...")
df = pd.read_excel(DATA_FILE)

if DATETIME_COLUMN not in df.columns:
    raise ValueError(f"No existe la columna {DATETIME_COLUMN}")

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"No existe la columna objetivo {TARGET_COLUMN}")

df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN])
df = df.sort_values(DATETIME_COLUMN).reset_index(drop=True)

df = add_time_features(df)

available_features = [col for col in MODEL_FEATURES if col in df.columns]
missing_features = [col for col in MODEL_FEATURES if col not in df.columns]

print(f"Total de registros: {len(df)}")
print(f"Rango temporal: {df[DATETIME_COLUMN].min()} a {df[DATETIME_COLUMN].max()}")
print(f"Features disponibles ({len(available_features)}): {available_features}")

if missing_features:
    print(f"Features faltantes: {missing_features}")

if not available_features:
    raise ValueError("No hay features disponibles para entrenar el modelo.")

# Eliminar filas con NA solo en columnas necesarias
train_df = df[[DATETIME_COLUMN, TARGET_COLUMN] + available_features].dropna().copy()

X = train_df[available_features]
y = train_df[TARGET_COLUMN]

print(f"Registros finales para entrenamiento: {len(train_df)}")


# =========================
# VALIDACIÓN TimeSeriesSplit
# =========================
print("\n" + "=" * 70)
print("VALIDACIÓN CON TIMESERIESSPLIT")
print("=" * 70)

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

fold_metrics = []
best_model = None
best_score = -np.inf

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metricas = calcular_metricas(y_test, y_pred)
    fold_metrics.append(metricas)

    print_metricas(f"Fold {fold}", metricas)

    if metricas["R2"] > best_score:
        best_score = metricas["R2"]
        best_model = model

# Promedios
avg_metrics = {
    "RMSE": np.mean([m["RMSE"] for m in fold_metrics]),
    "MAE": np.mean([m["MAE"] for m in fold_metrics]),
    "R2": np.mean([m["R2"] for m in fold_metrics]),
    "MAPE": np.mean([m["MAPE"] for m in fold_metrics]),
}

std_metrics = {
    "RMSE": np.std([m["RMSE"] for m in fold_metrics]),
    "MAE": np.std([m["MAE"] for m in fold_metrics]),
    "R2": np.std([m["R2"] for m in fold_metrics]),
    "MAPE": np.std([m["MAPE"] for m in fold_metrics]),
}

print("\n" + "=" * 70)
print("RESULTADOS PROMEDIO")
print("=" * 70)
print(
    f"RMSE: {avg_metrics['RMSE']:.4f} ± {std_metrics['RMSE']:.4f}\n"
    f"MAE:  {avg_metrics['MAE']:.4f} ± {std_metrics['MAE']:.4f}\n"
    f"R²:   {avg_metrics['R2']:.4f} ± {std_metrics['R2']:.4f}\n"
    f"MAPE: {avg_metrics['MAPE']:.2f}% ± {std_metrics['MAPE']:.2f}%"
)


# =========================
# ENTRENAMIENTO FINAL
# =========================
print("\n" + "=" * 70)
print("ENTRENAMIENTO FINAL CON TODOS LOS DATOS")
print("=" * 70)

final_model = RandomForestRegressor(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
)

final_model.fit(X, y)

feature_importance = pd.DataFrame({
    "feature": available_features,
    "importance": final_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 10 variables más importantes:")
print(feature_importance.head(10).to_string(index=False))

# Guardar artefacto completo
artifact = {
    "model": final_model,
    "features": available_features,
    "target": TARGET_COLUMN,
    "datetime_column": DATETIME_COLUMN,
    "metrics_mean": avg_metrics,
    "metrics_std": std_metrics,
    "feature_importance": feature_importance,
}

joblib.dump(artifact, MODEL_PATH)

print(f"\nModelo guardado en:\n{MODEL_PATH}")
print("\nEntrenamiento finalizado correctamente.")