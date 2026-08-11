from pathlib import Path

BASE_DIR = Path("/home/to-o/practicasProfeeionales/Prácticas Profesionales")
DATA_FILE = BASE_DIR / "JSON" / "Resultado_Homogenizado.xlsx"
MODEL_FILE = BASE_DIR / "models" / "random_forest_60d.joblib"

# =========================
# COLUMNAS PRINCIPALES
# =========================

# Nombre que todavía conserva el Excel original
SOURCE_TARGET_COLUMN = "valor (kWh)"

# Nombre correcto utilizado por la aplicación
TARGET_COLUMN = "valor (W)"

DATETIME_COLUMN = "fecha_hora"

DEFAULT_PREDICTION_DAYS = 60
MODEL_NAME = "Random Forest"

COLUMNAS_ENERGIA = [
    "fecha_hora",
    "valor (W)",
    "nominalPower (W)",
    "eToday (kWh)",
    "eTotal (kWh)",
    "power (kW)",
]

COLUMNAS_CLIMA = [
    "fecha_hora",
    "air_temp",
    "albedo",
    "azimuth",
    "cloud_opacity",
    "dhi",
    "dni",
    "ghi",
    "gti",
    "precipitable_water",
    "relative_humidity",
    "surface_pressure",
    "wind_direction_100m",
    "wind_direction_10m",
    "wind_speed_100m",
    "wind_speed_10m",
    "wind_gust",
]