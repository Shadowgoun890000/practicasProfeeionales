import pandas as pd

from app.config import DATA_FILE, DATETIME_COLUMN


def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE)

    if DATETIME_COLUMN not in df.columns:
        raise ValueError(f"No se encontró la columna '{DATETIME_COLUMN}' en el archivo de datos.")

    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN])

    # Eliminar columnas donde todos los valores son 0
    df = df.loc[:, (df != 0).any(axis=0)]

    # Ordenar por fecha por seguridad
    df = df.sort_values(DATETIME_COLUMN).reset_index(drop=True)

    return df