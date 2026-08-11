import pandas as pd

from app.config import (
    DATA_FILE,
    DATETIME_COLUMN,
    SOURCE_TARGET_COLUMN,
    TARGET_COLUMN,
)


def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE)

    # =========================
    # Homologación de nombres
    # =========================
    #
    # El archivo histórico conserva el nombre "valor (kWh)",
    # pero la variable corresponde a potencia instantánea en W.
    # Se corrige únicamente al cargar los datos para no modificar
    # el archivo fuente original.
    if SOURCE_TARGET_COLUMN in df.columns and TARGET_COLUMN not in df.columns:
        df = df.rename(
            columns={
                SOURCE_TARGET_COLUMN: TARGET_COLUMN
            }
        )

    # =========================
    # Validaciones
    # =========================
    if DATETIME_COLUMN not in df.columns:
        raise ValueError(
            f"No se encontró la columna '{DATETIME_COLUMN}' "
            "en el archivo de datos."
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"No se encontró la columna objetivo '{TARGET_COLUMN}' "
            "en el archivo de datos."
        )

    # Conversión temporal
    df[DATETIME_COLUMN] = pd.to_datetime(
        df[DATETIME_COLUMN],
        errors="coerce"
    )

    # Eliminar registros sin fecha válida
    df = df.dropna(subset=[DATETIME_COLUMN])

    # Eliminar columnas donde todos los valores son 0
    df = df.loc[:, (df != 0).any(axis=0)]

    # Orden cronológico
    df = (
        df.sort_values(DATETIME_COLUMN)
        .reset_index(drop=True)
    )

    return df