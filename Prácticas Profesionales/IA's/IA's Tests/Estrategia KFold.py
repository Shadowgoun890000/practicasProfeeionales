import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

file_path = (
    r"/home/to-o/practicasProfeeionales/"
    r"Prácticas Profesionales/JSON/Resultado_Homogenizado.xlsx"
)

N_FOLDS = 5

# K-Fold temporal por bloques deslizantes:
# 45 días de entrenamiento + 10 días de prueba.
# Cada nuevo fold avanza 10 días.
TRAIN_DAYS = 45
TEST_DAYS = 10
STEP_DAYS = 10

# Configuración común de Random Forest
RF_PARAMS = {
    "n_estimators": 150,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def buscar_columna(df, candidatos, obligatoria=False):
    """
    Devuelve el primer nombre de columna disponible entre varios candidatos.
    Permite ejecutar el script tanto con la nomenclatura antigua como con la
    nomenclatura corregida de la tesis.
    """
    for col in candidatos:
        if col in df.columns:
            return col

    if obligatoria:
        raise KeyError(
            f"No se encontró ninguna de las columnas requeridas: {candidatos}"
        )

    return None


def calcular_metricas(y_true, y_pred):
    """
    Calcula RMSE, MAE, R² y MAPE.

    Para MAPE se excluyen únicamente las observaciones cuyo valor real es
    exactamente cero, siguiendo la regla documentada en la tesis.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(
            np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        ) * 100
    else:
        mape = np.nan

    return rmse, mae, r2, mape


def crear_folds_temporales(data):
    """
    Genera 5 folds temporales con ventanas de entrenamiento de tamaño fijo.

    Fold 1: Train 45 días -> Test 10 días
    Fold 2: avanza 10 días -> Train 45 días -> Test 10 días
    ...
    Fold 5: termina al final de los 95 días disponibles.

    Esta estructura preserva estrictamente el orden temporal:
    todas las observaciones de entrenamiento anteceden a las de prueba.
    """
    inicio_global = data.index.min().normalize()

    folds = []

    for fold in range(1, N_FOLDS + 1):
        train_start = inicio_global + pd.Timedelta(days=(fold - 1) * STEP_DAYS)
        train_end = train_start + pd.Timedelta(days=TRAIN_DAYS)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=TEST_DAYS)

        train_data = data[
            (data.index >= train_start) &
            (data.index < train_end)
        ].copy()

        test_data = data[
            (data.index >= test_start) &
            (data.index < test_end)
        ].copy()

        if train_data.empty or test_data.empty:
            raise ValueError(
                f"El fold {fold} quedó vacío. "
                "Verifica el rango temporal y la continuidad del dataset."
            )

        folds.append({
            "fold": fold,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_data": train_data,
            "test_data": test_data,
        })

    return folds


# =============================================================================
# CARGA Y PREPARACIÓN
# =============================================================================

data = pd.read_excel(file_path)

if "fecha_hora" not in data.columns:
    raise KeyError("No se encontró la columna 'fecha_hora'.")

data["fecha_hora"] = pd.to_datetime(data["fecha_hora"])
data = data.sort_values("fecha_hora")
data.set_index("fecha_hora", inplace=True)

print("=" * 80)
print("K-FOLD TEMPORAL POR BLOQUES DESLIZANTES")
print("=" * 80)
print(f"Registros cargados: {len(data)}")
print(f"Rango temporal: {data.index.min()} a {data.index.max()}")
print(
    "Duración aproximada: "
    f"{(data.index.max() - data.index.min()).total_seconds() / 86400:.2f} días"
)

# Variable objetivo: se admite la nomenclatura corregida o la histórica.
target = buscar_columna(
    data,
    ["valor (W)", "valor (kWh)", "valor"],
    obligatoria=True
)

print(f"Variable objetivo utilizada: {target}")


# =============================================================================
# VARIABLES TEMPORALES
# =============================================================================

data["hora"] = data.index.hour
data["dia_semana"] = data.index.dayofweek
data["es_fin_semana"] = data["dia_semana"].isin([5, 6]).astype(int)
data["mes"] = data.index.month


# =============================================================================
# VARIABLES PREDICTORAS
# =============================================================================

# Se mantienen las mismas variables generales de los experimentos anteriores.
columnas_candidatas = [
    ["eToday", "eToday (kWh)"],
    ["eTotal", "eTotal (kWh)"],
    ["air_temp"],
    ["relative_humidity"],
    ["power", "power (kW)", "Power"],
    ["wind_speed_10m"],
    ["wind_direction_10m"],
    ["ghi"],
    ["dni"],
    ["gti"],
]

features = []

for candidatos in columnas_candidatas:
    col = buscar_columna(data, candidatos)
    if col is not None:
        features.append(col)

features += [
    "hora",
    "dia_semana",
    "es_fin_semana",
    "mes",
]

# Eliminar posibles duplicados conservando orden
features = list(dict.fromkeys(features))

print("\nVariables disponibles para Random Forest:")
for feature in features:
    print(f"  - {feature}")


# =============================================================================
# GENERACIÓN DE FOLDS
# =============================================================================

folds = crear_folds_temporales(data)

print("\n" + "=" * 80)
print("ESTRUCTURA DE LOS FOLDS")
print("=" * 80)

for info in folds:
    train_data = info["train_data"]
    test_data = info["test_data"]

    print(
        f"\nFold {info['fold']}:"
        f"\n  Train: {train_data.index.min()} -> {train_data.index.max()}"
        f"\n         {len(train_data)} registros"
        f"\n  Test : {test_data.index.min()} -> {test_data.index.max()}"
        f"\n         {len(test_data)} registros"
    )


# =============================================================================
# ENTRENAMIENTO Y EVALUACIÓN
# =============================================================================

resultados = []
predicciones_folds = {}

for info in folds:
    fold = info["fold"]
    train_data = info["train_data"]
    test_data = info["test_data"]

    print("\n" + "=" * 80)
    print(f"FOLD {fold}/{N_FOLDS}")
    print("=" * 80)

    y_train = train_data[target]
    y_test = test_data[target]

    predicciones = {}

    # -------------------------------------------------------------------------
    # 1. ARIMA
    # -------------------------------------------------------------------------
    print("\n[1/3] Entrenando ARIMA(5,1,0)...")

    try:
        arima_model = ARIMA(y_train, order=(5, 1, 0))
        arima_fit = arima_model.fit()
        arima_pred = arima_fit.forecast(steps=len(test_data))
        arima_pred = np.asarray(arima_pred)

        predicciones["ARIMA"] = arima_pred
        print("ARIMA: OK")

    except Exception as error:
        print(f"ARIMA: ERROR -> {error}")
        predicciones["ARIMA"] = None

    # -------------------------------------------------------------------------
    # 2. PROPHET
    # -------------------------------------------------------------------------
    # Se mantiene como modelo temporal sin regresores externos para que su
    # configuración sea comparable con TimeSeriesSplit y Train-Test Split.
    print("[2/3] Entrenando Prophet...")

    try:
        prophet_train = (
            train_data
            .reset_index()[["fecha_hora", target]]
            .rename(columns={"fecha_hora": "ds", target: "y"})
        )

        prophet_model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,
        )

        prophet_model.fit(prophet_train)

        future = prophet_model.make_future_dataframe(
            periods=len(test_data),
            freq="5min",
            include_history=False,
        )

        prophet_pred = prophet_model.predict(future)["yhat"].to_numpy()

        predicciones["Prophet"] = prophet_pred
        print("Prophet: OK")

    except Exception as error:
        print(f"Prophet: ERROR -> {error}")
        predicciones["Prophet"] = None

    # -------------------------------------------------------------------------
    # 3. RANDOM FOREST
    # -------------------------------------------------------------------------
    print("[3/3] Entrenando Random Forest...")

    try:
        X_train = train_data[features]
        X_test = test_data[features]

        rf_model = RandomForestRegressor(**RF_PARAMS)
        rf_model.fit(X_train, y_train)

        rf_pred = rf_model.predict(X_test)
        predicciones["Random Forest"] = rf_pred

        importancia = (
            pd.DataFrame({
                "Variable": features,
                "Importancia": rf_model.feature_importances_
            })
            .sort_values("Importancia", ascending=False)
        )

        print("Random Forest: OK")
        print("Principales variables:")
        for _, row in importancia.head(5).iterrows():
            print(
                f"  - {row['Variable']}: "
                f"{row['Importancia']:.4f}"
            )

    except Exception as error:
        print(f"Random Forest: ERROR -> {error}")
        predicciones["Random Forest"] = None

    # -------------------------------------------------------------------------
    # MÉTRICAS
    # -------------------------------------------------------------------------
    print("\nMétricas:")

    for modelo, pred in predicciones.items():
        if pred is None:
            continue

        rmse, mae, r2, mape = calcular_metricas(y_test.values, pred)

        resultados.append({
            "Fold": fold,
            "Modelo": modelo,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "MAPE": mape,
            "Train_Inicio": train_data.index.min(),
            "Train_Fin": train_data.index.max(),
            "Test_Inicio": test_data.index.min(),
            "Test_Fin": test_data.index.max(),
            "N_Train": len(train_data),
            "N_Test": len(test_data),
        })

        print(
            f"{modelo:15s} | "
            f"RMSE: {rmse:10.4f} | "
            f"MAE: {mae:10.4f} | "
            f"R²: {r2:9.4f} | "
            f"MAPE: {mape:10.2f}%"
        )

    predicciones_folds[fold] = {
        "fechas": test_data.index,
        "real": y_test.values,
        **predicciones,
    }

    # -------------------------------------------------------------------------
    # GRÁFICA DEL FOLD
    # -------------------------------------------------------------------------
    plt.figure(figsize=(14, 6))

    plt.plot(
        test_data.index,
        y_test.values,
        label="Real",
        linewidth=2,
        alpha=0.8,
    )

    for modelo, pred in predicciones.items():
        if pred is not None:
            plt.plot(
                test_data.index,
                pred,
                label=modelo,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
            )

    plt.title(
        f"K-Fold temporal por bloques - Fold {fold}\n"
        f"Train: {TRAIN_DAYS} días | Test: {TEST_DAYS} días"
    )
    plt.xlabel("Fecha y hora")
    plt.ylabel("Potencia (W)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    nombre_figura = f"KFold_Temporal_Fold_{fold}.png"
    plt.savefig(nombre_figura, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figura guardada: {nombre_figura}")


# =============================================================================
# RESULTADOS GENERALES
# =============================================================================

df_resultados = pd.DataFrame(resultados)

print("\n" + "=" * 80)
print("RESULTADOS POR FOLD")
print("=" * 80)
print(
    df_resultados[
        ["Fold", "Modelo", "RMSE", "MAE", "R2", "MAPE"]
    ].to_string(index=False)
)


# =============================================================================
# PROMEDIO Y DESVIACIÓN ESTÁNDAR
# =============================================================================

resumen = (
    df_resultados
    .groupby("Modelo")
    .agg(
        RMSE_promedio=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        MAE_promedio=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        R2_promedio=("R2", "mean"),
        R2_std=("R2", "std"),
        MAPE_promedio=("MAPE", "mean"),
        MAPE_std=("MAPE", "std"),
    )
    .reset_index()
)

print("\n" + "=" * 80)
print("PROMEDIO Y DESVIACIÓN ESTÁNDAR ENTRE LOS 5 FOLDS")
print("=" * 80)
print(resumen.to_string(index=False))


# =============================================================================
# GRÁFICA DE MÉTRICAS POR FOLD
# =============================================================================

for metrica in ["RMSE", "MAE", "R2", "MAPE"]:
    plt.figure(figsize=(10, 5))

    for modelo in df_resultados["Modelo"].unique():
        subset = df_resultados[df_resultados["Modelo"] == modelo]

        plt.plot(
            subset["Fold"],
            subset[metrica],
            marker="o",
            linewidth=2,
            label=modelo,
        )

    plt.xlabel("Fold")
    plt.ylabel("MAPE (%)" if metrica == "MAPE" else metrica)
    plt.title(f"{metrica} por fold - K-Fold temporal por bloques")
    plt.xticks(range(1, N_FOLDS + 1))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    archivo = f"KFold_Temporal_{metrica}.png"
    plt.savefig(archivo, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figura guardada: {archivo}")


# =============================================================================
# EXPORTACIÓN
# =============================================================================

df_resultados.to_csv(
    "resultados_kfold_temporal_folds.csv",
    index=False,
)

resumen.to_csv(
    "resultados_kfold_temporal_resumen.csv",
    index=False,
)

print("\nArchivos generados:")
print("  - resultados_kfold_temporal_folds.csv")
print("  - resultados_kfold_temporal_resumen.csv")
print("  - KFold_Temporal_Fold_1.png ... KFold_Temporal_Fold_5.png")
print("  - KFold_Temporal_RMSE.png")
print("  - KFold_Temporal_MAE.png")
print("  - KFold_Temporal_R2.png")
print("  - KFold_Temporal_MAPE.png")

print("\n" + "=" * 80)
print("K-FOLD TEMPORAL COMPLETADO")
print("=" * 80)
