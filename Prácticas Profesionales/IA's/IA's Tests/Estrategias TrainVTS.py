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

# Partición temporal real Train-Validation-Test
TRAIN_DAYS = 65
VALIDATION_DAYS = 15
TEST_DAYS = 15

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

    Esto permite ejecutar el script tanto con la nomenclatura histórica como
    con la nomenclatura corregida utilizada en la tesis.
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
    exactamente cero, de acuerdo con la regla documentada en la tesis.
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


def entrenar_y_predecir_arima(train_data, pred_data, target):
    """
    Ajusta ARIMA(5,1,0) exclusivamente con la variable objetivo
    y genera predicciones para la longitud del conjunto posterior.
    """
    model = ARIMA(train_data[target], order=(5, 1, 0))
    fit = model.fit()
    pred = fit.forecast(steps=len(pred_data))
    return np.asarray(pred)


def entrenar_y_predecir_prophet(train_data, pred_data, target):
    """
    Ajusta Prophet con la misma configuración utilizada en los demás
    experimentos. No se incorporan regresores externos para mantener
    consistencia entre estrategias.
    """
    prophet_train = (
        train_data
        .reset_index()[["fecha_hora", target]]
        .rename(columns={"fecha_hora": "ds", target: "y"})
    )

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )

    model.fit(prophet_train)

    future = model.make_future_dataframe(
        periods=len(pred_data),
        freq="5min",
        include_history=False,
    )

    pred = model.predict(future)["yhat"].to_numpy()
    return pred


def entrenar_y_predecir_rf(train_data, pred_data, target, features):
    """
    Ajusta Random Forest con las variables disponibles y devuelve tanto
    las predicciones como la importancia de características.
    """
    X_train = train_data[features]
    X_pred = pred_data[features]
    y_train = train_data[target]

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)

    pred = model.predict(X_pred)

    importancia = (
        pd.DataFrame({
            "Variable": features,
            "Importancia": model.feature_importances_
        })
        .sort_values("Importancia", ascending=False)
        .reset_index(drop=True)
    )

    return pred, importancia


def evaluar_modelos(train_data, eval_data, target, features, etapa):
    """
    Entrena ARIMA, Prophet y Random Forest con train_data y evalúa
    sobre eval_data.
    """
    print("\n" + "=" * 80)
    print(f"EVALUACIÓN: {etapa.upper()}")
    print("=" * 80)
    print(
        f"Entrenamiento: {train_data.index.min()} -> "
        f"{train_data.index.max()} ({len(train_data)} registros)"
    )
    print(
        f"Evaluación:    {eval_data.index.min()} -> "
        f"{eval_data.index.max()} ({len(eval_data)} registros)"
    )

    y_true = eval_data[target].values
    predicciones = {}
    importancias_rf = None

    # -------------------------------------------------------------------------
    # ARIMA
    # -------------------------------------------------------------------------
    print("\n[1/3] Entrenando ARIMA(5,1,0)...")
    try:
        predicciones["ARIMA"] = entrenar_y_predecir_arima(
            train_data, eval_data, target
        )
        print("ARIMA: OK")
    except Exception as error:
        print(f"ARIMA: ERROR -> {error}")
        predicciones["ARIMA"] = None

    # -------------------------------------------------------------------------
    # PROPHET
    # -------------------------------------------------------------------------
    print("[2/3] Entrenando Prophet...")
    try:
        predicciones["Prophet"] = entrenar_y_predecir_prophet(
            train_data, eval_data, target
        )
        print("Prophet: OK")
    except Exception as error:
        print(f"Prophet: ERROR -> {error}")
        predicciones["Prophet"] = None

    # -------------------------------------------------------------------------
    # RANDOM FOREST
    # -------------------------------------------------------------------------
    print("[3/3] Entrenando Random Forest...")
    try:
        rf_pred, importancias_rf = entrenar_y_predecir_rf(
            train_data, eval_data, target, features
        )
        predicciones["Random Forest"] = rf_pred

        print("Random Forest: OK")
        print("Principales variables:")
        for _, row in importancias_rf.head(5).iterrows():
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
    resultados = []

    print("\nMétricas:")

    for modelo, pred in predicciones.items():
        if pred is None:
            continue

        rmse, mae, r2, mape = calcular_metricas(y_true, pred)

        resultados.append({
            "Etapa": etapa,
            "Modelo": modelo,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "MAPE": mape,
            "N_Entrenamiento": len(train_data),
            "N_Evaluacion": len(eval_data),
            "Train_Inicio": train_data.index.min(),
            "Train_Fin": train_data.index.max(),
            "Eval_Inicio": eval_data.index.min(),
            "Eval_Fin": eval_data.index.max(),
        })

        print(
            f"{modelo:15s} | "
            f"RMSE: {rmse:10.4f} | "
            f"MAE: {mae:10.4f} | "
            f"R²: {r2:9.4f} | "
            f"MAPE: {mape:10.2f}%"
        )

    return pd.DataFrame(resultados), predicciones, importancias_rf


def graficar_comparacion(eval_data, target, predicciones, titulo, archivo):
    """
    Gráfica comparativa de la serie real frente a las predicciones.
    """
    plt.figure(figsize=(14, 6))

    plt.plot(
        eval_data.index,
        eval_data[target].values,
        label="Real",
        linewidth=2,
        alpha=0.8,
    )

    for modelo, pred in predicciones.items():
        if pred is not None:
            plt.plot(
                eval_data.index,
                pred,
                label=modelo,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
            )

    plt.title(titulo)
    plt.xlabel("Fecha y hora")
    plt.ylabel("Potencia (W)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(archivo, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figura guardada: {archivo}")


def graficar_error_acumulado(eval_data, target, predicciones, titulo, archivo):
    """
    Grafica el error acumulado y = real - predicción para cada modelo.
    """
    plt.figure(figsize=(14, 6))

    y_true = eval_data[target].values

    for modelo, pred in predicciones.items():
        if pred is not None:
            error = y_true - pred
            error_acumulado = np.cumsum(error)

            plt.plot(
                eval_data.index,
                error_acumulado,
                label=modelo,
                linewidth=2,
            )

    plt.axhline(y=0, linewidth=1, alpha=0.5)
    plt.title(titulo)
    plt.xlabel("Fecha y hora")
    plt.ylabel("Error acumulado (W)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(archivo, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figura guardada: {archivo}")


def guardar_predicciones(eval_data, target, predicciones, archivo):
    """
    Guarda valores reales y predicciones para análisis posterior.
    """
    salida = pd.DataFrame({
        "fecha_hora": eval_data.index,
        "Real": eval_data[target].values,
    })

    for modelo, pred in predicciones.items():
        if pred is not None:
            salida[modelo] = pred

    salida.to_csv(archivo, index=False)
    print(f"Predicciones guardadas: {archivo}")


# =============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

data = pd.read_excel(file_path)

if "fecha_hora" not in data.columns:
    raise KeyError("No se encontró la columna 'fecha_hora'.")

data["fecha_hora"] = pd.to_datetime(data["fecha_hora"])
data = data.sort_values("fecha_hora")
data.set_index("fecha_hora", inplace=True)

print("=" * 80)
print("TRAIN-VALIDATION-TEST TEMPORAL")
print("=" * 80)
print(f"Registros cargados: {len(data)}")
print(f"Rango temporal: {data.index.min()} a {data.index.max()}")
print(
    "Duración aproximada: "
    f"{(data.index.max() - data.index.min()).total_seconds() / 86400:.2f} días"
)


# =============================================================================
# VARIABLE OBJETIVO
# =============================================================================

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

features = list(dict.fromkeys(features))

print("\nVariables disponibles para Random Forest:")
for feature in features:
    print(f"  - {feature}")


# =============================================================================
# PARTICIÓN CRONOLÓGICA 65 / 15 / 15 DÍAS
# =============================================================================

inicio = data.index.min().normalize()

train_end = inicio + pd.Timedelta(days=TRAIN_DAYS)
validation_end = train_end + pd.Timedelta(days=VALIDATION_DAYS)
test_end = validation_end + pd.Timedelta(days=TEST_DAYS)

train_data = data[
    (data.index >= inicio) &
    (data.index < train_end)
].copy()

validation_data = data[
    (data.index >= train_end) &
    (data.index < validation_end)
].copy()

test_data = data[
    (data.index >= validation_end) &
    (data.index < test_end)
].copy()

if train_data.empty or validation_data.empty or test_data.empty:
    raise ValueError(
        "Alguno de los bloques Train/Validation/Test quedó vacío. "
        "Verifica el rango temporal del dataset."
    )

print("\n" + "=" * 80)
print("PARTICIÓN TEMPORAL")
print("=" * 80)

print(
    f"TRAIN      : {train_data.index.min()} -> {train_data.index.max()}"
    f"\n             {len(train_data)} registros"
)

print(
    f"VALIDATION : {validation_data.index.min()} -> "
    f"{validation_data.index.max()}"
    f"\n             {len(validation_data)} registros"
)

print(
    f"TEST       : {test_data.index.min()} -> {test_data.index.max()}"
    f"\n             {len(test_data)} registros"
)

print(
    f"\nTotal utilizado: "
    f"{len(train_data) + len(validation_data) + len(test_data)} registros"
)


# =============================================================================
# ETAPA 1: VALIDACIÓN
# =============================================================================

resultados_validacion, pred_validacion, importancia_rf_val = evaluar_modelos(
    train_data=train_data,
    eval_data=validation_data,
    target=target,
    features=features,
    etapa="Validación",
)

graficar_comparacion(
    validation_data,
    target,
    pred_validacion,
    titulo=(
        "TrainVTS - Desempeño sobre el conjunto de validación\n"
        f"Train: {TRAIN_DAYS} días | Validation: {VALIDATION_DAYS} días"
    ),
    archivo="TrainVTS_Validacion_Comparacion.png",
)

graficar_error_acumulado(
    validation_data,
    target,
    pred_validacion,
    titulo="TrainVTS - Error acumulado en validación",
    archivo="TrainVTS_Validacion_Error_Acumulado.png",
)

guardar_predicciones(
    validation_data,
    target,
    pred_validacion,
    "predicciones_trainvts_validacion.csv",
)


# =============================================================================
# ETAPA 2: ENTRENAMIENTO FINAL CON TRAIN + VALIDATION
# =============================================================================

train_validation_data = pd.concat(
    [train_data, validation_data]
).sort_index()

print("\n" + "=" * 80)
print("REENTRENAMIENTO FINAL")
print("=" * 80)
print(
    "Los modelos se ajustan nuevamente utilizando Train + Validation "
    "antes de evaluar el bloque Test."
)
print(
    f"Train + Validation: {train_validation_data.index.min()} -> "
    f"{train_validation_data.index.max()}"
)
print(f"Registros: {len(train_validation_data)}")


# =============================================================================
# ETAPA 3: TEST FINAL
# =============================================================================

resultados_test, pred_test, importancia_rf_test = evaluar_modelos(
    train_data=train_validation_data,
    eval_data=test_data,
    target=target,
    features=features,
    etapa="Test final",
)

graficar_comparacion(
    test_data,
    target,
    pred_test,
    titulo=(
        "TrainVTS - Evaluación final sobre el conjunto de prueba\n"
        f"Train + Validation: {TRAIN_DAYS + VALIDATION_DAYS} días | "
        f"Test: {TEST_DAYS} días"
    ),
    archivo="TrainVTS_Test_Comparacion.png",
)

graficar_error_acumulado(
    test_data,
    target,
    pred_test,
    titulo="TrainVTS - Error acumulado en el conjunto de prueba",
    archivo="TrainVTS_Test_Error_Acumulado.png",
)

guardar_predicciones(
    test_data,
    target,
    pred_test,
    "predicciones_trainvts_test.csv",
)


# =============================================================================
# COMPARACIÓN DE MÉTRICAS: VALIDACIÓN VS TEST
# =============================================================================

resultados_completos = pd.concat(
    [resultados_validacion, resultados_test],
    ignore_index=True
)

print("\n" + "=" * 80)
print("RESUMEN GENERAL TRAINVTS")
print("=" * 80)
print(
    resultados_completos[
        ["Etapa", "Modelo", "RMSE", "MAE", "R2", "MAPE"]
    ].to_string(index=False)
)


# Una gráfica independiente por métrica
for metrica in ["RMSE", "MAE", "R2", "MAPE"]:
    pivot = resultados_completos.pivot(
        index="Modelo",
        columns="Etapa",
        values=metrica
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 5),
        width=0.75,
    )

    ax.set_title(f"{metrica}: Validación vs Test final - TrainVTS")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("MAPE (%)" if metrica == "MAPE" else metrica)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=0)
    plt.tight_layout()

    archivo = f"TrainVTS_{metrica}_Validacion_vs_Test.png"
    plt.savefig(archivo, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figura guardada: {archivo}")


# =============================================================================
# EXPORTACIÓN DE RESULTADOS
# =============================================================================

resultados_validacion.to_csv(
    "resultados_trainvts_validacion.csv",
    index=False,
)

resultados_test.to_csv(
    "resultados_trainvts_test.csv",
    index=False,
)

resultados_completos.to_csv(
    "resultados_trainvts_completo.csv",
    index=False,
)

if importancia_rf_val is not None:
    importancia_rf_val.to_csv(
        "importancia_rf_trainvts_validacion.csv",
        index=False,
    )

if importancia_rf_test is not None:
    importancia_rf_test.to_csv(
        "importancia_rf_trainvts_test.csv",
        index=False,
    )

print("\nArchivos generados:")
print("  - resultados_trainvts_validacion.csv")
print("  - resultados_trainvts_test.csv")
print("  - resultados_trainvts_completo.csv")
print("  - predicciones_trainvts_validacion.csv")
print("  - predicciones_trainvts_test.csv")
print("  - TrainVTS_Validacion_Comparacion.png")
print("  - TrainVTS_Validacion_Error_Acumulado.png")
print("  - TrainVTS_Test_Comparacion.png")
print("  - TrainVTS_Test_Error_Acumulado.png")
print("  - TrainVTS_RMSE_Validacion_vs_Test.png")
print("  - TrainVTS_MAE_Validacion_vs_Test.png")
print("  - TrainVTS_R2_Validacion_vs_Test.png")
print("  - TrainVTS_MAPE_Validacion_vs_Test.png")

print("\n" + "=" * 80)
print("TRAINVTS COMPLETADO")
print("=" * 80)
