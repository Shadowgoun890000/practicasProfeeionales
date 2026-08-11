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

# Periodos históricos de evaluación
PERIODOS = {
    "30_dias": 30,
    "60_dias": 60,
    "90_dias": 90,
}

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
    Permite trabajar con la nomenclatura histórica o con la corregida.
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

    Para MAPE se excluyen únicamente los valores reales iguales a cero.
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


def entrenar_arima(train_data, test_data, target):
    """
    Ajusta ARIMA(5,1,0) únicamente con la variable objetivo.
    """
    model = ARIMA(train_data[target], order=(5, 1, 0))
    fit = model.fit()
    pred = fit.forecast(steps=len(test_data))
    return np.asarray(pred)


def entrenar_prophet(train_data, test_data, target):
    """
    Ajusta Prophet con la misma configuración utilizada en K-Fold temporal
    y TrainVTS. No utiliza regresores externos.
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
        periods=len(test_data),
        freq="5min",
        include_history=False,
    )

    pred = model.predict(future)["yhat"].to_numpy()
    return pred


def entrenar_random_forest(train_data, test_data, target, features):
    """
    Ajusta Random Forest con exactamente la misma configuración utilizada
    en las demás estrategias homologadas.
    """
    X_train = train_data[features]
    X_test = test_data[features]
    y_train = train_data[target]

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    importancia = (
        pd.DataFrame({
            "Variable": features,
            "Importancia": model.feature_importances_
        })
        .sort_values("Importancia", ascending=False)
        .reset_index(drop=True)
    )

    return pred, importancia


def graficar_comparacion(test_data, target, predicciones, dias):
    """
    Muestra valores reales y predicciones para el periodo evaluado.
    """
    plt.figure(figsize=(14, 6))

    plt.plot(
        test_data.index,
        test_data[target].values,
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
        f"Train-Test Split temporal - Periodo de evaluación: {dias} días"
    )
    plt.xlabel("Fecha y hora")
    plt.ylabel("Potencia (W)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    archivo = f"TrainTS_Comparacion_{dias}_dias.png"
    plt.savefig(archivo, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figura guardada: {archivo}")


def graficar_error_acumulado(test_data, target, predicciones, dias):
    """
    Grafica el error acumulado real - predicción para cada modelo.
    """
    plt.figure(figsize=(14, 6))

    y_true = test_data[target].values

    for modelo, pred in predicciones.items():
        if pred is not None:
            error = y_true - pred
            error_acumulado = np.cumsum(error)

            plt.plot(
                test_data.index,
                error_acumulado,
                label=modelo,
                linewidth=2,
            )

    plt.axhline(y=0, linewidth=1, alpha=0.5)
    plt.title(
        f"Train-Test Split - Error acumulado ({dias} días de evaluación)"
    )
    plt.xlabel("Fecha y hora")
    plt.ylabel("Error acumulado (W)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    archivo = f"TrainTS_Error_Acumulado_{dias}_dias.png"
    plt.savefig(archivo, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figura guardada: {archivo}")


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
print("TRAIN-TEST SPLIT TEMPORAL POR PERIODOS")
print("=" * 80)
print(f"Registros cargados: {len(data)}")
print(f"Rango temporal: {data.index.min()} a {data.index.max()}")
print(
    "Duración aproximada: "
    f"{(data.index.max() - data.index.min()).total_seconds() / 86400:.2f} días"
)

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
# EVALUACIÓN POR PERIODOS
# =============================================================================

resultados = []
predicciones_todos = {}
importancias_todos = {}

for nombre_periodo, dias in PERIODOS.items():
    print("\n" + "=" * 80)
    print(f"PERIODO HISTÓRICO DE EVALUACIÓN: {dias} DÍAS")
    print("=" * 80)

    # Últimos N días como prueba; todo lo anterior como entrenamiento.
    fecha_inicio_test = data.index.max().normalize() - pd.Timedelta(days=dias - 1)

    test_data = data[data.index >= fecha_inicio_test].copy()
    train_data = data[data.index < fecha_inicio_test].copy()

    if train_data.empty:
        print(
            f"ERROR: no existe historial suficiente para entrenar "
            f"con un periodo de evaluación de {dias} días."
        )
        continue

    if test_data.empty:
        print(f"ERROR: el conjunto de prueba de {dias} días está vacío.")
        continue

    print(
        f"Entrenamiento: {train_data.index.min()} -> "
        f"{train_data.index.max()} ({len(train_data)} registros)"
    )
    print(
        f"Prueba:        {test_data.index.min()} -> "
        f"{test_data.index.max()} ({len(test_data)} registros)"
    )
    print(
        f"Proporción aproximada Train/Test: "
        f"{len(train_data) / len(data) * 100:.2f}% / "
        f"{len(test_data) / len(data) * 100:.2f}%"
    )

    y_test = test_data[target].values
    predicciones = {}
    importancia_rf = None

    # -------------------------------------------------------------------------
    # ARIMA
    # -------------------------------------------------------------------------
    print("\n[1/3] Entrenando ARIMA(5,1,0)...")
    try:
        predicciones["ARIMA"] = entrenar_arima(
            train_data, test_data, target
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
        predicciones["Prophet"] = entrenar_prophet(
            train_data, test_data, target
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
        rf_pred, importancia_rf = entrenar_random_forest(
            train_data, test_data, target, features
        )

        predicciones["Random Forest"] = rf_pred

        print("Random Forest: OK")
        print("Principales variables:")
        for _, row in importancia_rf.head(5).iterrows():
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

        rmse, mae, r2, mape = calcular_metricas(y_test, pred)

        resultados.append({
            "Periodo": nombre_periodo,
            "Dias_Evaluacion": dias,
            "Modelo": modelo,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "MAPE": mape,
            "N_Train": len(train_data),
            "N_Test": len(test_data),
            "Train_Inicio": train_data.index.min(),
            "Train_Fin": train_data.index.max(),
            "Test_Inicio": test_data.index.min(),
            "Test_Fin": test_data.index.max(),
        })

        print(
            f"{modelo:15s} | "
            f"RMSE: {rmse:10.4f} | "
            f"MAE: {mae:10.4f} | "
            f"R²: {r2:9.4f} | "
            f"MAPE: {mape:10.2f}%"
        )

    predicciones_todos[nombre_periodo] = {
        "test_data": test_data,
        "predicciones": predicciones,
    }

    if importancia_rf is not None:
        importancias_todos[nombre_periodo] = importancia_rf
        importancia_rf.to_csv(
            f"importancia_rf_traints_{dias}_dias.csv",
            index=False,
        )

    # -------------------------------------------------------------------------
    # GRÁFICAS
    # -------------------------------------------------------------------------
    graficar_comparacion(
        test_data,
        target,
        predicciones,
        dias,
    )

    graficar_error_acumulado(
        test_data,
        target,
        predicciones,
        dias,
    )

    # Guardar predicciones
    df_pred = pd.DataFrame({
        "fecha_hora": test_data.index,
        "Real": y_test,
    })

    for modelo, pred in predicciones.items():
        if pred is not None:
            df_pred[modelo] = pred

    df_pred.to_csv(
        f"predicciones_traints_{dias}_dias.csv",
        index=False,
    )


# =============================================================================
# RESUMEN GENERAL
# =============================================================================

df_resultados = pd.DataFrame(resultados)

print("\n" + "=" * 80)
print("RESUMEN GENERAL TRAIN-TEST SPLIT")
print("=" * 80)

print(
    df_resultados[
        ["Dias_Evaluacion", "Modelo", "RMSE", "MAE", "R2", "MAPE"]
    ].to_string(index=False)
)


# =============================================================================
# GRÁFICAS DE MÉTRICAS POR PERIODO
# =============================================================================

for metrica in ["RMSE", "MAE", "R2", "MAPE"]:
    plt.figure(figsize=(10, 5))

    for modelo in df_resultados["Modelo"].unique():
        subset = (
            df_resultados[df_resultados["Modelo"] == modelo]
            .sort_values("Dias_Evaluacion")
        )

        plt.plot(
            subset["Dias_Evaluacion"],
            subset[metrica],
            marker="o",
            linewidth=2,
            label=modelo,
        )

    plt.xlabel("Días del periodo histórico de evaluación")
    plt.ylabel("MAPE (%)" if metrica == "MAPE" else metrica)
    plt.title(
        f"{metrica} por periodo - Train-Test Split temporal"
    )
    plt.xticks(sorted(PERIODOS.values()))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    archivo = f"TrainTS_{metrica}_por_periodo.png"
    plt.savefig(archivo, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figura guardada: {archivo}")


# =============================================================================
# EXPORTACIÓN
# =============================================================================

df_resultados.to_csv(
    "resultados_traints_periodos_homologados.csv",
    index=False,
)

resumen_modelos = (
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

resumen_modelos.to_csv(
    "resultados_traints_resumen_modelos.csv",
    index=False,
)

print("\nArchivos principales generados:")
print("  - resultados_traints_periodos_homologados.csv")
print("  - resultados_traints_resumen_modelos.csv")
print("  - predicciones_traints_30_dias.csv")
print("  - predicciones_traints_60_dias.csv")
print("  - predicciones_traints_90_dias.csv")
print("  - TrainTS_Comparacion_30_dias.png")
print("  - TrainTS_Comparacion_60_dias.png")
print("  - TrainTS_Comparacion_90_dias.png")
print("  - TrainTS_Error_Acumulado_30_dias.png")
print("  - TrainTS_Error_Acumulado_60_dias.png")
print("  - TrainTS_Error_Acumulado_90_dias.png")
print("  - TrainTS_RMSE_por_periodo.png")
print("  - TrainTS_MAE_por_periodo.png")
print("  - TrainTS_R2_por_periodo.png")
print("  - TrainTS_MAPE_por_periodo.png")

print("\n" + "=" * 80)
print("TRAIN-TEST SPLIT HOMOLOGADO COMPLETADO")
print("=" * 80)