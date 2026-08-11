import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
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

N_SPLITS = 5

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
    Permite ejecutar el script con la nomenclatura histórica o la corregida.
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

    Para MAPE se excluyen únicamente las observaciones cuyo valor real
    es exactamente cero.
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
    """Ajusta ARIMA(5,1,0) únicamente con la variable objetivo."""
    model = ARIMA(train_data[target], order=(5, 1, 0))
    fit = model.fit()
    pred = fit.forecast(steps=len(test_data))
    return np.asarray(pred)


def entrenar_prophet(train_data, test_data, target):
    """
    Ajusta Prophet con la misma configuración utilizada en las demás
    estrategias homologadas. No utiliza regresores externos.
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
    Ajusta Random Forest con la misma configuración e idéntico conjunto
    de variables utilizado en las demás estrategias homologadas.
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


def graficar_fold(test_data, target, predicciones, fold):
    """Compara valores reales y predicciones en un fold."""
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

    plt.title(f"TimeSeriesSplit - Fold {fold}")
    plt.xlabel("Fecha y hora")
    plt.ylabel("Potencia (W)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    archivo = f"TimeSCV_Fold_{fold}.png"
    plt.savefig(archivo, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figura guardada: {archivo}")


def graficar_error_acumulado(test_data, target, predicciones, fold):
    """Grafica error acumulado real - predicción en cada fold."""
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
    plt.title(f"TimeSeriesSplit - Error acumulado - Fold {fold}")
    plt.xlabel("Fecha y hora")
    plt.ylabel("Error acumulado (W)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    archivo = f"TimeSCV_Error_Acumulado_Fold_{fold}.png"
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
print("TIME SERIES CROSS-VALIDATION - TimeSeriesSplit")
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
# TIMESERIESSPLIT
# =============================================================================

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

resultados = []
info_folds = []
importancias_rf = {}

for fold, (train_index, test_index) in enumerate(tscv.split(data), start=1):
    train_data = data.iloc[train_index].copy()
    test_data = data.iloc[test_index].copy()

    print("\n" + "=" * 80)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("=" * 80)

    print(
        f"Entrenamiento: {train_data.index.min()} -> "
        f"{train_data.index.max()} ({len(train_data)} registros)"
    )
    print(
        f"Prueba:        {test_data.index.min()} -> "
        f"{test_data.index.max()} ({len(test_data)} registros)"
    )

    info_folds.append({
        "Fold": fold,
        "Train_Inicio": train_data.index.min(),
        "Train_Fin": train_data.index.max(),
        "Test_Inicio": test_data.index.min(),
        "Test_Fin": test_data.index.max(),
        "N_Train": len(train_data),
        "N_Test": len(test_data),
    })

    y_test = test_data[target].values
    predicciones = {}
    importancia = None

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
        rf_pred, importancia = entrenar_random_forest(
            train_data, test_data, target, features
        )

        predicciones["Random Forest"] = rf_pred
        importancias_rf[fold] = importancia

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

        rmse, mae, r2, mape = calcular_metricas(y_test, pred)

        resultados.append({
            "Fold": fold,
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

    # -------------------------------------------------------------------------
    # GRÁFICAS
    # -------------------------------------------------------------------------
    graficar_fold(
        test_data,
        target,
        predicciones,
        fold,
    )

    graficar_error_acumulado(
        test_data,
        target,
        predicciones,
        fold,
    )

    # -------------------------------------------------------------------------
    # GUARDAR PREDICCIONES DEL FOLD
    # -------------------------------------------------------------------------
    df_pred = pd.DataFrame({
        "fecha_hora": test_data.index,
        "Real": y_test,
    })

    for modelo, pred in predicciones.items():
        if pred is not None:
            df_pred[modelo] = pred

    df_pred.to_csv(
        f"predicciones_timescv_fold_{fold}.csv",
        index=False,
    )

    if importancia is not None:
        importancia.to_csv(
            f"importancia_rf_timescv_fold_{fold}.csv",
            index=False,
        )


# =============================================================================
# RESULTADOS POR FOLD
# =============================================================================

df_resultados = pd.DataFrame(resultados)
df_info_folds = pd.DataFrame(info_folds)

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
print("PROMEDIO Y DESVIACIÓN ESTÁNDAR ENTRE FOLDS")
print("=" * 80)
print(resumen.to_string(index=False))


# =============================================================================
# EVOLUCIÓN DE MÉTRICAS POR FOLD
# =============================================================================

for metrica in ["RMSE", "MAE", "R2", "MAPE"]:
    plt.figure(figsize=(10, 5))

    for modelo in df_resultados["Modelo"].unique():
        subset = (
            df_resultados[df_resultados["Modelo"] == modelo]
            .sort_values("Fold")
        )

        plt.plot(
            subset["Fold"],
            subset[metrica],
            marker="o",
            linewidth=2,
            label=modelo,
        )

    plt.xlabel("Fold")
    plt.ylabel("MAPE (%)" if metrica == "MAPE" else metrica)
    plt.title(f"{metrica} por fold - TimeSeriesSplit")
    plt.xticks(range(1, N_SPLITS + 1))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    archivo = f"TimeSCV_{metrica}_por_fold.png"
    plt.savefig(archivo, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figura guardada: {archivo}")


# =============================================================================
# VISUALIZACIÓN OPCIONAL SOBRE TODO EL CONJUNTO
# =============================================================================
#
# IMPORTANTE:
# Esta sección NO calcula métricas ni se utiliza como evaluación fuera de muestra.
# Solo permite visualizar el ajuste de cada modelo al conjunto completo.
#
# Para evitar confusión metodológica, se deja desactivada por defecto.
#

GENERAR_VISUALIZACION_AJUSTE_COMPLETO = False

if GENERAR_VISUALIZACION_AJUSTE_COMPLETO:
    print("\n" + "=" * 80)
    print("AJUSTE SOBRE EL CONJUNTO COMPLETO - SOLO VISUALIZACIÓN")
    print("=" * 80)

    pred_ajuste = {}

    try:
        arima_model = ARIMA(data[target], order=(5, 1, 0))
        arima_fit = arima_model.fit()
        pred_ajuste["ARIMA"] = np.asarray(arima_fit.fittedvalues)
    except Exception as error:
        print(f"ARIMA ajuste completo: ERROR -> {error}")
        pred_ajuste["ARIMA"] = None

    try:
        prophet_data = (
            data
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

        prophet_model.fit(prophet_data)

        future = prophet_model.make_future_dataframe(
            periods=0,
            freq="5min"
        )

        pred_ajuste["Prophet"] = (
            prophet_model.predict(future)["yhat"].to_numpy()
        )

    except Exception as error:
        print(f"Prophet ajuste completo: ERROR -> {error}")
        pred_ajuste["Prophet"] = None

    try:
        rf_model = RandomForestRegressor(**RF_PARAMS)
        rf_model.fit(data[features], data[target])
        pred_ajuste["Random Forest"] = rf_model.predict(data[features])

    except Exception as error:
        print(f"RF ajuste completo: ERROR -> {error}")
        pred_ajuste["Random Forest"] = None

    plt.figure(figsize=(15, 7))

    plt.plot(
        data.index,
        data[target],
        label="Real",
        linewidth=1.5,
        alpha=0.7,
    )

    for modelo, pred in pred_ajuste.items():
        if pred is not None:
            # ARIMA fittedvalues puede ser ligeramente más corto
            n = min(len(data), len(pred))
            plt.plot(
                data.index[-n:],
                np.asarray(pred)[-n:],
                label=modelo,
                linewidth=1,
                alpha=0.7,
            )

    plt.title(
        "Ajuste de los modelos sobre el conjunto completo "
        "(solo visualización)"
    )
    plt.xlabel("Fecha y hora")
    plt.ylabel("Potencia (W)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        "TimeSCV_Ajuste_Completo_Solo_Visualizacion.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# =============================================================================
# EXPORTACIÓN
# =============================================================================

df_resultados.to_csv(
    "resultados_timescv_folds_homologados.csv",
    index=False,
)

df_info_folds.to_csv(
    "informacion_folds_timescv_homologada.csv",
    index=False,
)

resumen.to_csv(
    "resultados_timescv_resumen_modelos.csv",
    index=False,
)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')

# Cargar el archivo y configurar la columna de fecha
file_path = r"/JSON/Resultado_Homogenizado.xlsx"
data = pd.read_excel(file_path)

data['fecha_hora'] = pd.to_datetime(data['fecha_hora'])
data.set_index('fecha_hora', inplace=True)

# Verificar datos
print(f"Datos cargados: {len(data)} registros")
print(f"Rango temporal: {data.index.min()} a {data.index.max()}")
print(f"Total días: {(data.index.max() - data.index.min()).days}")

# Variables
target = 'valor (kWh)'
features = [
    # Variables del inversor
    'eToday (kWh)', 'eTotal (kWh)',
    # Variables meteorológicas/energéticas
    'air_temp', 'relative_humidity',
    'power (kW)',
    'wind_speed_10m', 'wind_direction_10m',
    'ghi', 'dni', 'gti'
]
features = [col for col in features if col in data.columns]

# Añadir características temporales
data['hora'] = data.index.hour
data['dia_semana'] = data.index.dayofweek
data['es_fin_semana'] = data['dia_semana'].isin([5, 6]).astype(int)
data['mes'] = data.index.month

# Actualizar features con características temporales
features_extendidas = features + ['hora', 'dia_semana', 'es_fin_semana', 'mes']
features_extendidas = [col for col in features_extendidas if col in data.columns]


def calcular_metricas(y_true, y_pred):
    """Calcula RMSE, MAE, R² y MAPE."""

    try:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Se excluyen únicamente los valores reales iguales a cero.
        mask = y_true != 0

        if np.any(mask):
            mape = np.mean(
                np.abs(
                    (y_true[mask] - y_pred[mask]) /
                    y_true[mask]
                )
            ) * 100
        else:
            mape = np.nan

        return rmse, mae, r2, mape

    except Exception as error:
        print(f'Error calculando métricas: {error}')
        return np.nan, np.nan, np.nan, np.nan


# Time Series Cross-Validation (TimeSeriesSplit)
print("\n" + "=" * 80)
print("TIME SERIES CROSS-VALIDATION (TimeSeriesSplit)")
print("=" * 80)

# Configurar TimeSeriesSplit
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

# Listas para almacenar métricas de cada fold
arima_metrics_folds = []
prophet_metrics_folds = []
rf_metrics_folds = []

# Información de cada fold para análisis
folds_info = []

for fold, (train_index, test_index) in enumerate(tscv.split(data), 1):
    print(f"\n{'=' * 60}")
    print(f"FOLD {fold}/{n_splits}")
    print(f"{'=' * 60}")

    # Dividir datos manteniendo orden temporal
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]

    print(f"Entrenamiento: {train_data.index.min()} a {train_data.index.max()}")
    print(f"Prueba: {test_data.index.min()} a {test_data.index.max()}")
    print(f"Muestras entrenamiento: {len(train_data)}")
    print(f"Muestras prueba: {len(test_data)}")

    # Guardar información del fold
    folds_info.append({
        'fold': fold,
        'train_start': train_data.index.min(),
        'train_end': train_data.index.max(),
        'test_start': test_data.index.min(),
        'test_end': test_data.index.max(),
        'train_samples': len(train_data),
        'test_samples': len(test_data)
    })

    # 1. MODELO ARIMA
    print("\n[1/3] Entrenando ARIMA...")
    try:
        arima_model = ARIMA(train_data[target], order=(5, 1, 0))
        arima_model_fit = arima_model.fit()
        arima_pred = arima_model_fit.forecast(steps=len(test_data))
        arima_status = "OK"
    except Exception as e:
        print(f"   Error ARIMA: {e}")
        arima_pred = np.full(len(test_data), np.nan)
        arima_status = f"Error: {str(e)[:50]}..."

    # 2. MODELO PROPHET
    print("[2/3] Entrenando Prophet...")
    try:
        prophet_train = train_data.reset_index().rename(columns={'fecha_hora': 'ds', target: 'y'})

        prophet_model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )

        prophet_model.fit(prophet_train)

        future = prophet_model.make_future_dataframe(
            periods=len(test_data),
            freq='5T',
            include_history=False
        )

        forecast = prophet_model.predict(future)
        prophet_forecast = forecast['yhat'].values
        prophet_status = "OK"

    except Exception as e:
        print(f"   Error Prophet: {e}")
        prophet_forecast = np.full(len(test_data), np.nan)
        prophet_status = f"Error: {str(e)[:50]}..."

    # 3. MODELO RANDOM FOREST
    print("[3/3] Entrenando Random Forest...")
    try:
        # Verificar features disponibles
        features_disponibles = [f for f in features_extendidas if f in train_data.columns and f in test_data.columns]

        if len(features_disponibles) > 0:
            X_train = train_data[features_disponibles]
            X_test = test_data[features_disponibles]
            y_train = train_data[target]
            y_test = test_data[target]

            rf_model = RandomForestRegressor(
                n_estimators=150,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)

            # Calcular importancia de características
            importancia = pd.DataFrame({
                'Característica': features_disponibles,
                'Importancia': rf_model.feature_importances_
            }).sort_values('Importancia', ascending=False)

            print(f"   Top 3 características importantes:")
            for idx, row in importancia.head(3).iterrows():
                print(f"     - {row['Característica']}: {row['Importancia']:.4f}")

            rf_status = "OK"
        else:
            print("   No hay características disponibles")
            rf_pred = np.full(len(test_data), np.nan)
            rf_status = "Sin características"

    except Exception as e:
        print(f"   Error Random Forest: {e}")
        rf_pred = np.full(len(test_data), np.nan)
        rf_status = f"Error: {str(e)[:50]}..."

    # Calcular métricas para este fold
    metricas_arima = calcular_metricas(test_data[target].values, arima_pred) if arima_status == "OK" else (np.nan,
                                                                                                           np.nan,
                                                                                                           np.nan,
                                                                                                           np.nan)
    metricas_prophet = calcular_metricas(test_data[target].values, prophet_forecast) if prophet_status == "OK" else (
        np.nan, np.nan, np.nan, np.nan)
    metricas_rf = calcular_metricas(test_data[target].values, rf_pred) if rf_status == "OK" else (np.nan, np.nan,
                                                                                                  np.nan, np.nan)

    # Guardar métricas
    arima_metrics_folds.append(metricas_arima)
    prophet_metrics_folds.append(metricas_prophet)
    rf_metrics_folds.append(metricas_rf)

    # Imprimir métricas del fold
    print(f"\nMétricas Fold {fold}:")
    print(
        f"ARIMA - RMSE: {metricas_arima[0]:.4f}, MAE: {metricas_arima[1]:.4f}, R²: {metricas_arima[2]:.4f}, MAPE: {metricas_arima[3]:.2f}%")
    print(
        f"Prophet - RMSE: {metricas_prophet[0]:.4f}, MAE: {metricas_prophet[1]:.4f}, R²: {metricas_prophet[2]:.4f}, MAPE: {metricas_prophet[3]:.2f}%")
    print(
        f"Random Forest - RMSE: {metricas_rf[0]:.4f}, MAE: {metricas_rf[1]:.4f}, R²: {metricas_rf[2]:.4f}, MAPE: {metricas_rf[3]:.2f}%")

    # Gráfico del fold
    if arima_status == "OK" or prophet_status == "OK" or rf_status == "OK":
        plt.figure(figsize=(14, 8))

        # Gráfico de series temporales
        plt.subplot(2, 1, 1)
        plt.plot(test_data.index, test_data[target], label='Real', color='black', linewidth=2)

        if arima_status == "OK":
            plt.plot(test_data.index, arima_pred, label='ARIMA', color='blue', linestyle='--', alpha=0.7)

        if prophet_status == "OK":
            plt.plot(test_data.index, prophet_forecast, label='Prophet', color='green', linestyle='--', alpha=0.7)

        if rf_status == "OK":
            plt.plot(test_data.index, rf_pred, label='Random Forest', color='red', linestyle='--', alpha=0.7)

        plt.title(f'Fold {fold} - Comparación de Modelos')
        plt.ylabel('Generación (W)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Gráfico de errores
        plt.subplot(2, 1, 2)

        modelos = []
        if arima_status == "OK":
            modelos.append(('ARIMA', arima_pred, 'blue'))
        if prophet_status == "OK":
            modelos.append(('Prophet', prophet_forecast, 'green'))
        if rf_status == "OK":
            modelos.append(('Random Forest', rf_pred, 'red'))

        for modelo_nombre, prediccion, color in modelos:
            error = test_data[target].values - prediccion
            plt.plot(test_data.index, error, label=f'Error {modelo_nombre}', color=color, alpha=0.6)

        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Errores de Predicción')
        plt.ylabel('Error (W)')
        plt.xlabel('Fecha y Hora')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# Calcular promedios de todas las métricas
print("\n" + "=" * 80)
print("RESULTADOS PROMEDIO DE TIME SERIES CROSS-VALIDATION")
print("=" * 80)

# Convertir a arrays de numpy para cálculo
arima_metrics_array = np.array([m for m in arima_metrics_folds if not np.isnan(m[0])])
prophet_metrics_array = np.array([m for m in prophet_metrics_folds if not np.isnan(m[0])])
rf_metrics_array = np.array([m for m in rf_metrics_folds if not np.isnan(m[0])])

if len(arima_metrics_array) > 0:
    arima_avg = np.nanmean(arima_metrics_array, axis=0)
    print(f"\nARIMA (promedio de {len(arima_metrics_array)} folds válidos):")
    print(f"  RMSE: {arima_avg[0]:.4f} ± {np.nanstd(arima_metrics_array[:, 0]):.4f}")
    print(f"  MAE:  {arima_avg[1]:.4f} ± {np.nanstd(arima_metrics_array[:, 1]):.4f}")
    print(f"  R²:   {arima_avg[2]:.4f} ± {np.nanstd(arima_metrics_array[:, 2]):.4f}")
    print(f"  MAPE: {arima_avg[3]:.2f}% ± {np.nanstd(arima_metrics_array[:, 3]):.2f}%")

if len(prophet_metrics_array) > 0:
    prophet_avg = np.nanmean(prophet_metrics_array, axis=0)
    print(f"\nProphet (promedio de {len(prophet_metrics_array)} folds válidos):")
    print(f"  RMSE: {prophet_avg[0]:.4f} ± {np.nanstd(prophet_metrics_array[:, 0]):.4f}")
    print(f"  MAE:  {prophet_avg[1]:.4f} ± {np.nanstd(prophet_metrics_array[:, 1]):.4f}")
    print(f"  R²:   {prophet_avg[2]:.4f} ± {np.nanstd(prophet_metrics_array[:, 2]):.4f}")
    print(f"  MAPE: {prophet_avg[3]:.2f}% ± {np.nanstd(prophet_metrics_array[:, 3]):.2f}%")

if len(rf_metrics_array) > 0:
    rf_avg = np.nanmean(rf_metrics_array, axis=0)
    print(f"\nRandom Forest (promedio de {len(rf_metrics_array)} folds válidos):")
    print(f"  RMSE: {rf_avg[0]:.4f} ± {np.nanstd(rf_metrics_array[:, 0]):.4f}")
    print(f"  MAE:  {rf_avg[1]:.4f} ± {np.nanstd(rf_metrics_array[:, 1]):.4f}")
    print(f"  R²:   {rf_avg[2]:.4f} ± {np.nanstd(rf_metrics_array[:, 2]):.4f}")
    print(f"  MAPE: {rf_avg[3]:.2f}% ± {np.nanstd(rf_metrics_array[:, 3]):.2f}%")

# Gráfico de evolución de métricas por fold
print("\n" + "=" * 80)
print("EVOLUCIÓN DE MÉTRICAS POR FOLD")
print("=" * 80)

# Crear dataframe con métricas por fold
df_folds = pd.DataFrame({
    'Fold': range(1, n_splits + 1),
    'ARIMA_RMSE': [m[0] if not np.isnan(m[0]) else np.nan for m in arima_metrics_folds],
    'Prophet_RMSE': [m[0] if not np.isnan(m[0]) else np.nan for m in prophet_metrics_folds],
    'RF_RMSE': [m[0] if not np.isnan(m[0]) else np.nan for m in rf_metrics_folds],
    'ARIMA_R2': [m[2] if not np.isnan(m[2]) else np.nan for m in arima_metrics_folds],
    'Prophet_R2': [m[2] if not np.isnan(m[2]) else np.nan for m in prophet_metrics_folds],
    'RF_R2': [m[2] if not np.isnan(m[2]) else np.nan for m in rf_metrics_folds]
})

# Gráfico de RMSE por fold
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for modelo, color in [('ARIMA', 'blue'), ('Prophet', 'green'), ('RF', 'red')]:
    if f'{modelo}_RMSE' in df_folds.columns:
        plt.plot(df_folds['Fold'], df_folds[f'{modelo}_RMSE'],
                 label=modelo, marker='o', color=color, linewidth=2)

plt.xlabel('Fold')
plt.ylabel('RMSE')
plt.title('RMSE por Fold (TimeSeriesSplit)')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico de R² por fold
plt.subplot(1, 2, 2)
for modelo, color in [('ARIMA', 'blue'), ('Prophet', 'green'), ('RF', 'red')]:
    if f'{modelo}_R2' in df_folds.columns:
        plt.plot(df_folds['Fold'], df_folds[f'{modelo}_R2'],
                 label=modelo, marker='s', color=color, linewidth=2)

plt.xlabel('Fold')
plt.ylabel('R²')
plt.title('R² por Fold (TimeSeriesSplit)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Información detallada de los folds
print("\n" + "=" * 80)
print("INFORMACIÓN DETALLADA DE LOS FOLDS")
print("=" * 80)

df_folds_info = pd.DataFrame(folds_info)
print(df_folds_info.to_string(index=False))

# Evaluación final en todo el conjunto de datos
print("\n" + "=" * 80)
print("EVALUACIÓN FINAL EN TODO EL CONJUNTO DE DATOS")
print("=" * 80)

# Para evaluación final, usamos todo el dataset para entrenar
# y predecir todo (esto es solo para visualización)

# Entrenar modelos con todos los datos
print("\nEntrenando modelos con todos los datos para visualización final...")

# ARIMA final
try:
    arima_model_final = ARIMA(data[target], order=(5, 1, 0))
    arima_model_fit_final = arima_model_final.fit()
    arima_pred_final = arima_model_fit_final.fittedvalues
except Exception as e:
    print(f"Error ARIMA final: {e}")
    arima_pred_final = np.full(len(data), np.nan)

# Prophet final
try:
    prophet_data_final = data.reset_index().rename(columns={'fecha_hora': 'ds', target: 'y'})
    prophet_model_final = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False
    )
    prophet_model_final.fit(prophet_data_final)
    future_final = prophet_model_final.make_future_dataframe(periods=0, freq='5T')
    prophet_forecast_final = prophet_model_final.predict(future_final)['yhat']
except Exception as e:
    print(f"Error Prophet final: {e}")
    prophet_forecast_final = np.full(len(data), np.nan)

# Random Forest final
try:
    features_disponibles = [f for f in features_extendidas if f in data.columns]
    if len(features_disponibles) > 0:
        X_full = data[features_disponibles]
        y_full = data[target]
        rf_model_final = RandomForestRegressor(
            n_estimators=150,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model_final.fit(X_full, y_full)
        rf_pred_final = rf_model_final.predict(X_full)
    else:
        rf_pred_final = np.full(len(data), np.nan)
except Exception as e:
    print(f"Error Random Forest final: {e}")
    rf_pred_final = np.full(len(data), np.nan)

# Gráfico final comparativo
plt.figure(figsize=(15, 10))

# Gráfico de las series temporales reales y las predicciones
plt.subplot(3, 1, 1)
plt.plot(data.index, data[target], label='Real', color='black', alpha=0.7)
if not np.all(np.isnan(arima_pred_final)):
    plt.plot(data.index, arima_pred_final, label='ARIMA', color='blue', alpha=0.6)
plt.title('Predicción de ARIMA (entrenado con todos los datos)')
plt.ylabel('Generación (W)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(data.index, data[target], label='Real', color='black', alpha=0.7)
if not np.all(np.isnan(prophet_forecast_final)):
    plt.plot(data.index, prophet_forecast_final, label='Prophet', color='green', alpha=0.6)
plt.title('Predicción de Prophet (entrenado con todos los datos)')
plt.ylabel('Generación (W)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(data.index, data[target], label='Real', color='black', alpha=0.7)
if not np.all(np.isnan(rf_pred_final)):
    plt.plot(data.index, rf_pred_final, label='Random Forest', color='red', alpha=0.6)
plt.title('Predicción de Random Forest (entrenado con todos los datos)')
plt.ylabel('Generación (W)')
plt.xlabel('Fecha y Hora')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Guardar resultados
try:
    # Guardar métricas por fold
    df_metrics_folds = pd.DataFrame({
        'Fold': range(1, n_splits + 1),
        'ARIMA_RMSE': [m[0] for m in arima_metrics_folds],
        'ARIMA_MAE': [m[1] for m in arima_metrics_folds],
        'ARIMA_R2': [m[2] for m in arima_metrics_folds],
        'ARIMA_MAPE': [m[3] for m in arima_metrics_folds],
        'Prophet_RMSE': [m[0] for m in prophet_metrics_folds],
        'Prophet_MAE': [m[1] for m in prophet_metrics_folds],
        'Prophet_R2': [m[2] for m in prophet_metrics_folds],
        'Prophet_MAPE': [m[3] for m in prophet_metrics_folds],
        'RF_RMSE': [m[0] for m in rf_metrics_folds],
        'RF_MAE': [m[1] for m in rf_metrics_folds],
        'RF_R2': [m[2] for m in rf_metrics_folds],
        'RF_MAPE': [m[3] for m in rf_metrics_folds]
    })

    df_metrics_folds.to_csv('resultados_timescv_folds.csv', index=False)
    print("\nResultados por fold guardados en 'resultados_timescv_folds.csv'")

    # Guardar información de folds
    df_folds_info.to_csv('informacion_folds_timescv.csv', index=False)
    print("Información de folds guardada en 'informacion_folds_timescv.csv'")

except Exception as e:
    print(f"\nError guardando resultados: {e}")

print("\n" + "=" * 80)
print("TIME SERIES CROSS-VALIDATION COMPLETADA")
print("=" * 80)
print("\nArchivos principales generados:")
print("  - resultados_timescv_folds_homologados.csv")
print("  - informacion_folds_timescv_homologada.csv")
print("  - resultados_timescv_resumen_modelos.csv")
print("  - predicciones_timescv_fold_1.csv ... fold_5.csv")
print("  - importancia_rf_timescv_fold_1.csv ... fold_5.csv")
print("  - TimeSCV_Fold_1.png ... TimeSCV_Fold_5.png")
print("  - TimeSCV_Error_Acumulado_Fold_1.png ... Fold_5.png")
print("  - TimeSCV_RMSE_por_fold.png")
print("  - TimeSCV_MAE_por_fold.png")
print("  - TimeSCV_R2_por_fold.png")
print("  - TimeSCV_MAPE_por_fold.png")

print("\n" + "=" * 80)
print("TIMESERIESSPLIT HOMOLOGADO COMPLETADO")
print("=" * 80)