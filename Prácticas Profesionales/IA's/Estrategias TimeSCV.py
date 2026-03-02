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
file_path = r"/home/to-o/practicasProfeeionales/Prácticas Profesionales/JSON/Resultado_Homogenizado.xlsx"
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


# Función para calcular métricas
def calcular_métricas(y_true, y_pred):
    """Calcula métricas de evaluación"""
    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # MAPE con protección contra división por cero
        mask = y_true != 0
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan

        return rmse, mae, r2, mape
    except Exception as e:
        print(f"Error calculando métricas: {e}")
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
            yearly_seasonality=True,
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
    metricas_arima = calcular_métricas(test_data[target].values, arima_pred) if arima_status == "OK" else (np.nan,
                                                                                                           np.nan,
                                                                                                           np.nan,
                                                                                                           np.nan)
    metricas_prophet = calcular_métricas(test_data[target].values, prophet_forecast) if prophet_status == "OK" else (
        np.nan, np.nan, np.nan, np.nan)
    metricas_rf = calcular_métricas(test_data[target].values, rf_pred) if rf_status == "OK" else (np.nan, np.nan,
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
        plt.ylabel('Generación (kWh)')
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
        plt.ylabel('Error (kWh)')
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
        yearly_seasonality=True
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
plt.ylabel('Generación (kWh)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(data.index, data[target], label='Real', color='black', alpha=0.7)
if not np.all(np.isnan(prophet_forecast_final)):
    plt.plot(data.index, prophet_forecast_final, label='Prophet', color='green', alpha=0.6)
plt.title('Predicción de Prophet (entrenado con todos los datos)')
plt.ylabel('Generación (kWh)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(data.index, data[target], label='Real', color='black', alpha=0.7)
if not np.all(np.isnan(rf_pred_final)):
    plt.plot(data.index, rf_pred_final, label='Random Forest', color='red', alpha=0.6)
plt.title('Predicción de Random Forest (entrenado con todos los datos)')
plt.ylabel('Generación (kWh)')
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