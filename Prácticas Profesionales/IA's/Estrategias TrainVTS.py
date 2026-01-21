import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
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
    'eToday (kWh)', 'eTotal (kWh)', 'power (kW)',
    'wind_speed_100m', 'wind_speed_10m',
    'wind_direction_10m', "relative_humidity"
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


# Función para calcular métricas mejorada
def calcular_métricas(y_true, y_pred):
    """Calcula métricas de evaluación con manejo robusto"""
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


# PERIODOS MODIFICADOS: 1 mes, bimestral, trimestral, anual
periodos = {
    "1_mes": "30D",
    "2_meses_bimestral": "60D",
    "3_meses_trimestral": "90D",
    "1_año_anual": "365D"
}

resultados = {}
predicciones_detalladas = {}

print("\n" + "=" * 80)
print("EVALUACIÓN CON MÚLTIPLES PERIODOS DE PRUEBA")
print("=" * 80)

for nombre, periodo in periodos.items():
    print(f"\n{'=' * 60}")
    print(f"PROCESANDO: {nombre} ({periodo})")
    print(f"{'=' * 60}")

    # Extraer conjunto de prueba
    test_data = data.last(periodo)
    train_data = data.iloc[:-len(test_data)]

    # Verificar que tenemos datos suficientes
    if len(train_data) == 0:
        print(f"ERROR: No hay datos de entrenamiento para {nombre}")
        continue

    if len(test_data) == 0:
        print(f"ERROR: No hay datos de prueba para {nombre}")
        continue

    print(f"Período de entrenamiento: {len(train_data)} muestras")
    print(f"Período de prueba: {len(test_data)} muestras")
    print(f"Fecha inicio entrenamiento: {train_data.index.min()}")
    print(f"Fecha fin entrenamiento: {train_data.index.max()}")
    print(f"Fecha inicio prueba: {test_data.index.min()}")
    print(f"Fecha fin prueba: {test_data.index.max()}")

    # Series para modelos ARIMA y Prophet
    train_series = train_data[target]
    test_series = test_data[target]

    # 1. MODELO ARIMA
    print("\n[1/3] Entrenando ARIMA...")
    try:
        arima_model = ARIMA(train_series, order=(5, 1, 0))
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

        # Configurar Prophet con estacionalidades
        prophet_model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )

        prophet_model.fit(prophet_train)

        # Crear dataframe futuro
        future = prophet_model.make_future_dataframe(
            periods=len(test_data),
            freq='5T',
            include_history=False
        )

        forecast = prophet_model.predict(future)
        prophet_forecast = forecast['yhat'].values
        prophet_status = "OK"

        # Extraer componentes estacionales si están disponibles
        if 'daily' in forecast.columns:
            estacionalidad_diaria = forecast['daily'].values
        else:
            estacionalidad_diaria = None

    except Exception as e:
        print(f"   Error Prophet: {e}")
        prophet_forecast = np.full(len(test_data), np.nan)
        prophet_status = f"Error: {str(e)[:50]}..."
        estacionalidad_diaria = None

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

            # Random Forest con parámetros optimizados para series temporales
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

            print(f"   Características más importantes:")
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

    # Calcular métricas para cada modelo
    metricas_arima = calcular_métricas(test_series.values, arima_pred) if arima_status == "OK" else (np.nan, np.nan,
                                                                                                     np.nan, np.nan)
    metricas_prophet = calcular_métricas(test_series.values, prophet_forecast) if prophet_status == "OK" else (np.nan,
                                                                                                               np.nan,
                                                                                                               np.nan,
                                                                                                               np.nan)
    metricas_rf = calcular_métricas(test_series.values, rf_pred) if rf_status == "OK" else (np.nan, np.nan, np.nan,
                                                                                            np.nan)

    # Guardar resultados
    resultados[nombre] = {
        "ARIMA": metricas_arima,
        "Prophet": metricas_prophet,
        "RandomForest": metricas_rf
    }

    # Guardar predicciones detalladas
    predicciones_detalladas[nombre] = {
        'real': test_series.values,
        'ARIMA': arima_pred,
        'Prophet': prophet_forecast,
        'RandomForest': rf_pred,
        'fechas': test_data.index,
        'estacionalidad': estacionalidad_diaria
    }

    # Gráficos comparativos
    print("\nGenerando gráficos...")

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    modelos_info = [
        ('ARIMA', arima_pred, 'blue', arima_status),
        ('Prophet', prophet_forecast, 'green', prophet_status),
        ('RandomForest', rf_pred, 'red', rf_status)
    ]

    plot_idx = 0
    for modelo_nombre, prediccion, color, status in modelos_info:
        if status == "OK" and not np.all(np.isnan(prediccion)):
            # Gráfico 1: Serie temporal comparativa
            ax1 = axes[plot_idx]
            ax1.plot(test_data.index, test_series.values, label='Real', color='black', linewidth=2, alpha=0.7)
            ax1.plot(test_data.index, prediccion, label=f'{modelo_nombre} (Pred)', color=color, linestyle='--',
                     linewidth=1.5)
            ax1.set_title(f'{modelo_nombre} - {nombre}\nRMSE: {metricas_arima[0]:.3f}' if modelo_nombre == 'ARIMA' else
                          f'{modelo_nombre} - {nombre}\nRMSE: {metricas_prophet[0]:.3f}' if modelo_nombre == 'Prophet' else
                          f'{modelo_nombre} - {nombre}\nRMSE: {metricas_rf[0]:.3f}')
            ax1.set_ylabel('Generación (kWh)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Gráfico 2: Error acumulado
            ax2 = axes[plot_idx + 1]
            error = test_series.values - prediccion
            error_acumulado = np.cumsum(error)
            ax2.plot(test_data.index, error_acumulado, color=color, linewidth=2)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.fill_between(test_data.index, error_acumulado, 0,
                             where=(error_acumulado >= 0), color='red', alpha=0.2, label='Error +')
            ax2.fill_between(test_data.index, error_acumulado, 0,
                             where=(error_acumulado < 0), color='blue', alpha=0.2, label='Error -')
            ax2.set_title(f'Error Acumulado - {modelo_nombre}')
            ax2.set_ylabel('Error Acumulado (kWh)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plot_idx += 2

    # Ajustar diseño y mostrar
    plt.tight_layout()
    plt.show()

    # Gráfico adicional: Comparación de los 3 modelos juntos
    if any(status == "OK" for _, _, _, status in modelos_info):
        plt.figure(figsize=(14, 6))

        plt.plot(test_data.index, test_series.values, label='Real', color='black', linewidth=3, alpha=0.6)

        for modelo_nombre, prediccion, color, status in modelos_info:
            if status == "OK" and not np.all(np.isnan(prediccion)):
                plt.plot(test_data.index, prediccion, label=modelo_nombre, linewidth=1.5, alpha=0.8)

        plt.title(f'Comparación de Modelos - {nombre} ({periodo})')
        plt.ylabel('Generación (kWh)')
        plt.xlabel('Fecha y Hora')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Añadir métricas en texto
        metricas_texto = ""
        for modelo_nombre, _, _, status in modelos_info:
            if status == "OK":
                if modelo_nombre == 'ARIMA':
                    metricas_texto += f"ARIMA RMSE: {metricas_arima[0]:.3f}\n"
                elif modelo_nombre == 'Prophet':
                    metricas_texto += f"Prophet RMSE: {metricas_prophet[0]:.3f}\n"
                elif modelo_nombre == 'RandomForest':
                    metricas_texto += f"RF RMSE: {metricas_rf[0]:.3f}\n"

        plt.figtext(0.02, 0.02, metricas_texto, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()
        plt.show()

# Mostrar resultados finales
print("\n" + "=" * 80)
print("RESULTADOS FINALES")
print("=" * 80)

for periodo, metricas in resultados.items():
    print(f"\n{periodo.upper()}:")
    print("-" * 60)

    for modelo_nombre in ['ARIMA', 'Prophet', 'RandomForest']:
        vals = metricas[modelo_nombre]
        if not np.isnan(vals[0]):
            print(
                f"{modelo_nombre:15s} | RMSE: {vals[0]:6.4f} | MAE: {vals[1]:6.4f} | R²: {vals[2]:6.4f} | MAPE: {vals[3]:6.2f}%")
        else:
            print(f"{modelo_nombre:15s} | Métricas no disponibles")

# Análisis estadístico de resultados
print("\n" + "=" * 80)
print("ANÁLISIS ESTADÍSTICO")
print("=" * 80)

# Crear DataFrame para análisis
filas_analisis = []
for periodo, metricas in resultados.items():
    for modelo, vals in metricas.items():
        if not np.isnan(vals[0]):
            filas_analisis.append({
                'Periodo': periodo,
                'Modelo': modelo,
                'RMSE': vals[0],
                'MAE': vals[1],
                'R2': vals[2],
                'MAPE': vals[3]
            })

if filas_analisis:
    df_analisis = pd.DataFrame(filas_analisis)

    # Resumen por modelo
    print("\nResumen por Modelo (promedio entre todos los periodos):")
    resumen_modelo = df_analisis.groupby('Modelo').agg({
        'RMSE': ['mean', 'std', 'min', 'max'],
        'MAE': ['mean', 'std'],
        'R2': ['mean', 'std'],
        'MAPE': ['mean', 'std']
    }).round(4)

    print(resumen_modelo)

    # Resumen por periodo
    print("\nResumen por Periodo (promedio entre todos los modelos):")
    resumen_periodo = df_analisis.groupby('Periodo').agg({
        'RMSE': ['mean', 'std'],
        'MAE': ['mean', 'std'],
        'R2': ['mean', 'std'],
        'MAPE': ['mean', 'std']
    }).round(4)

    print(resumen_periodo)

    # Gráfico de rendimiento por periodo
    plt.figure(figsize=(12, 8))

    metricas_grafico = ['RMSE', 'MAE', 'R2']
    periodos_ordenados = ['1_mes', '2_meses_bimestral', '3_meses_trimestral', '1_año_anual']

    for i, metrica in enumerate(metricas_grafico, 1):
        plt.subplot(2, 2, i)

        # Filtrar y pivotar datos
        df_metrica = df_analisis[['Periodo', 'Modelo', metrica]].copy()
        df_metrica['Periodo'] = pd.Categorical(df_metrica['Periodo'], categories=periodos_ordenados, ordered=True)
        df_pivot = df_metrica.pivot(index='Periodo', columns='Modelo', values=metrica)

        df_pivot.plot(kind='bar', ax=plt.gca(), width=0.8)
        plt.title(f'{metrica} por Periodo y Modelo')
        plt.ylabel(metrica)
        plt.xlabel('Periodo de Prueba')
        plt.xticks(rotation=45)
        plt.legend(title='Modelo')
        plt.grid(True, alpha=0.3, axis='y')

    # Gráfico de MAPE
    plt.subplot(2, 2, 4)
    df_mape = df_analisis[['Periodo', 'Modelo', 'MAPE']].copy()
    df_mape['Periodo'] = pd.Categorical(df_mape['Periodo'], categories=periodos_ordenados, ordered=True)
    df_mape_pivot = df_mape.pivot(index='Periodo', columns='Modelo', values='MAPE')

    df_mape_pivot.plot(kind='bar', ax=plt.gca(), width=0.8)
    plt.title('MAPE por Periodo y Modelo')
    plt.ylabel('MAPE (%)')
    plt.xlabel('Periodo de Prueba')
    plt.xticks(rotation=45)
    plt.legend(title='Modelo')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # Guardar resultados
    try:
        df_analisis.to_csv('resultados_trainvts_periodos_largos.csv', index=False)
        print(f"\nResultados detallados guardados en 'resultados_trainvts_periodos_largos.csv'")
    except Exception as e:
        print(f"\nError guardando resultados: {e}")

print("\n" + "=" * 80)
print("EVALUACIÓN COMPLETADA")
print("=" * 80)