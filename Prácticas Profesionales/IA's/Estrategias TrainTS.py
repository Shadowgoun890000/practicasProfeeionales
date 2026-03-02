import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
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

# Añadir características temporales
data['hora'] = data.index.hour
data['dia_semana'] = data.index.dayofweek
data['es_fin_semana'] = data['dia_semana'].isin([5, 6]).astype(int)
data['mes'] = data.index.month
data['estacion'] = (data['mes'] % 12 + 3) // 3  # 1: Primavera, 2: Verano, 3: Otoño, 4: Invierno

# Variables objetivo y características
target = 'valor (kWh)'
features_base = [
    # Variables del inversor
    'eToday (kWh)', 'eTotal (kWh)',
    # Variables meteorológicas/energéticas
    'air_temp', 'relative_humidity',
    'power (kW)',
    'wind_speed_10m', 'wind_direction_10m',
    'ghi', 'dni', 'gti'
]
# Filtrar características disponibles
features = [col for col in features_base if col in data.columns]
features_extendidas = features + ['hora', 'dia_semana', 'es_fin_semana', 'mes', 'estacion']
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


# PERIODOS: 1 mes, bimestral, trimestral, anual
periodos = {
    "1_mes": "30D",
    "2_meses_bimestral": "60D",
    "3_meses_trimestral": "90D",
    "1_año_anual": "365D"
}

print("\n" + "=" * 80)
print("EVALUACIÓN CON TRAIN-TEST SPLIT POR PERIODOS")
print("=" * 80)

# Análisis Exploratorio de Datos (EDA)
print("\n1. ANÁLISIS EXPLORATORIO DE DATOS")
print("-" * 40)

print("\nEstadísticas descriptivas de la variable objetivo:")
print(data[target].describe())

print(f"\nCorrelaciones con {target}:")
correlaciones = data[features_extendidas + [target]].corr()[target].sort_values(ascending=False)
for var, corr in correlaciones.items():
    print(f"  {var:25s}: {corr:.4f}")

# Gráficos EDA
plt.figure(figsize=(15, 10))

# 1. Distribución de la generación
plt.subplot(2, 2, 1)
sns.histplot(data[target], kde=True, bins=50)
plt.title(f'Distribución de {target}')
plt.xlabel('Generación (kWh)')
plt.ylabel('Frecuencia')

# 2. Serie temporal completa
plt.subplot(2, 2, 2)
plt.plot(data.index, data[target], linewidth=0.5, alpha=0.7)
plt.title('Serie Temporal Completa')
plt.xlabel('Fecha')
plt.ylabel('Generación (kWh)')
plt.grid(True, alpha=0.3)

# 3. Boxplot por hora del día
plt.subplot(2, 2, 3)
sns.boxplot(x=data['hora'], y=data[target])
plt.title('Generación por Hora del Día')
plt.xlabel('Hora')
plt.ylabel('Generación (kWh)')
plt.grid(True, alpha=0.3)

# 4. Boxplot por día de la semana
plt.subplot(2, 2, 4)
sns.boxplot(x=data['dia_semana'], y=data[target])
plt.title('Generación por Día de la Semana')
plt.xlabel('Día de la Semana (0=Lunes, 6=Domingo)')
plt.ylabel('Generación (kWh)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Matriz de correlación
plt.figure(figsize=(12, 10))
correlation_matrix = data[features_extendidas[:10] + [target]].corr()  # Mostrar solo las primeras 10
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', center=0)
plt.title('Matriz de Correlación (primeras 10 variables)')
plt.tight_layout()
plt.show()

# Evaluación para cada periodo
resultados = {}

for nombre, periodo in periodos.items():
    print(f"\n{'=' * 60}")
    print(f"EVALUACIÓN PARA: {nombre} ({periodo})")
    print(f"{'=' * 60}")

    # Extraer conjunto de prueba (último periodo)
    test_data = data.last(periodo)
    train_data = data.iloc[:-len(test_data)]

    # Verificar que tenemos datos suficientes
    if len(train_data) == 0:
        print(f"ERROR: No hay datos de entrenamiento para {nombre}")
        continue

    if len(test_data) == 0:
        print(f"ERROR: No hay datos de prueba para {nombre}")
        continue

    print(f"\nDatos de entrenamiento: {len(train_data)} muestras")
    print(f"Datos de prueba: {len(test_data)} muestras")
    print(
        f"Proporción entrenamiento/prueba: {len(train_data) / len(data) * 100:.1f}% / {len(test_data) / len(data) * 100:.1f}%")

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

            print(f"   Top 5 características importantes:")
            for idx, row in importancia.head(5).iterrows():
                print(f"     {row['Característica']:25s}: {row['Importancia']:.4f}")

            rf_status = "OK"
        else:
            print("   No hay características disponibles")
            rf_pred = np.full(len(test_data), np.nan)
            rf_status = "Sin características"

    except Exception as e:
        print(f"   Error Random Forest: {e}")
        rf_pred = np.full(len(test_data), np.nan)
        rf_status = f"Error: {str(e)[:50]}..."

    # Calcular métricas
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

    # Imprimir métricas
    print(f"\nMÉTRICAS PARA {nombre}:")
    print("-" * 40)

    for modelo, metricas in [("ARIMA", metricas_arima), ("Prophet", metricas_prophet), ("Random Forest", metricas_rf)]:
        if not np.isnan(metricas[0]):
            print(
                f"{modelo:15s} - RMSE: {metricas[0]:.4f} | MAE: {metricas[1]:.4f} | R²: {metricas[2]:.4f} | MAPE: {metricas[3]:.2f}%")

    # Gráfico comparativo para este periodo
    if arima_status == "OK" or prophet_status == "OK" or rf_status == "OK":
        plt.figure(figsize=(14, 10))

        # 1. Comparación de modelos
        plt.subplot(3, 1, 1)
        plt.plot(test_data.index, test_series.values, label='Real', color='black', linewidth=2, alpha=0.8)

        if arima_status == "OK":
            plt.plot(test_data.index, arima_pred, label='ARIMA', color='blue', linestyle='--', linewidth=1.5, alpha=0.7)

        if prophet_status == "OK":
            plt.plot(test_data.index, prophet_forecast, label='Prophet', color='green', linestyle='--', linewidth=1.5,
                     alpha=0.7)

        if rf_status == "OK":
            plt.plot(test_data.index, rf_pred, label='Random Forest', color='red', linestyle='--', linewidth=1.5,
                     alpha=0.7)

        plt.title(f'Comparación de Modelos - {nombre} ({periodo})')
        plt.ylabel('Generación (kWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Gráfico de errores
        plt.subplot(3, 1, 2)

        modelos_graficos = []
        if arima_status == "OK":
            modelos_graficos.append(('ARIMA', arima_pred, 'blue'))
        if prophet_status == "OK":
            modelos_graficos.append(('Prophet', prophet_forecast, 'green'))
        if rf_status == "OK":
            modelos_graficos.append(('Random Forest', rf_pred, 'red'))

        for modelo_nombre, prediccion, color in modelos_graficos:
            error = test_series.values - prediccion
            plt.plot(test_data.index, error, label=modelo_nombre, color=color, alpha=0.6)

        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Errores de Predicción')
        plt.ylabel('Error (kWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. Gráfico de error acumulado
        plt.subplot(3, 1, 3)

        for modelo_nombre, prediccion, color in modelos_graficos:
            error = test_series.values - prediccion
            error_acumulado = np.cumsum(error)
            plt.plot(test_data.index, error_acumulado, label=f'{modelo_nombre} (Acumulado)', color=color, linewidth=2)

        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Error Acumulado de Predicción')
        plt.ylabel('Error Acumulado (kWh)')
        plt.xlabel('Fecha y Hora')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# Resumen final de resultados
print("\n" + "=" * 80)
print("RESUMEN FINAL DE RESULTADOS")
print("=" * 80)

# Crear DataFrame con resultados
filas_resumen = []
for periodo, metricas_modelos in resultados.items():
    for modelo, vals in metricas_modelos.items():
        if not np.isnan(vals[0]):
            filas_resumen.append({
                'Periodo': periodo,
                'Modelo': modelo,
                'RMSE': vals[0],
                'MAE': vals[1],
                'R2': vals[2],
                'MAPE': vals[3]
            })

if filas_resumen:
    df_resumen = pd.DataFrame(filas_resumen)

    print("\nResultados por Modelo y Periodo:")
    print("-" * 60)

    # Pivot table para mejor visualización
    pivot_rmse = df_resumen.pivot(index='Periodo', columns='Modelo', values='RMSE')
    pivot_r2 = df_resumen.pivot(index='Periodo', columns='Modelo', values='R2')

    print("\nRMSE por Periodo y Modelo:")
    print(pivot_rmse.round(4))

    print("\nR² por Periodo y Modelo:")
    print(pivot_r2.round(4))

    # Análisis estadístico
    print("\nEstadísticas por Modelo (promedio entre todos los periodos):")
    stats_modelo = df_resumen.groupby('Modelo').agg({
        'RMSE': ['mean', 'std', 'min', 'max'],
        'MAE': ['mean', 'std'],
        'R2': ['mean', 'std'],
        'MAPE': ['mean', 'std']
    }).round(4)

    print(stats_modelo)

    # Gráfico de comparación de modelos por periodo
    periodos_ordenados = ['1_mes', '2_meses_bimestral', '3_meses_trimestral', '1_año_anual']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metrica in enumerate(['RMSE', 'MAE', 'R2', 'MAPE']):
        ax = axes[idx]

        # Preparar datos para esta métrica
        df_metrica = df_resumen[['Periodo', 'Modelo', metrica]].copy()
        df_metrica['Periodo'] = pd.Categorical(df_metrica['Periodo'], categories=periodos_ordenados, ordered=True)
        df_pivot = df_metrica.pivot(index='Periodo', columns='Modelo', values=metrica)

        # Graficar
        df_pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'{metrica} por Periodo y Modelo')
        ax.set_xlabel('Periodo de Prueba')
        ax.set_ylabel('RMSE' if metrica == 'RMSE' else
                      'MAE' if metrica == 'MAE' else
                      'R²' if metrica == 'R2' else 'MAPE (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Modelo')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # Gráfico de tendencia de error vs longitud del periodo
    print("\nAnálisis de Tendencia: Error vs Longitud del Periodo")
    print("-" * 60)

    # Mapear nombres de periodo a días
    dias_por_periodo = {
        '1_mes': 30,
        '2_meses_bimestral': 60,
        '3_meses_trimestral': 90,
        '1_año_anual': 365
    }

    df_resumen['Dias'] = df_resumen['Periodo'].map(dias_por_periodo)

    plt.figure(figsize=(12, 8))

    modelos_unicos = df_resumen['Modelo'].unique()
    colores = {'ARIMA': 'blue', 'Prophet': 'green', 'RandomForest': 'red'}

    for i, metrica in enumerate(['RMSE', 'MAE', 'R2', 'MAPE'], 1):
        plt.subplot(2, 2, i)

        for modelo in modelos_unicos:
            df_modelo = df_resumen[df_resumen['Modelo'] == modelo]
            if not df_modelo.empty:
                plt.plot(df_modelo['Dias'], df_modelo[metrica],
                         label=modelo, color=colores.get(modelo, 'black'),
                         marker='o', linewidth=2, markersize=8)

        plt.xlabel('Días del Periodo de Prueba')
        plt.ylabel(metrica)
        plt.title(f'Tendencia de {metrica} vs Longitud del Periodo')
        plt.grid(True, alpha=0.3)

        if i == 1:  # Solo poner leyenda en el primer gráfico
            plt.legend()

    plt.tight_layout()
    plt.show()

    # Guardar resultados
    try:
        df_resumen.to_csv('resultados_traints_periodos_largos.csv', index=False)
        print(f"\nResultados detallados guardados en 'resultados_traints_periodos_largos.csv'")

        # Guardar también en Excel con formato
        with pd.ExcelWriter('IA', engine='openpyxl') as writer:
            df_resumen.to_excel(writer, sheet_name='Resultados', index=False)

            # Crear hoja de resumen
            resumen_excel = df_resumen.groupby(['Periodo', 'Modelo']).mean().reset_index()
            resumen_excel.to_excel(writer, sheet_name='Resumen', index=False)

        print("Resultados guardados en 'resultados_traints_periodos_largos.xlsx'")

    except Exception as e:
        print(f"\nError guardando resultados: {e}")

print("\n" + "=" * 80)
print("EVALUACIÓN TRAIN-TEST SPLIT COMPLETADA")
print("=" * 80)