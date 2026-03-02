import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Cargar los datos
file_path = r"/home/to-o/practicasProfeeionales/Prácticas Profesionales/JSON/Resultado_Homogenizado.xlsx"
data = pd.read_excel(file_path)

# Preprocesamiento de fechas
data['fecha_hora'] = pd.to_datetime(data['fecha_hora'])
data.set_index('fecha_hora', inplace=True)

# Verificar que tenemos suficientes datos para todos los periodos
total_dias = (data.index.max() - data.index.min()).days
print(f"Total de días en dataset: {total_dias}")
print(f"Rango de fechas: {data.index.min()} a {data.index.max()}")

# Agregar variables temporales
data['hora'] = data.index.hour
data['dia_semana'] = data.index.dayofweek
data['es_fin_semana'] = data['dia_semana'].isin([5, 6]).astype(int)
data['mes'] = data.index.month
data['estacion'] = (data['mes'] % 12 + 3) // 3  # 1: Primavera, 2: Verano, 3: Otoño, 4: Invierno

# Definir variables
objetivo = 'valor (kWh)'
features = [
    # Variables del inversor
    'eToday (kWh)', 'eTotal (kWh)',

    # Variables meteorológicas/energéticas
    'air_temp', 'relative_humidity',
    'power (kW)',
    'wind_speed_10m', 'wind_direction_10m',
    'ghi', 'dni', 'gti',

    # Variables temporales
    'hora', 'dia_semana', 'es_fin_semana', 'mes', 'estacion'
]
features = [col for col in features if col in data.columns]


# Funciones de métricas
def calcular_métricas(y_real, y_pred):
    """Calcula métricas de evaluación con manejo de errores"""
    try:
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))
        mae = mean_absolute_error(y_real, y_pred)
        r2 = r2_score(y_real, y_pred)
        # MAPE (Mean Absolute Percentage Error) con protección contra división por cero
        mape = np.mean(np.abs((y_real - y_pred) / np.where(y_real == 0, 1, y_real))) * 100
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

# Verificar que tenemos datos suficientes para cada periodo
for nombre, periodo in periodos.items():
    dias = int(periodo[:-1])
    print(f"{nombre}: Requiere {dias} días de datos")

resultados = {}
predicciones_completas = {}

for nombre, periodo in periodos.items():
    print(f"\n{'=' * 50}")
    print(f"Evaluando periodo: {nombre} ({periodo})")
    print(f"{'=' * 50}")

    # Extraer datos de prueba
    test_data = data.last(periodo)
    train_data = data.iloc[:-len(test_data)]

    print(f"Entrenamiento: {train_data.index.min().date()} a {train_data.index.max().date()}")
    print(f"Prueba: {test_data.index.min().date()} a {test_data.index.max().date()}")
    print(f"Muestras entrenamiento: {len(train_data)}")
    print(f"Muestras prueba: {len(test_data)}")

    if len(train_data) == 0:
        print(f"ERROR: No hay datos de entrenamiento para {nombre}. Saltando...")
        continue

    if len(test_data) == 0:
        print(f"ERROR: No hay datos de prueba para {nombre}. Saltando...")
        continue

    # ARIMA
    print("\n1. Entrenando modelo ARIMA...")
    try:
        arima_model = ARIMA(train_data[objetivo], order=(5, 1, 0))
        arima_model_fit = arima_model.fit()
        arima_pred = arima_model_fit.forecast(steps=len(test_data))
        arima_success = True
    except Exception as e:
        print(f"   Error en ARIMA: {e}")
        arima_pred = np.full(len(test_data), np.nan)
        arima_success = False

    # Prophet con regresores
    print("2. Entrenando modelo Prophet...")
    try:
        prophet_data = train_data.reset_index().rename(columns={'fecha_hora': 'ds', objetivo: 'y'})

        # Seleccionar regresores disponibles
        regresores_disponibles = ['air_temp', 'relative_humidity', 'power (kW)', 'wind_speed_10m', 'wind_direction_10m', 'ghi', 'dni', 'gti', 'eToday (kWh)', 'eTotal (kWh)']
        regresores_usar = [r for r in regresores_disponibles if r in train_data.columns]

        prophet_model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='multiplicative'
        )

        # Añadir regresores
        for reg in regresores_usar:
            prophet_data[reg] = train_data[reg].values
            prophet_model.add_regressor(reg)
            print(f"   Añadido regresor: {reg}")

        prophet_model.fit(prophet_data)

        # Crear dataframe futuro con regresores
        future = prophet_model.make_future_dataframe(periods=len(test_data), freq='5T', include_history=False)

        # Añadir regresores al futuro (usamos valores de test_data)
        for reg in regresores_usar:
            if reg in test_data.columns:
                # Asegurarnos de que coincida la longitud
                future[reg] = test_data[reg].reset_index(drop=True).values[:len(future)]

        forecast = prophet_model.predict(future)
        prophet_forecast = forecast['yhat'].values
        prophet_success = True
    except Exception as e:
        print(f"   Error en Prophet: {e}")
        prophet_forecast = np.full(len(test_data), np.nan)
        prophet_success = False

    # Random Forest
    print("3. Entrenando modelo Random Forest...")
    try:
        # Asegurarnos de que todas las features están disponibles
        features_disponibles = [f for f in features if f in train_data.columns and f in test_data.columns]

        if len(features_disponibles) > 0:
            X_train = train_data[features_disponibles]
            X_test = test_data[features_disponibles]
            y_train = train_data[objetivo]
            y_test = test_data[objetivo]

            rf_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                max_depth=15,
                min_samples_split=5
            )
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)

            # Calcular importancia de características
            importancia = pd.DataFrame({
                'feature': features_disponibles,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"   Top 5 características importantes:")
            for idx, row in importancia.head().iterrows():
                print(f"      {row['feature']}: {row['importance']:.4f}")

            rf_success = True
        else:
            print("   No hay características disponibles para Random Forest")
            rf_pred = np.full(len(test_data), np.nan)
            rf_success = False
    except Exception as e:
        print(f"   Error en Random Forest: {e}")
        rf_pred = np.full(len(test_data), np.nan)
        rf_success = False

    # Calcular métricas solo si las predicciones son válidas
    metricas_arima = calcular_métricas(test_data[objetivo], arima_pred) if arima_success else (np.nan, np.nan, np.nan,
                                                                                               np.nan)
    metricas_prophet = calcular_métricas(test_data[objetivo], prophet_forecast) if prophet_success else (np.nan, np.nan,
                                                                                                         np.nan, np.nan)
    metricas_rf = calcular_métricas(test_data[objetivo], rf_pred) if rf_success else (np.nan, np.nan, np.nan, np.nan)

    # Guardar métricas
    resultados[nombre] = {
        "ARIMA": metricas_arima,
        "Prophet": metricas_prophet,
        "RandomForest": metricas_rf
    }

    # Guardar predicciones para análisis posterior
    predicciones_completas[nombre] = {
        'test_real': test_data[objetivo].values,
        'ARIMA': arima_pred,
        'Prophet': prophet_forecast,
        'RandomForest': rf_pred,
        'fechas': test_data.index
    }

    # Graficar resultados
    modelos = {'ARIMA': arima_pred, 'Prophet': prophet_forecast, 'RandomForest': rf_pred}

    for modelo, prediccion in modelos.items():
        if not np.all(np.isnan(prediccion)):
            plt.figure(figsize=(14, 6))

            # Gráfico principal
            plt.subplot(2, 1, 1)
            plt.plot(test_data.index, test_data[objetivo], label='Real', color='black', linewidth=2, alpha=0.7)
            plt.plot(test_data.index, prediccion, label=modelo, linestyle='--', linewidth=1.5)
            plt.title(f'{modelo} - Predicción para {nombre} (Periodo: {periodo})')
            plt.ylabel('Generación (kWh)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Gráfico de errores
            plt.subplot(2, 1, 2)
            error = test_data[objetivo].values - prediccion
            plt.plot(test_data.index, error, label='Error', color='red', linewidth=1)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.fill_between(test_data.index, error, 0, where=(error >= 0), color='red', alpha=0.2,
                             label='Sobreestimación')
            plt.fill_between(test_data.index, error, 0, where=(error < 0), color='blue', alpha=0.2,
                             label='Subestimación')
            plt.title(f'Error de Predicción - {modelo}')
            plt.ylabel('Error (kWh)')
            plt.xlabel('Fecha y Hora')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
        else:
            print(f"   No se pudo graficar {modelo}: predicciones no disponibles")

# Mostrar resultados
print("\n" + "=" * 80)
print("RESULTADOS POR PERIODO")
print("=" * 80)

for periodo, metricas in resultados.items():
    print(f"\n{periodo}:")
    print("-" * 50)

    for modelo, valores in metricas.items():
        if not np.isnan(valores[0]):
            print(
                f"{modelo:15s} - RMSE: {valores[0]:.4f} | MAE: {valores[1]:.4f} | R²: {valores[2]:.4f} | MAPE: {valores[3]:.2f}%")
        else:
            print(f"{modelo:15s} - Métricas no disponibles")

# Resultados promedio (solo para modelos con métricas válidas)
print("\n" + "=" * 80)
print("RESULTADOS PROMEDIO POR MODELO (excluyendo NaN)")
print("=" * 80)

promedios = {}
for modelo in ['ARIMA', 'Prophet', 'RandomForest']:
    # Recoger solo métricas válidas
    modelo_vals = []
    for periodo in resultados:
        vals = resultados[periodo][modelo]
        if not np.isnan(vals[0]):  # Si RMSE no es NaN
            modelo_vals.append(vals)

    if modelo_vals:
        promedio = np.nanmean(modelo_vals, axis=0)
        promedios[modelo] = promedio

        # Contar cuántos periodos tuvieron métricas válidas
        periodos_validos = len(modelo_vals)
        print(f"\n{modelo}:")
        print(f"  Periodos con métricas válidas: {periodos_validos}/{len(periodos)}")
        print(f"  RMSE promedio:  {promedio[0]:.4f}")
        print(f"  MAE promedio:   {promedio[1]:.4f}")
        print(f"  R² promedio:    {promedio[2]:.4f}")
        print(f"  MAPE promedio:  {promedio[3]:.2f}%")

# Análisis de tendencia del error por periodo
print("\n" + "=" * 80)
print("ANÁLISIS DE TENDENCIA: ERROR VS LONGITUD DEL PERIODO")
print("=" * 80)

# Crear dataframe para análisis
df_tendencia = []
for periodo, metricas in resultados.items():
    for modelo in ['ARIMA', 'Prophet', 'RandomForest']:
        vals = metricas[modelo]
        if not np.isnan(vals[0]):
            # Extraer días del periodo
            dias = int(periodos[periodo][:-1])
            df_tendencia.append({
                'Periodo': periodo,
                'Dias': dias,
                'Modelo': modelo,
                'RMSE': vals[0],
                'MAE': vals[1],
                'R2': vals[2],
                'MAPE': vals[3]
            })

if df_tendencia:
    df_tendencia = pd.DataFrame(df_tendencia)

    # Graficar error vs días
    plt.figure(figsize=(12, 8))

    modelos_unicos = df_tendencia['Modelo'].unique()
    colores = {'ARIMA': 'blue', 'Prophet': 'green', 'RandomForest': 'red'}
    marcadores = {'ARIMA': 'o', 'Prophet': 's', 'RandomForest': '^'}

    for i, metrica in enumerate(['RMSE', 'MAE', 'R2', 'MAPE'], 1):
        plt.subplot(2, 2, i)

        for modelo in modelos_unicos:
            df_modelo = df_tendencia[df_tendencia['Modelo'] == modelo]
            if not df_modelo.empty:
                plt.plot(df_modelo['Dias'], df_modelo[metrica],
                         label=modelo, color=colores[modelo],
                         marker=marcadores[modelo], linewidth=2, markersize=8)

        plt.xlabel('Días del Periodo de Prueba')
        plt.ylabel(metrica)
        plt.title(f'{metrica} vs Longitud del Periodo')
        plt.grid(True, alpha=0.3)

        if i == 1:  # Solo poner leyenda en el primer gráfico
            plt.legend()

    plt.tight_layout()
    plt.show()

    # Mostrar tabla resumen
    print("\nResumen por modelo y longitud de periodo:")
    print(df_tendencia.pivot_table(index=['Modelo', 'Dias'], values=['RMSE', 'MAE', 'R2', 'MAPE']).round(4))
else:
    print("No hay suficientes datos para análisis de tendencia")

# Guardar resultados en CSV
try:
    resultados_df = pd.DataFrame.from_dict({(i, j): resultados[i][j]
                                            for i in resultados.keys()
                                            for j in resultados[i].keys()},
                                           orient='index')
    resultados_df.index = pd.MultiIndex.from_tuples(resultados_df.index)
    resultados_df.columns = ['RMSE', 'MAE', 'R2', 'MAPE']

    resultados_df.to_csv('resultados_kfold_periodos_largos.csv')
    print(f"\nResultados guardados en 'resultados_kfold_periodos_largos.csv'")
except Exception as e:
    print(f"\nError guardando resultados: {e}")

print("\n¡Evaluación completada!")