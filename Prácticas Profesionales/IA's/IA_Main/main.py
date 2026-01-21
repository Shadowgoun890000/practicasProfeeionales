from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics, generate_cutoffs

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Carga y preprocesamiento de datos ---
def cargar_datos(ruta_archivo, columna_fecha='fecha_hora', target='valor (kWh)', features=None):
    data = pd.read_excel(ruta_archivo)
    data[columna_fecha] = pd.to_datetime(data[columna_fecha])
    data.set_index(columna_fecha, inplace=True)

    # Filtrar features que estén en el dataset
    if features is not None:
        features = [f for f in features if f in data.columns]
    else:
        features = []

    return data, target, features

# --- Preparar dataframe para Prophet con regresores ---
def preparar_dataframe_prophet(data, target, features):
    df = data.reset_index().rename(columns={'fecha_hora': 'ds', target: 'y'})
    for f in features:
        df[f] = data[f].values
    return df

# --- Entrenamiento del modelo Prophet con regresores y estacionalidades ---
def entrenar_prophet(df, features, changepoint_prior_scale=0.1, seasonality_prior_scale=10.0):
    modelo = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale
    )
    # Añadir regresores externos
    for f in features:
        modelo.add_regressor(f)

    modelo.fit(df)
    return modelo

# --- Predicción con modelo Prophet ---
def predecir_prophet(modelo, test_df, features):
    future = test_df[['ds'] + features].copy()
    forecast = modelo.predict(future)
    return forecast['yhat'].values

# --- Calcular métricas de error ---
def calcular_metricas(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# --- Validación cruzada con Prophet ---
def validacion_cruzada_prophet(modelo, df, initial='40 days', period='20 days', horizon='10 days'):

    try:
       
       generate_cutoffs(df, pd.Timedelta(horizon), pd.Timedelta(initial), pd.Timedelta(period))
       df_cv = cross_validation(modelo, initial=initial, period=period, horizon=horizon)
       df_p = performance_metrics(df_cv)
       return df_cv, df_p
    
    except ValueError as e:
        print(f"X Validación cruzada fallida: {e}")
        return None, None
    
# --- Visualización comparativa de entrenamiento y prueba ---
def graficar_comparacion(train_df, test_df, features):
    fig, axs = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
    axs = axs.ravel()

    variables = ['y'] + features

    for i, var in enumerate(variables):
        axs[i].plot(train_df['ds'], train_df[var], label='Entrenamiento', color='blue', alpha=0.7)
        axs[i].plot(test_df['ds'], test_df[var], label='Prueba', color='orange', alpha=0.7)
        axs[i].set_ylabel(var)
        axs[i].legend()

    axs[-1].set_xlabel('Fecha')
    fig.suptitle('Comparación entre Entrenamiento y Prueba', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

# --- Función principal para ejecutar todo ---
def main():
    ruta_archivo =r"/home/to-o/practicasProfeeionales\Prácticas Profesionales\JSON\Resultado_Homogenizado.xlsx"
    features = ['eToday (kWh)', 'eTotal (kWh)', 'power (kW)',
                'wind_speed_100m', 'wind_speed_10m',
                'wind_direction_10m', "relative_humidity"]

    # Cargar datos
    data, target, features = cargar_datos(ruta_archivo, features=features)

    # Dividir datos en train y test (últimos 30 días para test)
    periodo_test = '30D'
    test_data = data.last(periodo_test)
    train_data = data.iloc[:-len(test_data)]

    # Preparar datos para Prophet
    train_df = preparar_dataframe_prophet(train_data, target, features)
    test_df = preparar_dataframe_prophet(test_data, target, features)

    # Entrenar modelo
    modelo_prophet = entrenar_prophet(train_df, features)

    # Predecir
    y_pred = predecir_prophet(modelo_prophet, test_df, features)

    y_true = test_df['y'].values

    # Calcular métricas
    rmse, mae, r2 = calcular_metricas(y_true, y_pred)

    print(f"Evaluación modelo Prophet con regresores para {periodo_test}:")
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")


    dias_totales = (data.index.max() - data.index.min()).days
    print(f"Días totales en el dataset: {dias_totales}")


    # Validación cruzada para evaluar el modelo a lo largo del tiempo
    print("\nEjecutando validación cruzada Prophet (puede tardar)...")
    df_cv, df_p = validacion_cruzada_prophet(modelo_prophet, train_df, initial='40 days', period='20 days', horizon='10 days')

    if df_cv is not None and df_p is not None:
       print("\nMétricas de validación cruzada (RMSE y MAE de performance_metrics):")
       print(df_p[['horizon', 'rmse', 'mae']].groupby('horizon').mean())

    # Crear la columna 'horizon' en días
       df_cv['horizon'] = (df_cv['ds'] - df_cv['cutoff']).dt.days

    # Agrupar por horizon y calcular métricas manualmente
       df_r2 = df_cv.groupby('horizon').apply(
        lambda x: pd.Series({
            'rmse': np.sqrt(np.mean((x['y'] - x['yhat'])**2)),
            'mae': np.mean(np.abs(x['y'] - x['yhat'])),
            'r2': r2_score(x['y'], x['yhat'])
        })
    )
    print("\nMétricas de validación cruzada con R2 calculado manualmente:")
    print(df_r2)


    print(f"Periodo de entrenamiento: {train_data.index.min().date()} a {train_data.index.max().date()}")
    print(f"Periodo de prueba: {test_data.index.min().date()} a {test_data.index.max().date()}")
    print(f"Total de datos: {len(data)}")
    print(f"Datos de entrenamiento: {len(train_data)}, Datos de prueba: {len(test_data)}")
    graficar_comparacion(train_df, test_df, features)

     # Calcular porcentaje de coincidencias exactas o cercanas
    tolerancia = 0.05  # 5% de tolerancia
    coincidencias = np.isclose(y_true, y_pred, rtol=tolerancia)
    porcentaje_coincidencia = 100 * np.sum(coincidencias) / len(y_true)

    print(f"Coincidencias dentro del {tolerancia*100:.0f}% de tolerancia: {np.sum(coincidencias)} de {len(y_true)}")
    print(f"Porcentaje de coincidencia aproximada: {porcentaje_coincidencia:.2f}%")
       
   
if __name__ == "__main__":
    main()
