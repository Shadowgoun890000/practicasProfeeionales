import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Cargar el archivo y configurar la columna de fecha
file_path = r"C:\Prácticas Profesionales\JSON\Resultado_Homogenizado.xlsx"
data = pd.read_excel(file_path)
data['fecha_hora'] = pd.to_datetime(data['fecha_hora'])
data.set_index('fecha_hora', inplace=True)

# Variables
target = 'valor (kWh)'
features = [
    'eToday (kWh)', 'eTotal (kWh)', 'power (kW)',
    'wind_speed_100m', 'wind_speed_10m',
    'wind_direction_10m', "relative_humidity"
]
features = [col for col in features if col in data.columns]

# Función para calcular métricas
def calcular_métricas(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# Definición de periodos a evaluar
periodos = {
    "3_horas": "3H",
    "3_dias": "3D",
    "3_semanas": "21D",
    "1_mes": "30D"
}

resultados = {}

for nombre, periodo in periodos.items():
    print(f"\nEvaluando periodo: {nombre}")
    test_data = data.last(periodo)
    train_data = data.iloc[:-len(test_data)]

    # Series para modelos ARIMA y Prophet
    train_series = train_data[target]
    test_series = test_data[target]

    # ARIMA
    arima_model = ARIMA(train_series, order=(5,1,0))
    arima_model_fit = arima_model.fit()
    arima_pred = arima_model_fit.forecast(steps=len(test_data))

    # Prophet
    prophet_train = train_data.reset_index().rename(columns={'fecha_hora': 'ds', target: 'y'})
    prophet_model = Prophet()
    prophet_model.fit(prophet_train)
    future = prophet_model.make_future_dataframe(periods=len(test_data), freq='5T')
    prophet_forecast = prophet_model.predict(future)['yhat'].iloc[-len(test_data):]

    # Random Forest
    if all(f in train_data.columns for f in features):
        X_train, X_test = train_data[features], test_data[features]
        y_train, y_test = train_data[target], test_data[target]
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
    else:
        rf_pred = np.full(len(test_data), np.nan)
        y_test = test_data[target]

    # Guardar métricas
    resultados[nombre] = {
        "ARIMA": calcular_métricas(test_series, arima_pred),
        "Prophet": calcular_métricas(test_series, prophet_forecast),
        "RandomForest": calcular_métricas(y_test, rf_pred)
    }

    # Graficar predicciones
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(test_series.index, test_series, label='Real', color='black')
    plt.plot(test_series.index, arima_pred, label='ARIMA', color='blue')
    plt.title(f'Predicción ARIMA - {nombre}')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(test_series.index, test_series, label='Real', color='black')
    plt.plot(test_series.index, prophet_forecast, label='Prophet', color='green')
    plt.title(f'Predicción Prophet - {nombre}')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(test_series.index, test_series, label='Real', color='black')
    plt.plot(test_series.index, rf_pred, label='Random Forest', color='red')
    plt.title(f'Predicción Random Forest - {nombre}')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Mostrar métricas para todos los periodos
for periodo, metricas in resultados.items():
    print(f"\nResultados para {periodo}:")
    for modelo, valores in metricas.items():
        print(f"{modelo} - RMSE: {valores[0]:.2f}, MAE: {valores[1]:.2f}, R2: {valores[2]:.2f}")
