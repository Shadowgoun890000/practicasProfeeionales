import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Cargar los datos
file_path = r"C:\Prácticas Profesionales\JSON\Resultado_Homogenizado.xlsx"
data = pd.read_excel(file_path)

# Preprocesamiento de fechas
data['fecha_hora'] = pd.to_datetime(data['fecha_hora'])
data.set_index('fecha_hora', inplace=True)

# Agregar variables temporales
data['hora'] = data.index.hour
data['dia_semana'] = data.index.dayofweek
data['es_fin_semana'] = data['dia_semana'].isin([5, 6]).astype(int)

# Definir variables
objetivo = 'valor (kWh)'
features = ['eToday (kWh)', 'eTotal (kWh)', 'power (kW)',
            'wind_speed_100m', 'wind_speed_10m', 'wind_direction_10m',
            'relative_humidity', 'hora', 'dia_semana', 'es_fin_semana']
features = [col for col in features if col in data.columns]

# Funciones de métricas
def calcular_métricas(y_real, y_pred):
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    return rmse, mae, r2

# Periodos de prueba
periodos = {
    "1_mes": "30D",
    "3_dias": "3D",
    "3_semanas": "21D",
    "3_horas": "3H"
}

resultados = {}

for nombre, periodo in periodos.items():
    print(f"\nEvaluando periodo: {nombre}")
    test_data = data.last(periodo)
    train_data = data.iloc[:-len(test_data)]

    # ARIMA
    arima_model = ARIMA(train_data[objetivo], order=(5,1,0))
    arima_model_fit = arima_model.fit()
    arima_pred = arima_model_fit.forecast(steps=len(test_data))

    # Prophet con regresores si es posible
    prophet_data = train_data.reset_index().rename(columns={'fecha_hora': 'ds', objetivo: 'y'})
    usar_regresores = 'power (kW)' in train_data.columns
    if usar_regresores:
        prophet_data['power'] = train_data['power (kW)'].values

    prophet_model = Prophet()
    if usar_regresores:
        prophet_model.add_regressor('power')

    prophet_model.fit(prophet_data)
    future = prophet_model.make_future_dataframe(periods=len(test_data), freq='5T')
    if usar_regresores:
        future['power'] = data['power (kW)'].iloc[:len(future)].values

    forecast = prophet_model.predict(future)
    prophet_forecast = forecast['yhat'].iloc[-len(test_data):]

    # Random Forest
    X_train, X_test = train_data[features], test_data[features]
    y_train, y_test = train_data[objetivo], test_data[objetivo]
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    # Guardar métricas
    resultados[nombre] = {
        "ARIMA": calcular_métricas(test_data[objetivo], arima_pred),
        "Prophet": calcular_métricas(test_data[objetivo], prophet_forecast),
        "RandomForest": calcular_métricas(y_test, rf_pred)
    }

    # Graficar resultados
    modelos = {'ARIMA': arima_pred, 'Prophet': prophet_forecast, 'RandomForest': rf_pred}
    for modelo, prediccion in modelos.items():
        plt.figure(figsize=(12, 5))
        plt.plot(test_data.index, test_data[objetivo], label='Real', color='black', linewidth=2)
        plt.plot(test_data.index, prediccion, label=modelo, linestyle='--')
        plt.title(f'{modelo} - Predicción para {nombre}')
        plt.legend()
        plt.grid(True)
        plt.show()

# Mostrar resultados
print("\nResultados por periodo:")
for periodo, metricas in resultados.items():
    print(f"\n{periodo}:")
    for modelo, valores in metricas.items():
        print(f"{modelo} - RMSE: {valores[0]:.2f}, MAE: {valores[1]:.2f}, R2: {valores[2]:.2f}")

# Resultados promedio
print("\nResultados promedio por modelo:")
promedios = {}
for modelo in ['ARIMA', 'Prophet', 'RandomForest']:
    modelo_vals = [resultados[periodo][modelo] for periodo in resultados]
    promedio = np.mean(modelo_vals, axis=0)
    promedios[modelo] = promedio
    print(f"{modelo} - RMSE: {promedio[0]:.2f}, MAE: {promedio[1]:.2f}, R2: {promedio[2]:.2f}")
