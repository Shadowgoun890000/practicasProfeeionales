import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Cargar el archivo y configurar la columna de fecha
file_path = r"C:\Prácticas Profesionales\JSON\Resultado_Homogenizado.xlsx"
data = pd.read_excel(file_path)

data['fecha_hora'] = pd.to_datetime(data['fecha_hora'])
data.set_index('fecha_hora', inplace=True)

# Selección de la columna de objetivo y características específicas para Saltillo
target = 'valor (kWh)'

# Características meteorológicas seleccionadas que se alinean con Saltillo
features = [ 
    'eToday (kWh)', 'eTotal (kWh)', 'power (kW)',
    'wind_speed_100m', 'wind_speed_10m', 
    'wind_direction_10m', "relative_humidity"
]

# Asegurarse de que las características estén presentes en los datos
features = [col for col in features if col in data.columns]

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)  # 5 splits para Time Series Cross-Validation
arima_metrics = []
prophet_metrics = []
rf_metrics = []

# Función para calcular métricas de evaluación
def calcular_métricas(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# Time Series Cross-Validation
for train_index, val_index in tscv.split(data):
    # Dividir los datos en conjuntos de entrenamiento y validación
    train, val = data.iloc[train_index], data.iloc[val_index]

    # Extraer las series temporales de entrenamiento y validación
    train_series = train[target]
    val_series = val[target]

    # 1. ARIMA (utiliza solo la serie temporal univariante)
    arima_model = ARIMA(train_series, order=(5,1,0))
    arima_model_fit = arima_model.fit()
    arima_pred = arima_model_fit.forecast(steps=len(val_series))  # Predicciones en el conjunto de validación

    # 2. Prophet (modelo de series temporales con estacionalidad)
    prophet_data = train.reset_index().rename(columns={'fecha_hora': 'ds', target: 'y'})
    prophet_model = Prophet()
    prophet_model.fit(prophet_data)
    future = prophet_model.make_future_dataframe(periods=len(val_series), freq='5T')  # Predicción en validación
    prophet_forecast = prophet_model.predict(future)['yhat'].iloc[-len(val_series):]

    # 3. Random Forest Regressor (modelo supervisado con características adicionales)
    X_train, X_val = train[features], val[features]
    y_train, y_val = train[target], val[target]
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_val)

    # Calcular métricas para cada modelo en el conjunto de validación
    arima_metrics.append(calcular_métricas(val_series, arima_pred))
    prophet_metrics.append(calcular_métricas(val_series, prophet_forecast))
    rf_metrics.append(calcular_métricas(y_val, rf_pred))

# Promedio de las métricas obtenidas en cada fold
arima_avg_metrics = np.mean(arima_metrics, axis=0)
prophet_avg_metrics = np.mean(prophet_metrics, axis=0)
rf_avg_metrics = np.mean(rf_metrics, axis=0)

# Resultados de las métricas promedio de Time Series Cross-Validation
print("Promedio de métricas de ARIMA en Cross-Validation - RMSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}".format(*arima_avg_metrics))
print("Promedio de métricas de Prophet en Cross-Validation - RMSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}".format(*prophet_avg_metrics))
print("Promedio de métricas de Random Forest en Cross-Validation - RMSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}".format(*rf_avg_metrics))

# Evaluación final en el conjunto completo de datos (test)
# Usar el conjunto completo para entrenar los modelos y obtener las métricas finales

# 1. ARIMA (usando el mejor modelo obtenido)
arima_model_final = ARIMA(data[target], order=(5,1,0))
arima_model_fit_final = arima_model_final.fit()
arima_pred_final = arima_model_fit_final.fittedvalues

# 2. Prophet (modelo de series temporales con estacionalidad)
prophet_data_final = data.reset_index().rename(columns={'fecha_hora': 'ds', target: 'y'})
prophet_model_final = Prophet()
prophet_model_final.fit(prophet_data_final)
future_final = prophet_model_final.make_future_dataframe(periods=0, freq='5T')  # Predicción en todos los datos
prophet_forecast_final = prophet_model_final.predict(future_final)['yhat']

# 3. Random Forest Regressor (modelo supervisado con características adicionales)
X_full = data[features]
y_full = data[target]
rf_model_final = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_final.fit(X_full, y_full)
rf_pred_final = rf_model_final.predict(X_full)

# Calcular métricas para cada modelo en el conjunto completo
arima_metrics_final = calcular_métricas(data[target], arima_pred_final)
prophet_metrics_final = calcular_métricas(data[target], prophet_forecast_final)
rf_metrics_final = calcular_métricas(data[target], rf_pred_final)

# Resultados de las métricas finales
print("\nMétricas finales de ARIMA en todo el conjunto de datos - RMSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}".format(*arima_metrics_final))
print("Métricas finales de Prophet en todo el conjunto de datos - RMSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}".format(*prophet_metrics_final))
print("Métricas finales de Random Forest en todo el conjunto de datos - RMSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}".format(*rf_metrics_final))

# Graficar los resultados finales
plt.figure(figsize=(12, 8))

# Gráfico de las series temporales reales y las predicciones
plt.subplot(3, 1, 1)
plt.plot(data.index, data[target], label='Real', color='black')
plt.plot(data.index, arima_pred_final, label='ARIMA', color='blue')
plt.title('Predicción de ARIMA')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(data.index, data[target], label='Real', color='black')
plt.plot(data.index, prophet_forecast_final, label='Prophet', color='green')
plt.title('Predicción de Prophet')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(data.index, data[target], label='Real', color='black')
plt.plot(data.index, rf_pred_final, label='Random Forest', color='red')
plt.title('Predicción de Random Forest')
plt.legend()

plt.tight_layout()
plt.show()

# Verificar correlaciones con la variable objetivo
correlaciones = data[features + [target]].corr()[target]
print("Correlaciones con la variable objetivo:")
print(correlaciones)

# Verificar constantes (varianza igual a 0)
varianzas = data[features].var()
constantes = varianzas[varianzas == 0].index.tolist()
print("Variables constantes:", constantes)
