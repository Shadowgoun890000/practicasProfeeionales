import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
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

# Crear conjuntos de entrenamiento, validación y prueba
train, temp = train_test_split(data, test_size=0.3, shuffle=False)  # 70% entrenamiento, 30% temporal (validación + test)
val, test = train_test_split(temp, test_size=0.5, shuffle=False)  # 50% de la parte temporal para validación y 50% para prueba

# Extraer las series temporales de entrenamiento, validación y prueba
train_series = train[target]
val_series = val[target]
test_series = test[target]

# Análisis Exploratorio de Datos (EDA)
print("Estadísticas descriptivas de las variables:")
print(data.describe())

# Distribución de la generación de energía y consumo
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
sns.histplot(data[target], kde=True)
plt.title(f'Distribución de {target}')

# Correlación entre las variables
correlation_matrix = data[features + [target]].corr()
plt.subplot(2, 1, 2)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación')
plt.tight_layout()
plt.show()

# Modelos a probar
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

# Función para calcular métricas de evaluación
def calcular_métricas(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# Calcular métricas para cada modelo en el conjunto de validación
arima_metrics = calcular_métricas(val_series, arima_pred)
prophet_metrics = calcular_métricas(val_series, prophet_forecast)
rf_metrics = calcular_métricas(y_val, rf_pred)

# Resultados de las métricas en el conjunto de validación
print("Métricas de ARIMA en Validación - RMSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}".format(*arima_metrics))
print("Métricas de Prophet en Validación - RMSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}".format(*prophet_metrics))
print("Métricas de Random Forest en Validación - RMSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}".format(*rf_metrics))

# Evaluación final en el conjunto de prueba (Test)
# 1. ARIMA (usando el mejor modelo obtenido)
arima_pred_test = arima_model_fit.forecast(steps=len(test_series))  # Predicciones en el conjunto de prueba

# 2. Prophet (modelo de series temporales con estacionalidad)
prophet_forecast_test = prophet_model.predict(prophet_model.make_future_dataframe(periods=len(test_series), freq='5T'))['yhat'].iloc[-len(test_series):]

# 3. Random Forest Regressor (modelo supervisado con características adicionales)
X_test = test[features]
y_test = test[target]
rf_pred_test = rf_model.predict(X_test)

# Calcular métricas para cada modelo en el conjunto de prueba
arima_metrics_test = calcular_métricas(test_series, arima_pred_test)
prophet_metrics_test = calcular_métricas(test_series, prophet_forecast_test)
rf_metrics_test = calcular_métricas(y_test, rf_pred_test)

# Resultados de las métricas en el conjunto de prueba
print("\nMétricas de ARIMA en Test - RMSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}".format(*arima_metrics_test))
print("Métricas de Prophet en Test - RMSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}".format(*prophet_metrics_test))
print("Métricas de Random Forest en Test - RMSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}".format(*rf_metrics_test))

# Graficar los resultados de prueba
plt.figure(figsize=(12, 8))

# Gráfico de las series temporales reales y las predicciones
plt.subplot(3, 1, 1)
plt.plot(test_series.index, test_series, label='Real', color='black')
plt.plot(test_series.index, arima_pred_test, label='ARIMA', color='blue')
plt.title('Predicción de ARIMA')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(test_series.index, test_series, label='Real', color='black')
plt.plot(test_series.index, prophet_forecast_test, label='Prophet', color='green')
plt.title('Predicción de Prophet')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(test_series.index, test_series, label='Real', color='black')
plt.plot(test_series.index, rf_pred_test, label='Random Forest', color='red')
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
